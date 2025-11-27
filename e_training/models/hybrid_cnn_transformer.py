from __future__ import annotations
from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sine/cosine positional encoding (no trainable params)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # [1, L, D]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, p_drop: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2, bias=True)
        try:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        except ValueError:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        return self.drop(x)


class HybridCNNTransformer(nn.Module):
    """
    x: [B, L, F]
    Optional station ids: sid_idx: [B] (long) if station embeddings enabled.
    Returns: dict mapping horizon H -> [B] predictions.
    """

    def __init__(
        self,
        input_dim: int,
        cnn_channels: List[int],
        cnn_kernels: List[int],
        cnn_dropout: float,
        d_model: int,
        nhead: int,
        num_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        horizons: List[int],
        head_hidden: Optional[int] = None,
        ff_mult: int = 4,
        pool: str = "gap",          # {"gap", "gmp", "cls"}
        posenc: str = "sin",        # {"sin", "none"}
        use_lstm: bool = False,
        lstm_hidden: Optional[int] = None,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        ln_eps: float = 1e-6,
        pe_max_len: int = 4096,
        # optional station embedding
        num_stations: Optional[int] = None,
        station_embed_dim: int = 16,
    ):
        super().__init__()
        assert len(cnn_channels) == len(
            cnn_kernels
        ), "cnn_channels and cnn_kernels must match"
        assert pool in {"gap", "gmp", "cls"}

        self.pool_mode = pool
        self.use_lstm = use_lstm
        self.horizons = [int(h) for h in horizons]
        self.pe_max_len = int(pe_max_len)
        self.num_layers = int(num_layers)

        # ---------------- CNN tower ----------------
        C_in = input_dim
        convs: List[nn.Module] = []
        for C_out, K in zip(cnn_channels, cnn_kernels):
            convs.append(ConvBlock(C_in, C_out, K, cnn_dropout))
            C_in = C_out
        self.cnn = nn.Sequential(*convs)

        # Project to transformer dim
        self.proj = nn.Linear(C_in, d_model, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

        # Positional enc + CLS
        self.posenc = (
            PositionalEncoding(d_model, max_len=self.pe_max_len)
            if posenc == "sin"
            else nn.Identity()
        )
        self.use_cls = pool == "cls"
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls else None
        )
        if self.use_cls:
            nn.init.zeros_(self.cls_token)

        # ---------------- Transformer encoder (optional) ----------------
        if self.num_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_mult * d_model,
                dropout=attn_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=self.num_layers,
                enable_nested_tensor=False,
                norm=nn.LayerNorm(d_model, eps=ln_eps),
            )
        else:
            self.encoder = None
        self.post_enc_dropout = nn.Dropout(ffn_dropout)

        # ---------------- Optional LSTM on sequence ----------------
        head_in = d_model
        if self.use_lstm:
            self.lstm_hidden = lstm_hidden or d_model
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=self.lstm_hidden,
                num_layers=lstm_layers,
                dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
                batch_first=True,
                bidirectional=False,
            )
            head_in = self.lstm_hidden

        # ---------------- Optional station embedding ----------------
        self.use_station_emb = (
            num_stations is not None and num_stations > 0 and station_embed_dim > 0
        )
        if self.use_station_emb:
            self.station_emb = nn.Embedding(num_stations, station_embed_dim)
            nn.init.normal_(self.station_emb.weight, mean=0.0, std=0.02)
            head_in = head_in + station_embed_dim

        # ---------------- Pool ----------------
        if self.pool_mode == "gap":
            self.pool_gap = nn.AdaptiveAvgPool1d(1)
        elif self.pool_mode == "gmp":
            self.pool_gmp = nn.AdaptiveMaxPool1d(1)

        # ---------------- Prediction heads (one per horizon) ----------------
        Hhid = head_hidden or head_in
        self.heads = nn.ModuleDict()
        for h in self.horizons:
            head = nn.Sequential(
                nn.LayerNorm(head_in, eps=ln_eps),
                nn.Linear(head_in, Hhid),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(Hhid, 1),
            )
            nn.init.xavier_uniform_(head[1].weight)
            nn.init.zeros_(head[1].bias)
            nn.init.xavier_uniform_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads[str(h)] = head

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _apply_pool(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: [B, L, D]
        if self.pool_mode == "cls":
            return z_seq[:, 0, :]  # [B, D]
        z = z_seq.transpose(1, 2)  # [B, D, L]
        if self.pool_mode == "gap":
            return self.pool_gap(z).squeeze(-1)  # [B, D]
        else:
            return self.pool_gmp(z).squeeze(-1)  # [B, D]

    @torch.no_grad()
    def _check_len(self, L: int) -> None:
        limit = self.pe_max_len - (1 if self.use_cls else 0)
        if L > limit:
            raise ValueError(
                f"Sequence length {L} exceeds pe_max_len={self.pe_max_len} "
                f"(pool='{self.pool_mode}'). Increase pe_max_len or shorten lookback."
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,  # [B, L, F]
        *,
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, L] (True=pad)
        attn_mask: Optional[torch.Tensor] = None,  # [L(+1), L(+1)] or [B,L(+1),L(+1)]
        sid_idx: Optional[torch.Tensor] = None,  # [B] if station embedding is used
    ) -> Dict[int, torch.Tensor]:
        B, L, _ = x.shape
        self._check_len(L)

        # CNN: [B, L, F] -> [B, C', L]
        z = x.transpose(1, 2)
        z = self.cnn(z)
        z = z.transpose(1, 2)  # [B, L, C']

        # Project to transformer / LSTM dim
        z = self.proj(z)  # [B, L, D]

        # CLS prepend if used
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            z = torch.cat([cls, z], dim=1)  # [B, L+1, D]
            if key_padding_mask is not None:
                pad0 = torch.zeros(
                    B,
                    1,
                    dtype=key_padding_mask.dtype,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat(
                    [pad0, key_padding_mask], dim=1
                )  # [B, L+1]

        # Positional encoding
        z = self.posenc(z)

        # Transformer encoder (optional)
        if self.encoder is not None and self.num_layers > 0:
            z = self.encoder(
                z,
                mask=attn_mask,
                src_key_padding_mask=key_padding_mask,
            )
        z = self.post_enc_dropout(z)

        # Optional LSTM on sequence
        if self.use_lstm:
            z, _ = self.lstm(z)

        # Pool → [B, D]
        feats = self._apply_pool(z)

        # Optional station embedding
        if self.use_station_emb:
            if sid_idx is None:
                raise ValueError(
                    "sid_idx must be provided when station embedding is enabled."
                )
            emb = self.station_emb(sid_idx)  # [B, E]
            feats = torch.cat([feats, emb], dim=-1)  # [B, D+E]

        # One head per horizon
        out: Dict[int, torch.Tensor] = {}
        for hname, head in self.heads.items():
            out[int(hname)] = head(feats).squeeze(-1)  # [B]
        return out