# e_training/models/hybrid_cnn_transformer.py
from __future__ import annotations
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding (no learnable params)."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class ConvBlock(nn.Module):
    """1D conv block: Conv1d -> BN -> GELU -> Dropout"""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        return self.net(x)


class HybridCNNTransformer(nn.Module):
    """
    CNN (2–3 blocks) → linear proj to d_model → (optional CLS) → (optional PE)
    → TransformerEncoder (2–4 layers) → (optional LSTM) → pool(time) → per-horizon MLP heads.

    Pooling options:
      - "gap": global average pooling over time
      - "gmp": global max pooling over time
      - "cls": uses a learnable [CLS] token and takes the first output vector
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

        # New/optional knobs to match your config style
        ff_mult: int = 4,
        pool: str = "gap",          # "gap" | "gmp" | "cls"
        posenc: str = "sin",        # "sin" | "none"
        use_lstm: bool = False,
        lstm_hidden: Optional[int] = None,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        ln_eps: float = 1e-6,
    ):
        super().__init__()
        assert len(cnn_channels) == len(cnn_kernels), "cnn_channels and cnn_kernels length mismatch"
        assert pool in {"gap", "gmp", "cls"}, f"Invalid pool='{pool}'"
        self.pool_mode = pool
        self.use_lstm = use_lstm
        self.horizons = horizons

        # ---------------- CNN tower ----------------
        C_in = input_dim
        convs = []
        for C_out, K in zip(cnn_channels, cnn_kernels):
            convs.append(ConvBlock(C_in, C_out, K, cnn_dropout))
            C_in = C_out
        self.cnn = nn.Sequential(*convs)

        # ---------------- Proj to d_model ----------------
        # CNN output is [B, C', L] → transpose to [B, L, C'] then Linear(C'→d_model)
        self.proj = nn.Linear(C_in, d_model, bias=False)

        # ---------------- Positional encoding + CLS ----------------
        self.posenc = PositionalEncoding(d_model) if posenc == "sin" else nn.Identity()
        self.use_cls = (pool == "cls")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_cls else None

        # ---------------- Transformer encoder ----------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=attn_dropout,             # attn + ffn dropout inside the layer
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
            norm=nn.LayerNorm(d_model, eps=ln_eps),
        )
        self.post_enc_dropout = nn.Dropout(ffn_dropout)

        # ---------------- Optional LSTM refinement ----------------
        if self.use_lstm:
            self.lstm_hidden = lstm_hidden or d_model
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=self.lstm_hidden,
                num_layers=lstm_layers,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=False,
            )
            head_in = self.lstm_hidden
        else:
            head_in = d_model

        # ---------------- Pooling heads ----------------
        if self.pool_mode == "gap":
            self.pool_gap = nn.AdaptiveAvgPool1d(1)
        elif self.pool_mode == "gmp":
            self.pool_gmp = nn.AdaptiveMaxPool1d(1)

        # ---------------- Per-horizon MLP heads (includes FC) ----------------
        # This is your fully-connected head: Linear → GELU → Dropout → Linear(→1)
        Hhid = head_hidden or head_in
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.LayerNorm(head_in, eps=ln_eps),
                nn.Linear(head_in, Hhid),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(Hhid, 1),
            )
            for h in horizons
        })

        # Init a bit
        nn.init.xavier_uniform_(self.proj.weight)
        if self.use_cls:
            nn.init.zeros_(self.cls_token)

    def _apply_pool(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: [B, L, D_or_H]
        returns: [B, D_or_H]
        """
        if self.pool_mode == "cls":
            # first position is CLS by construction
            return z_seq[:, 0, :]
        # GAP/GMP operate over time dimension
        z = z_seq.transpose(1, 2)  # [B, D, L]
        if self.pool_mode == "gap":
            return self.pool_gap(z).squeeze(-1)  # [B, D]
        else:
            return self.pool_gmp(z).squeeze(-1)  # [B, D]

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x: [B, L, F]
        returns: {h: [B]} predictions in scaled space
        """
        B, L, F = x.shape

        # CNN expects [B, C, L] with channels=C=features
        z = x.transpose(1, 2)          # [B, F, L]
        z = self.cnn(z)                 # [B, C', L]

        # back to [B, L, C']
        z = z.transpose(1, 2)           # [B, L, C']

        # project to d_model
        z = self.proj(z)                # [B, L, d_model]

        # (optional) prepend CLS token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            z = torch.cat([cls, z], dim=1)          # [B, L+1, D]

        # positional encoding
        z = self.posenc(z)              # [B, L(+1), d_model]

        # Transformer encoder
        z = self.encoder(z)             # [B, L(+1), d_model]
        z = self.post_enc_dropout(z)

        # Optional LSTM refinement
        if self.use_lstm:
            z, _ = self.lstm(z)         # [B, L(+1), H]

        # Pool over time (or CLS)
        h = self._apply_pool(z)         # [B, D/H]

        # Parallel per-horizon heads (this is where your FC lives)
        out: Dict[int, torch.Tensor] = {}
        for hname, head in self.heads.items():
            y = head(h).squeeze(-1)     # [B]
            out[int(hname)] = y
        return out