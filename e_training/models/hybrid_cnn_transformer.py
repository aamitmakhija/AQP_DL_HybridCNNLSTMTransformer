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
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max_len))
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
    CNN (2–3 blocks) → linear proj to d_model → positional encoding
    → TransformerEncoder (2–4 layers) → GAP over time → multi-head MLP heads.
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
    ):
        super().__init__()

        assert len(cnn_channels) == len(cnn_kernels), "cnn_channels and cnn_kernels length mismatch"
        C_in = input_dim
        convs = []
        for C_out, K in zip(cnn_channels, cnn_kernels):
            convs.append(ConvBlock(C_in, C_out, K, cnn_dropout))
            C_in = C_out
        self.cnn = nn.Sequential(*convs)

        # Project CNN channels → d_model (time stays as L)
        self.proj = nn.Linear(C_in, d_model, bias=False)
        self.posenc = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=attn_dropout,           # attention + MLP dropout inside layer
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.dropout_ffn = nn.Dropout(ffn_dropout)

        # Global average pool over time
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Per-horizon heads (direct multi-head)
        Hhid = head_hidden or d_model
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(d_model, Hhid),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(Hhid, 1),
            )
            for h in horizons
        })
        self.horizons = horizons

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x: [B, L, F]
        returns: {h: [B]} predictions in scaled space
        """
        B, L, F = x.shape

        # CNN expects [B, C, L] where C=features
        z = x.transpose(1, 2)          # [B, F, L]
        z = self.cnn(z)                 # [B, C', L]

        # back to [B, L, C']
        z = z.transpose(1, 2)           # [B, L, C']

        # project to d_model + PE + encoder
        z = self.proj(z)                # [B, L, d_model]
        z = self.posenc(z)              # [B, L, d_model]
        z = self.encoder(z)             # [B, L, d_model]
        z = self.dropout_ffn(z)

        # GAP over time
        z = z.transpose(1, 2)           # [B, d_model, L]
        z = self.gap(z).squeeze(-1)     # [B, d_model]

        # parallel heads
        out: Dict[int, torch.Tensor] = {}
        for h, head in self.heads.items():
            y = head(z).squeeze(-1)     # [B]
            out[int(h)] = y
        return out