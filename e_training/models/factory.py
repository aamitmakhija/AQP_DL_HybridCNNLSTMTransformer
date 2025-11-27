# e_training/models/factory.py
from __future__ import annotations
from typing import Any


def build_hybrid(
    mh: dict,
    *,
    input_dim: int,
    has_sid: bool,
    n_stations: int | None,
    horizon: int,
) -> Any:
    """
    Build hybrid model variants based on model.hybrid.impl:

      - "hybrid_cnn_transformer" (default)
      - "hybrid_cnn"
      - "cnn_only"
      - "lstm_only"
      - "transformer_only"
      - "legacy_hybrid_encoder" / "legacy"
    """

    impl = str(mh.get("impl", "hybrid_cnn_transformer")).lower()

    # ------------------------------------------------------------
    # TRANSFORMER ONLY — PURE TRANSFORMER ENCODER + HEAD
    # ------------------------------------------------------------
    if impl == "transformer_only":
        from e_training.models.hybrid_cnn_transformer import HybridCNNTransformer

        return HybridCNNTransformer(
            input_dim=input_dim,
            cnn_channels=[],                # disable CNN entirely
            cnn_kernels=[],
            cnn_dropout=0.0,

            d_model=int(mh["d_model"]),
            nhead=int(mh["nhead"]),
            num_layers=int(mh["num_layers"]),    # transformer depth
            ff_mult=int(mh["ff_mult"]),
            attn_dropout=float(mh["attn_dropout"]),
            ffn_dropout=float(mh["ffn_dropout"]),

            horizons=[int(horizon)],
            pool=str(mh.get("pool", "gap")),
            posenc=str(mh.get("posenc", "sin")),

            num_stations=0,
            station_embed_dim=0,

            use_lstm=False,
            lstm_hidden=None,
            lstm_layers=0,
            lstm_dropout=0.0,

            ln_eps=float(mh.get("ln_eps", 1e-6)),
        )

    # ------------------------------------------------------------
    # MAIN HYBRID BUILDER (CNN, LSTM, Transformer mixes)
    # ------------------------------------------------------------
    if impl in {"hybrid_cnn_transformer", "hybrid_cnn", "cnn_only", "lstm_only"}:
        from e_training.models.hybrid_cnn_transformer import HybridCNNTransformer

        # YAML-defined transformer depth
        num_layers = int(mh["num_layers"])
        use_lstm = bool(mh.get("use_lstm", False))

        # Variant overrides
        if impl == "cnn_only":
            num_layers = 0         # disable Transformer
            use_lstm = False
        elif impl == "lstm_only":
            num_layers = 0         # disable Transformer
            use_lstm = True
        elif impl == "hybrid_cnn":
            # CNN + Transformer, no LSTM
            use_lstm = False

        # Station embedding logic
        use_station_embedding = bool(mh.get("use_station_embedding", False))
        if use_station_embedding and has_sid and (n_stations or 0) > 0:
            num_stations_arg = int(n_stations or 0)
            station_embed_dim = int(mh.get("station_embed_dim", 32))
        else:
            num_stations_arg = 0
            station_embed_dim = 0

        return HybridCNNTransformer(
            input_dim=input_dim,
            cnn_channels=list(mh["cnn_channels"]),
            cnn_kernels=list(mh["cnn_kernels"]),
            cnn_dropout=float(mh["cnn_dropout"]),

            d_model=int(mh["d_model"]),
            nhead=int(mh["nhead"]),
            num_layers=num_layers,
            ff_mult=int(mh["ff_mult"]),
            attn_dropout=float(mh["attn_dropout"]),
            ffn_dropout=float(mh["ffn_dropout"]),

            horizons=[int(horizon)],
            pool=str(mh.get("pool", "gap")),
            posenc=str(mh.get("posenc", "sin")),

            num_stations=num_stations_arg,
            station_embed_dim=station_embed_dim,

            use_lstm=use_lstm,
            lstm_hidden=mh.get("lstm_hidden", None),
            lstm_layers=int(mh.get("lstm_layers", 1)),
            lstm_dropout=float(mh.get("lstm_dropout", 0.0)),

            ln_eps=float(mh.get("ln_eps", 1e-6)),
        )

    # ------------------------------------------------------------
    # LEGACY ENCODER
    # ------------------------------------------------------------
    if impl in {"legacy_hybrid_encoder", "legacy"}:
        from e_training.models.hybrid_encoder_legacy import HybridEncoder

        use_station_embedding = bool(mh.get("use_station_embedding", False))
        use_station_embedding = (
            use_station_embedding and has_sid and (n_stations or 0) > 0
        )

        n_stations_eff = int(n_stations or 0) if use_station_embedding else 0
        station_dim_eff = (
            int(mh.get("station_embed_dim", 0)) if use_station_embedding else 0
        )

        return HybridEncoder(
            in_dim=input_dim,
            cnn_channels=list(mh["cnn_channels"]),
            cnn_kernels=list(mh["cnn_kernels"]),
            cnn_dropout=float(mh["cnn_dropout"]),

            d_model=int(mh["d_model"]),
            nhead=int(mh["nhead"]),
            num_layers=int(mh["num_layers"]),
            ff_mult=int(mh["ff_mult"]),
            attn_dropout=float(mh["attn_dropout"]),
            ffn_dropout=float(mh["ffn_dropout"]),

            pool=str(mh.get("pool", "gap")),
            posenc=str(mh.get("posenc", "sin")),

            n_stations=n_stations_eff,
            station_embed_dim=station_dim_eff,
        )

    # ------------------------------------------------------------
    # INVALID OPTION
    # ------------------------------------------------------------
    raise ValueError(f"Unknown model.hybrid.impl={impl}")