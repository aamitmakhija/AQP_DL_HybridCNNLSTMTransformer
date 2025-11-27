#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE="configs/cpu.yaml"
PRESET="${1:-}"; shift || true

case "$PRESET" in
  cnn_lags_global)
    OVS="configs/overlays/impl_cnn.yaml,configs/overlays/scale_global.yaml,configs/overlays/feat_lags_only.yaml" ;;
  lstm_lags_global)
    OVS="configs/overlays/impl_lstm.yaml,configs/overlays/scale_global.yaml,configs/overlays/feat_lags_only.yaml" ;;
  trans_lagsroll_station)
    OVS="configs/overlays/impl_transformer.yaml,configs/overlays/scale_station.yaml,configs/overlays/feat_lags_roll.yaml" ;;
  hybrid_full_station)
    OVS="configs/overlays/impl_hybrid.yaml,configs/overlays/scale_station.yaml,configs/overlays/feat_full.yaml" ;;
  *)
    OVS="$(printf "%s" "$*" | tr ' ' ',')"
    ;;
esac

export CONFIG="${BASE}${OVS:+,${OVS}}"

# Auto-relock so the lock matches the selected feature set
bash a_scripts/00_clean.sh --relock

# Full pipeline
bash a_scripts/run_all.sh