
# Your current baseline (hybrid + per-station + full features)
bash a_scripts/run_preset.sh hybrid_full_station

# Try CNN-only, global scaling, lags-only
bash a_scripts/run_preset.sh cnn_lags_global

# Same as above but keep existing lock (not recommended when changing feature overlays)
SKIP_RELOCK=1 bash a_scripts/run_preset.sh cnn_lags_global