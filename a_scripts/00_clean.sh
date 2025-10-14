#!/usr/bin/env bash
set -e
echo "[00_clean] Removing old artifacts (seq, features, models, reports)"
rm -rf experiments/artifacts/seq
rm -rf experiments/artifacts/features_scaled_ps
rm -rf experiments/artifacts/features_scaled
rm -rf experiments/artifacts/models
rm -rf experiments/artifacts/reports