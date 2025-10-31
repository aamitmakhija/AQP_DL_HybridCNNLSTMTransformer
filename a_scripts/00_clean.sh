#!/usr/bin/env bash
# ============================================================
# 00_clean.sh
# ------------------------------------------------------------
# Purpose:
#   Cleans previous experiment artifacts to ensure a fresh run.
#   Removes old sequence windows, scaled features, model checkpoints,
#   and evaluation reports that can cause mismatched state_dicts
#   or outdated data.
#
# Usage:
#   ./a_scripts/00_clean.sh                # standard clean
#   ./a_scripts/00_clean.sh --dry-run      # show what would be removed
#   ./a_scripts/00_clean.sh --deep         # also remove features, splits, locks
#   ./a_scripts/00_clean.sh --models-only  # only wipe models + reports
#   ./a_scripts/00_clean.sh --relock       # only remove feature lock
#
# Safe to run multiple times.
# ============================================================

set -euo pipefail

# ---------- CLI flags ----------
DRY_RUN=false
DEEP=false
MODELS_ONLY=false
RELOCK=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY_RUN=true ;;
    --deep)         DEEP=true ;;
    --models-only)  MODELS_ONLY=true ;;
    --relock)       RELOCK=true ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# ---------- Paths ----------
ART_DIR="experiments/artifacts"

# ---------- Removal groups ----------
# Standard clean: fast and safe; clears regenerated artifacts.
STANDARD_REMOVE=(
  "${ART_DIR}/seq"
  "${ART_DIR}/features_scaled_ps"
  "${ART_DIR}/features_scaled"
  "${ART_DIR}/models"
  "${ART_DIR}/reports"
  "${ART_DIR}/checkpoints"
  "${ART_DIR}/duplicates_summary.json"
  "${ART_DIR}/split_summary.json"
)

# Deep clean: includes features, splits, dataset stream, and locks.
DEEP_EXTRA_REMOVE=(
  "${ART_DIR}/features"
  "${ART_DIR}/features_locked"
  "${ART_DIR}/splits"
  "${ART_DIR}/dataset_stream"
)

# Models-only: just models and reports.
MODELS_ONLY_REMOVE=(
  "${ART_DIR}/models"
  "${ART_DIR}/reports"
  "${ART_DIR}/checkpoints"
)

# Relock: minimal mode—only removes the feature lock folder.
RELOCK_REMOVE=(
  "${ART_DIR}/features_locked"
)

# ---------- Helpers ----------
say() { printf "%s\n" "$*"; }
exists() { [[ -e "$1" ]]; }
rm_path() {
  local p="$1"
  if exists "$p"; then
    if $DRY_RUN; then
      say " - would remove: $p"
    else
      rm -rf "$p"
      say " - removed:     $p"
    fi
  else
    say " - not found:    $p"
  fi
}

# ---------- Header ----------
say "****************************************"
say "[00_clean] Starting cleanup"
say "  repo: $(pwd)"
say "  artifacts root: ${ART_DIR}"
$DRY_RUN && say "  mode: DRY RUN (no files will be deleted)"
$DEEP && say "  mode: DEEP CLEAN (features/splits/locks will be removed)"
$MODELS_ONLY && say "  mode: MODELS ONLY (only models/reports/checkpoints)"
$RELOCK && say "  mode: RELOCK (only removes feature lock)"
say "****************************************"

# ---------- Selection ----------
TO_REMOVE=()

if $MODELS_ONLY; then
  TO_REMOVE=("${MODELS_ONLY_REMOVE[@]}")
elif $RELOCK; then
  TO_REMOVE=("${RELOCK_REMOVE[@]}")
else
  TO_REMOVE=("${STANDARD_REMOVE[@]}")
  if $DEEP; then
    TO_REMOVE+=("${DEEP_EXTRA_REMOVE[@]}")
  fi
fi

# ---------- Do the work ----------
say "[00_clean] Removing artifacts:"
for p in "${TO_REMOVE[@]}"; do
  rm_path "$p"
done

# Recreate necessary dirs
RECREATE_DIRS=(
  "${ART_DIR}/models"
  "${ART_DIR}/reports"
)

if $DRY_RUN; then
  for d in "${RECREATE_DIRS[@]}"; do
    say " - would create: $d"
  done
else
  for d in "${RECREATE_DIRS[@]}"; do
    mkdir -p "$d"
    say " - created:      $d"
  done
fi

say "****************************************"
say "[00_clean] Done."
$DRY_RUN || {
  say "[00_clean] Fresh directories ready:"
  say "           → ${ART_DIR}/models"
  say "           → ${ART_DIR}/reports"
}
say "****************************************"