#!/usr/bin/env bash
# a_scripts/00_clean.sh
# Fresh cleanup of experiment artifacts (safe on macOS/Linux).

set -euo pipefail

# ---- options ----
DRY=false; DEEP=false; MODELS_ONLY=false; RELOCK=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY=true ;;
    --deep)         DEEP=true ;;
    --models-only)  MODELS_ONLY=true ;;
    --relock)       RELOCK=true ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# ---- roots (override via env if needed) ----
ART_DIR="${ART_DIR:-experiments/artifacts}"

say() { printf "%s\n" "$*"; }
exists() { [ -e "$1" ]; }
zap() {
  local p="$1"
  if exists "$p"; then
    $DRY && say " - would remove: $p" || { rm -rf "$p"; say " - removed:     $p"; }
  else
    say " - not found:    $p"
  fi
}

say "****************************************"
say "[00_clean] repo=$(pwd)"
say "[00_clean] artifacts=${ART_DIR}"
$DRY        && say "[00_clean] mode: DRY RUN"
$DEEP       && say "[00_clean] mode: DEEP"
$MODELS_ONLY&& say "[00_clean] mode: MODELS ONLY"
$RELOCK     && say "[00_clean] mode: RELOCK"
say "****************************************"

# ---- removal sets ----
STANDARD_REMOVE="
${ART_DIR}/seq
${ART_DIR}/features_scaled_ps
${ART_DIR}/features_scaled
${ART_DIR}/models
${ART_DIR}/models_mps
${ART_DIR}/reports
${ART_DIR}/checkpoints
${ART_DIR}/duplicates_summary.json
${ART_DIR}/split_summary.json
"

DEEP_EXTRA_REMOVE="
${ART_DIR}/features
${ART_DIR}/features_locked
${ART_DIR}/splits
${ART_DIR}/dataset_stream
"

MODELS_ONLY_REMOVE="
${ART_DIR}/models
${ART_DIR}/models_mps
${ART_DIR}/reports
${ART_DIR}/checkpoints
"

RELOCK_REMOVE="
${ART_DIR}/features_locked
"

# ---- choose set ----
TO_REMOVE="$STANDARD_REMOVE"
$DEEP        && TO_REMOVE="$TO_REMOVE
$DEEP_EXTRA_REMOVE"
$MODELS_ONLY && TO_REMOVE="$MODELS_ONLY_REMOVE"
$RELOCK      && TO_REMOVE="$RELOCK_REMOVE"

# ---- do work ----
say "[00_clean] Removing:"
# shellcheck disable=SC2086
for p in $TO_REMOVE; do
  [ -n "$p" ] && zap "$p"
done

# ---- recreate minimal dirs for fresh outputs ----
RECREATE="
${ART_DIR}/models
${ART_DIR}/models_mps
${ART_DIR}/reports
"
if $DRY; then
  for d in $RECREATE; do [ -n "$d" ] && say " - would create: $d"; done
else
  for d in $RECREATE; do [ -n "$d" ] && { mkdir -p "$d"; say " - created:      $d"; }; done
fi

say "****************************************"
$DRY || say "[00_clean] Fresh dirs → ${ART_DIR}/models, ${ART_DIR}/models_mps, ${ART_DIR}/reports"
say "[00_clean] Done."
say "****************************************"