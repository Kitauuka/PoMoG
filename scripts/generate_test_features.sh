#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TEST_FEATURE_ROOT="${TEST_FEATURE_ROOT:-features/pomog_test}"
TEST_FASTA="${TEST_FASTA:-data/fasta/test.fasta}"
POCKET_CACHE_ROOT="${POCKET_CACHE_ROOT:-features/pocket_cache}"
POCKET_WORKERS="${POCKET_WORKERS:-4}"
P2RANK_PRANK="${P2RANK_PRANK:-external_tools/p2rank_2.5.1/prank}"
FPOCKET_BIN="${FPOCKET_BIN:-fpocket}"
POMOG_POCKET_TOOL_ENV="${POMOG_POCKET_TOOL_ENV:-pomog-pocket-tools}"
STATS_SOURCE_FEATURE_ROOT="${STATS_SOURCE_FEATURE_ROOT:-models/PoMoG/normalization}"

generate_pocket_cache() {
  local source="$1"
  local cmd=(
    $PYTHON scripts/generate_pocket_cache.py
    --split data/splits/test.pkl
    --feature-dir "$TEST_FEATURE_ROOT"
    --output-dir "$POCKET_CACHE_ROOT"
    --source "$source"
    --workers "$POCKET_WORKERS"
    --prank-bin "$P2RANK_PRANK"
    --fpocket-bin "$FPOCKET_BIN"
  )
  if [[ -n "$POMOG_POCKET_TOOL_ENV" ]]; then
    cmd+=(--tool-conda-env "$POMOG_POCKET_TOOL_ENV")
  fi
  "${cmd[@]}"
}

$PYTHON Features/features.py \
  --fasta "$TEST_FASTA" \
  --save_dir "$TEST_FEATURE_ROOT" \
  --gpu "$GPU" \
  --run_esmfold \
  --run_prottrans \
  --run_dssp \
  --only_missing

$PYTHON scripts/prepare_prottrans_stats.py \
  --source-feature-dir "$STATS_SOURCE_FEATURE_ROOT" \
  --target-feature-dir "$TEST_FEATURE_ROOT" \
  --copy-only \
  --manifest "$TEST_FEATURE_ROOT/ProtTrans/normalization_manifest.json"

for SOURCE in p2rank fpocket; do
  generate_pocket_cache "$SOURCE"
done
