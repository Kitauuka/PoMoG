#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TRAIN_FEATURE_ROOT="${TRAIN_FEATURE_ROOT:-${FEATURE_ROOT:-features/pomog}}"
TEST_FEATURE_ROOT="${TEST_FEATURE_ROOT:-features/pomog_test}"
TRAIN_FASTA="${TRAIN_FASTA:-data/fasta/train.fasta}"
TEST_FASTA="${TEST_FASTA:-data/fasta/test.fasta}"
POCKET_CACHE_ROOT="${POCKET_CACHE_ROOT:-features/pocket_cache}"
POCKET_WORKERS="${POCKET_WORKERS:-4}"
P2RANK_PRANK="${P2RANK_PRANK:-external_tools/p2rank_2.5.1/prank}"
FPOCKET_BIN="${FPOCKET_BIN:-fpocket}"
POMOG_POCKET_TOOL_ENV="${POMOG_POCKET_TOOL_ENV:-pomog-pocket-tools}"

generate_pocket_cache() {
  local split_path="$1"
  local feature_root="$2"
  local source="$3"
  local cmd=(
    $PYTHON scripts/generate_pocket_cache.py
    --split "$split_path"
    --feature-dir "$feature_root"
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
  --fasta "$TRAIN_FASTA" \
  --save_dir "$TRAIN_FEATURE_ROOT" \
  --gpu "$GPU" \
  --run_esmfold \
  --run_prottrans \
  --run_dssp \
  --compute_stats \
  --only_missing

$PYTHON Features/features.py \
  --fasta "$TEST_FASTA" \
  --save_dir "$TEST_FEATURE_ROOT" \
  --gpu "$GPU" \
  --run_esmfold \
  --run_prottrans \
  --run_dssp \
  --only_missing

$PYTHON scripts/prepare_prottrans_stats.py \
  --train-split data/splits/train.pkl \
  --source-feature-dir "$TRAIN_FEATURE_ROOT" \
  --target-feature-dir "$TRAIN_FEATURE_ROOT" \
  --target-feature-dir "$TEST_FEATURE_ROOT" \
  --manifest "$TRAIN_FEATURE_ROOT/ProtTrans/normalization_manifest.json"

for SOURCE in p2rank fpocket; do
  generate_pocket_cache data/splits/train.pkl "$TRAIN_FEATURE_ROOT" "$SOURCE"
done

for SOURCE in p2rank fpocket; do
  generate_pocket_cache data/splits/test.pkl "$TEST_FEATURE_ROOT" "$SOURCE"
done
