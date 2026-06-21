#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
CANONICAL_FEATURE_ROOT="${CANONICAL_FEATURE_ROOT:-features/canonical_strict}"
if [[ -z "${FEATURE_ROOT:-}" ]]; then
  if [[ -e "$CANONICAL_FEATURE_ROOT" ]]; then
    FEATURE_ROOT="$CANONICAL_FEATURE_ROOT"
  else
    FEATURE_ROOT="features/pomog_test"
  fi
fi
MODEL_DIR="${MODEL_DIR:-models/PoMoG/model}"
RUN_NAME="${RUN_NAME:-pomog_test}"
STATS_SOURCE_FEATURE_ROOT="${STATS_SOURCE_FEATURE_ROOT:-models/PoMoG/normalization}"

if [[ ! -f "$FEATURE_ROOT/ProtTrans/Max_ProtTrans.npy" || ! -f "$FEATURE_ROOT/ProtTrans/Min_ProtTrans.npy" ]]; then
  if [[ -f "$STATS_SOURCE_FEATURE_ROOT/ProtTrans/Max_ProtTrans.npy" && -f "$STATS_SOURCE_FEATURE_ROOT/ProtTrans/Min_ProtTrans.npy" ]]; then
    $PYTHON scripts/prepare_prottrans_stats.py \
      --source-feature-dir "$STATS_SOURCE_FEATURE_ROOT" \
      --target-feature-dir "$FEATURE_ROOT" \
      --copy-only \
      --manifest "$FEATURE_ROOT/ProtTrans/normalization_manifest.json"
  else
    echo "Missing ProtTrans normalization stats in $FEATURE_ROOT/ProtTrans." >&2
    echo "Generate train features first, or set STATS_SOURCE_FEATURE_ROOT to a directory containing train-derived stats." >&2
    exit 1
  fi
fi

$PYTHON scripts/predict.py \
  --dataset_path data/splits/test.pkl \
  --feature_dir "$FEATURE_ROOT" \
  --model_dir "$MODEL_DIR" \
  --output_root outputs/test \
  --run_name "$RUN_NAME" \
  --gpu "$GPU" \
  --allow_overwrite
