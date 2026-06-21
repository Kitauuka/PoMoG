#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
FEATURE_ROOT="${FEATURE_ROOT:-features/pomog}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/PoMoG}"
TRAIN_CONFIG="$OUTPUT_DIR/config.train.json"

mkdir -p "$OUTPUT_DIR"

$PYTHON scripts/prepare_prottrans_stats.py \
  --train-split data/splits/train.pkl \
  --source-feature-dir "$FEATURE_ROOT" \
  --target-feature-dir "$FEATURE_ROOT" \
  --manifest "$FEATURE_ROOT/ProtTrans/normalization_manifest.json"

$PYTHON - <<PY
import json
from pathlib import Path

config = json.loads(Path("models/PoMoG/config.train.json").read_text())
config["model_save_dir"] = "${OUTPUT_DIR}/model"
config["p2rank_cache_dir"] = "features/pocket_cache"
config["fpocket_cache_dir"] = "features/pocket_cache"
config["pomog_pocket_cache_dir"] = "features/pocket_cache/PoMoG"
config["pomog_pocket_cache_required"] = False
config["pomog_pocket_cache_write"] = True
Path("${TRAIN_CONFIG}").write_text(json.dumps(config, indent=2) + "\\n")
PY

$PYTHON -m pomog.training \
  --data_path data/splits/train.pkl \
  --feature_dir "$FEATURE_ROOT" \
  --gpu "$GPU" \
  --config "$TRAIN_CONFIG" \
  --summary_path "$OUTPUT_DIR/training_summary.json"
