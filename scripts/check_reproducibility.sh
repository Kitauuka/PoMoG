#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

PYTHON="${PYTHON:-python}"
read -r -a PYTHON_CMD <<< "$PYTHON"

TMP_CHECK="$(mktemp)"
trap 'rm -f "$TMP_CHECK"' EXIT

cat > "$TMP_CHECK" <<'PY'
import csv
import importlib
import json
from pathlib import Path

root = Path(".")

for module_name in ["torch", "torch_geometric", "torch_scatter", "sklearn"]:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"[OK] import {module_name}: {version}")

required_paths = [
    "data/splits/train.pkl",
    "data/splits/test.pkl",
    "data/splits/test_gold.pkl",
    "data/splits/test_silver.pkl",
    "models/PoMoG/config.train.json",
    "models/PoMoG/model/fold0.ckpt",
    "models/PoMoG/model/fold1.ckpt",
    "models/PoMoG/model/fold2.ckpt",
    "models/PoMoG/model/fold3.ckpt",
    "models/PoMoG/model/fold4.ckpt",
    "models/PoMoG/normalization/ProtTrans/Max_ProtTrans.npy",
    "models/PoMoG/normalization/ProtTrans/Min_ProtTrans.npy",
    "results/paper_results.csv",
    "scripts/test_pretrained.sh",
    "scripts/generate_features.sh",
    "scripts/train_5fold.sh",
    "scripts/test_model.sh",
]
missing = [path for path in required_paths if not (root / path).exists()]
if missing:
    raise FileNotFoundError(missing)
print(f"[OK] required artifacts: {len(required_paths)}")

config = json.loads((root / "models/PoMoG/config.train.json").read_text())
if int(config.get("folds", 0)) != 5:
    raise RuntimeError("release config must use 5 folds")
if int(config.get("seed", -1)) != 2026:
    raise RuntimeError("release config seed changed from 2026")
print("[OK] release config: folds=5 seed=2026")

with (root / "results/paper_results.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise RuntimeError("results/paper_results.csv is empty")
row = rows[0]
if row.get("model") != "PoMoG" or int(row.get("test_samples", 0)) != 342:
    raise RuntimeError("paper result row does not match the released PoMoG test set")
print(
    "[OK] paper metrics: "
    f"AUPR={row['test_aupr']} AUC={row['test_auc']} F1={row['test_f1']} MCC={row['test_mcc']}"
)
PY

"${PYTHON_CMD[@]}" "$TMP_CHECK"
PYTHON="$PYTHON" bash scripts/smoke_test.sh

cat <<'EOF'
[OK] lightweight reproducibility preflight passed

Path 1: released-weight reproduction
  bash scripts/download_assets.sh
  bash scripts/test_pretrained.sh

Path 2: full reproduction with retraining
  bash scripts/download_assets.sh
  GPU=0 bash scripts/generate_features.sh
  GPU=0 bash scripts/train_5fold.sh
  MODEL_DIR=outputs/train/PoMoG/model FEATURE_ROOT=features/pomog_test STATS_SOURCE_FEATURE_ROOT=features/pomog bash scripts/test_model.sh
EOF
