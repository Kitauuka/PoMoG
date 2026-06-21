#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

PYTHON="${PYTHON:-python}"
read -r -a PYTHON_CMD <<< "$PYTHON"

TMP_CHECK="$(mktemp)"
trap 'rm -f "$TMP_CHECK"' EXIT

cat > "$TMP_CHECK" <<'PY'
import json
import pickle
from pathlib import Path

from pomog.model import PoMoG
from pomog.train.train_pomog import DEFAULT_CONFIG, _model_kwargs, load_config

root = Path(".")
required = [
    "data/splits/train.pkl",
    "data/splits/test.pkl",
    "data/splits/test_gold.pkl",
    "data/splits/test_silver.pkl",
    "models/PoMoG/model/fold0.ckpt",
    "models/PoMoG/model/fold1.ckpt",
    "models/PoMoG/model/fold2.ckpt",
    "models/PoMoG/model/fold3.ckpt",
    "models/PoMoG/model/fold4.ckpt",
]
missing = [path for path in required if not (root / path).exists()]
if missing:
    raise FileNotFoundError(missing)

with open(root / "data/splits/test.pkl", "rb") as handle:
    test = pickle.load(handle)
with open(root / "data/splits/test_gold.pkl", "rb") as handle:
    gold = pickle.load(handle)
with open(root / "data/splits/test_silver.pkl", "rb") as handle:
    silver = pickle.load(handle)
if len(test) != len(gold) + len(silver):
    raise RuntimeError("gold and silver subsets do not cover the merged test set")

json.loads((root / "models/PoMoG/config.train.json").read_text())

expected_param_count = 3226284
default_model = PoMoG(**_model_kwargs(DEFAULT_CONFIG, augment_eps=0.0))
release_config = load_config(DEFAULT_CONFIG, "models/PoMoG/config.train.json")
release_model = PoMoG(**_model_kwargs(release_config, augment_eps=0.0))
default_param_count = sum(param.numel() for param in default_model.parameters())
release_param_count = sum(param.numel() for param in release_model.parameters())
if default_param_count != expected_param_count:
    raise RuntimeError(f"default model parameter count changed: {default_param_count} != {expected_param_count}")
if release_param_count != expected_param_count:
    raise RuntimeError(f"release model parameter count changed: {release_param_count} != {expected_param_count}")

print("[OK] PoMoG smoke test passed")
print(f"[OK] test: {len(test)} proteins; gold: {len(gold)}; silver: {len(silver)}")
print(f"[OK] model parameters: {release_param_count}")
PY

"${PYTHON_CMD[@]}" "$TMP_CHECK"
