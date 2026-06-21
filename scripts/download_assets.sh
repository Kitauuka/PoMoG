#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
mkdir -p models features

SHARED_MODEL_ROOT="${POMOG_SHARED_MODEL_ROOT:-/home/sharedata/djh/models}"
PROTT5_SOURCE="${POMOG_PROTT5_SOURCE:-$SHARED_MODEL_ROOT/Prot-T5-XL-U50}"
ESMFOLD_SOURCE="${POMOG_ESMFOLD_SOURCE:-$SHARED_MODEL_ROOT/esmfold_3B_v1.pt}"
ESM2_SOURCE="${POMOG_ESM2_SOURCE:-$SHARED_MODEL_ROOT/ESM2/esm2_t36_3B_UR50D.pt}"
CANONICAL_FEATURE_SOURCE="${POMOG_CANONICAL_FEATURE_SOURCE:-}"
POCKET_CACHE_SOURCE="${POMOG_POCKET_CACHE_SOURCE:-}"
SETUP_POCKET_TOOLS="${SETUP_POCKET_TOOLS:-1}"
POMOG_POCKET_TOOL_ENV="${POMOG_POCKET_TOOL_ENV:-pomog-pocket-tools}"

if [[ -d "$PROTT5_SOURCE" && ! -e models/Prot-T5-XL-U50 ]]; then
  ln -s "$PROTT5_SOURCE" models/Prot-T5-XL-U50
  echo "Linked ProtT5 from $PROTT5_SOURCE"
fi

if [[ -f "$ESMFOLD_SOURCE" && ! -e models/esmfold_3B_v1.pt ]]; then
  ln -s "$ESMFOLD_SOURCE" models/esmfold_3B_v1.pt
  echo "Linked ESMFold from $ESMFOLD_SOURCE"
fi

if [[ -f "$ESM2_SOURCE" && ! -e models/ESM2/esm2_t36_3B_UR50D.pt ]]; then
  mkdir -p models/ESM2
  ln -s "$ESM2_SOURCE" models/ESM2/esm2_t36_3B_UR50D.pt
  echo "Linked ESM2 from $ESM2_SOURCE"
fi

if [[ -n "$CANONICAL_FEATURE_SOURCE" && -e "$CANONICAL_FEATURE_SOURCE" && ! -e features/canonical_strict ]]; then
  ln -s "$CANONICAL_FEATURE_SOURCE" features/canonical_strict
  echo "Linked canonical strict features from $CANONICAL_FEATURE_SOURCE"
fi

if [[ -n "$POCKET_CACHE_SOURCE" && -e "$POCKET_CACHE_SOURCE" && ! -e features/pocket_cache/PoMoG ]]; then
  mkdir -p features/pocket_cache
  ln -s "$POCKET_CACHE_SOURCE" features/pocket_cache/PoMoG
  echo "Linked PoMoG pocket cache from $POCKET_CACHE_SOURCE"
fi

if [[ "$SETUP_POCKET_TOOLS" != "0" ]]; then
  $PYTHON scripts/setup_pocket_tools.py --env-name "$POMOG_POCKET_TOOL_ENV"
fi

if [[ -d models/Prot-T5-XL-U50 && -e models/esmfold_3B_v1.pt && -e models/ESM2/esm2_t36_3B_UR50D.pt ]]; then
  echo "PoMoG pretrained feature assets are available."
  echo "DSSP is bundled at Features/dssp-2.0.4/mkdssp"
  exit 0
fi

$PYTHON - <<'PY'
from pathlib import Path
from transformers import T5EncoderModel, T5Tokenizer

target = Path("models/Prot-T5-XL-U50")
target.mkdir(parents=True, exist_ok=True)
name = "Rostlab/prot_t5_xl_uniref50"

T5Tokenizer.from_pretrained(name, do_lower_case=False).save_pretrained(target)
T5EncoderModel.from_pretrained(name).save_pretrained(target)
print(f"ProtT5 saved to {target}")
PY

$PYTHON - <<'PY'
import esm

esm.pretrained.esmfold_v1()
print("ESMFold weights are available in the torch cache.")
PY

echo "DSSP is bundled at Features/dssp-2.0.4/mkdssp"
