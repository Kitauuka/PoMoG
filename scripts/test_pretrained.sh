#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="models/PoMoG/model" bash scripts/test_model.sh
