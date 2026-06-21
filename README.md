# PoMoG

PoMoG is a residue-level enzyme active-site predictor. This repository contains
the code, strict split dataset, five-fold PoMoG model weights, and reproduction
scripts used for the open-source release.

## File Structure

```text
PoMoG/
├── README.md
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── splits/
│   ├── fasta/
│   └── metadata/
├── models/
│   └── PoMoG/
│       ├── config.train.json
│       ├── model/
│       └── normalization/
├── Features/
├── scripts/
├── results/
└── src/
    └── pomog/
        ├── data/       # residue graph data and pocket-cache adapters
        ├── models/     # PoMoG network and neural layers
        ├── train/      # five-fold training loop
        ├── dataset.py  # public dataset import
        ├── model.py    # public model import
        ├── metrics.py
        └── training.py # python -m pomog.training
```

`src/` contains the PoMoG dataset, model, training, and inference code.
`Features/` contains the feature-generation entrypoint and the bundled DSSP
binary used by the scripts.

## Environment

The reference environment uses Python 3.8, PyTorch 1.13.1, CUDA 11.7, and PyG
2.3.0.

```bash
conda env create -f environment.yml
conda activate PoMoG
pip install -e .
```

If you do not activate the environment, pass the Python command explicitly:

```bash
PYTHON="conda run -n PoMoG python" bash scripts/test_pretrained.sh
```

## Dataset

The included dataset is the strict PoMoG split.

```text
data/splits/train.pkl        training set
data/splits/test.pkl         merged test set
data/splits/test_gold.pkl    gold-only subset of test.pkl
data/splits/test_silver.pkl  silver-only subset of test.pkl
```

Gold labels are M-CSA curated catalytic residue annotations. Silver labels are
reviewed UniProt ACT_SITE annotations supported by literature or structure
evidence. `data/metadata/split_members.csv` records the split and quality tier
for every protein.

## Feature Normalization

PoMoG uses ProtTrans min-max normalization. The max/min statistics must be
computed from the training split only and then reused for every test set. Do not
fit normalization statistics on the test set.

The release scripts enforce this rule:

- `scripts/generate_features.sh` generates train/test features, computes
  `Max_ProtTrans.npy` and `Min_ProtTrans.npy` from `data/splits/train.pkl`, and
  installs the same files into the test feature directory.
- `scripts/train_5fold.sh` recomputes train-derived stats before training.
- `scripts/test_model.sh` refuses to test a feature directory without
  ProtTrans stats unless `STATS_SOURCE_FEATURE_ROOT` points to a directory that
  already contains train-derived stats. By default this points to the released
  stats in `models/PoMoG/normalization`.

The helper script can also be run directly:

```bash
python scripts/prepare_prottrans_stats.py \
  --train-split data/splits/train.pkl \
  --source-feature-dir features/pomog \
  --target-feature-dir features/pomog \
  --target-feature-dir features/pomog_test
```

For testing the released weights with newly generated test features only, the
repository includes the training-split max/min files under:

```text
models/PoMoG/normalization/ProtTrans/
```

## Released Weights

```text
model name: PoMoG
model path: models/PoMoG/model
folds: 5
test samples: 342
test AUPR: 0.671153
test AUC: 0.986371
test F1: 0.627719
test MCC: 0.624979
gold-only AUPR: 0.652929
silver-only AUPR: 0.793463
```

The full metric table is stored in `results/paper_results.csv`.

## Path 1: Test the Released Weights

Use this path when you want to reproduce the released model result without
retraining.

First link or download the pretrained feature models and optional released
feature/cache artifacts:

```bash
POMOG_CANONICAL_FEATURE_SOURCE=/path/to/released/strict/features \
POMOG_POCKET_CACHE_SOURCE=/path/to/released/pocket_cache \
bash scripts/download_assets.sh
```

Then evaluate the bundled five-fold PoMoG weights:

```bash
bash scripts/test_pretrained.sh
```

If canonical feature tensors are not available, generate only the test features
and use the released training-split normalization files:

```bash
bash scripts/download_assets.sh
GPU=0 bash scripts/generate_test_features.sh
bash scripts/test_pretrained.sh
```

The output is written to:

```text
outputs/test/pomog_test/summary.json
outputs/test/pomog_test/prediction_results.pkl
outputs/test/pomog_test/reports/
```

For exact paper-table reproduction, use the released canonical feature tensors
and matching PoMoG pocket cache. If you regenerate ESMFold structures locally,
small numerical differences are expected.

## Path 2: Full Reproduction With Retraining

Use this path when you want to regenerate features, train PoMoG from scratch,
and evaluate the trained five-fold model.

Install or link external assets:

```bash
bash scripts/download_assets.sh
```

Generate train/test features and pocket caches:

```bash
GPU=0 bash scripts/generate_features.sh
```

This creates:

```text
features/pomog/       train features with train-derived ProtTrans stats
features/pomog_test/  test features using the same train-derived stats
features/pocket_cache/
```

Run five-fold training:

```bash
GPU=0 bash scripts/train_5fold.sh
```

Evaluate the newly trained model on the merged test set:

```bash
MODEL_DIR=outputs/train/PoMoG/model \
FEATURE_ROOT=features/pomog_test \
STATS_SOURCE_FEATURE_ROOT=features/pomog \
bash scripts/test_model.sh
```

The output is written to:

```text
outputs/test/pomog_test/summary.json
```

## Gold and Silver Test Subsets

The main script evaluates `data/splits/test.pkl`. To evaluate a subset directly:

```bash
python scripts/predict.py \
  --dataset_path data/splits/test_gold.pkl \
  --feature_dir features/pomog_test \
  --model_dir models/PoMoG/model \
  --output_root outputs/test \
  --run_name pomog_test_gold \
  --gpu 0 \
  --allow_overwrite
```

Use `data/splits/test_silver.pkl` for the silver-only subset.

## Quick Check

Run the lightweight reproducibility preflight before testing released weights
or starting a full reproduction:

```bash
bash scripts/check_reproducibility.sh
```

This checks core dependency imports, split files, released weights, published
metric metadata, and the PoMoG model parameter count. For the smallest package
smoke check only, run:

```bash
bash scripts/smoke_test.sh
```

Feature generation is the slow step because ESMFold and ProtTrans are large
models. P2Rank/fpocket also run on every generated structure. The scripts keep
generated tensors under `features/`, external tools under `external_tools/`, and
all run outputs under `outputs/`; these directories are ignored by git.
