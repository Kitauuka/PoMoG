"""Prepare ProtTrans normalization statistics from the training split."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


def _load_split(path: Path) -> dict:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            import numpy.core as _np_core
            import sys

            sys.modules.setdefault("numpy._core", _np_core)
            handle.seek(0)
            return pickle.load(handle)


def _find_tensor(prottrans_dir: Path, protein_id: str) -> Path:
    for candidate in (protein_id, protein_id.upper(), protein_id.lower()):
        path = prottrans_dir / f"{candidate}.tensor"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing ProtTrans tensor for {protein_id} in {prottrans_dir}")


def _compute_stats(prottrans_dir: Path, protein_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray, int, int]:
    global_min: np.ndarray | None = None
    global_max: np.ndarray | None = None
    protein_count = 0
    residue_count = 0

    protein_ids = list(protein_ids)
    for protein_id in tqdm(protein_ids, desc="ProtTrans train stats"):
        tensor = torch.load(_find_tensor(prottrans_dir, protein_id), map_location="cpu").float()
        if tensor.ndim != 2 or tensor.shape[1] != 1024:
            raise ValueError(f"{protein_id} has invalid ProtTrans shape {tuple(tensor.shape)}")
        arr = tensor.numpy()
        row_min = arr.min(axis=0)
        row_max = arr.max(axis=0)
        global_min = row_min if global_min is None else np.minimum(global_min, row_min)
        global_max = row_max if global_max is None else np.maximum(global_max, row_max)
        protein_count += 1
        residue_count += int(arr.shape[0])

    if global_min is None or global_max is None:
        raise ValueError("no training ProtTrans tensors were found")
    return global_min, global_max, protein_count, residue_count


def _write_stats(feature_dir: Path, min_values: np.ndarray, max_values: np.ndarray) -> None:
    prottrans_dir = feature_dir / "ProtTrans"
    prottrans_dir.mkdir(parents=True, exist_ok=True)
    np.save(prottrans_dir / "Min_ProtTrans.npy", min_values)
    np.save(prottrans_dir / "Max_ProtTrans.npy", max_values)


def _copy_stats(source_feature_dir: Path, target_feature_dir: Path) -> None:
    source_prottrans = source_feature_dir / "ProtTrans"
    target_prottrans = target_feature_dir / "ProtTrans"
    target_prottrans.mkdir(parents=True, exist_ok=True)
    for name in ("Min_ProtTrans.npy", "Max_ProtTrans.npy"):
        shutil.copy2(source_prottrans / name, target_prottrans / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train-derived ProtTrans normalization stats")
    parser.add_argument("--train-split", default="data/splits/train.pkl")
    parser.add_argument("--source-feature-dir", required=True)
    parser.add_argument("--target-feature-dir", action="append", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--copy-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_feature_dir = Path(args.source_feature_dir)
    source_prottrans = source_feature_dir / "ProtTrans"
    if not source_prottrans.exists():
        raise FileNotFoundError(f"missing ProtTrans directory: {source_prottrans}")

    if args.copy_only:
        for name in ("Min_ProtTrans.npy", "Max_ProtTrans.npy"):
            if not (source_prottrans / name).exists():
                raise FileNotFoundError(f"missing source stat file: {source_prottrans / name}")
        protein_count = residue_count = None
    else:
        train_split = _load_split(Path(args.train_split))
        min_values, max_values, protein_count, residue_count = _compute_stats(source_prottrans, train_split.keys())
        _write_stats(source_feature_dir, min_values, max_values)

    for target in args.target_feature_dir:
        target_feature_dir = Path(target)
        if target_feature_dir.resolve() != source_feature_dir.resolve():
            _copy_stats(source_feature_dir, target_feature_dir)

    manifest_path = Path(args.manifest) if args.manifest else source_feature_dir / "ProtTrans" / "normalization_manifest.json"
    manifest = {
        "normalization": "ProtTrans min-max",
        "stats_source": "training split",
        "train_split": str(Path(args.train_split)),
        "source_feature_dir": str(source_feature_dir),
        "target_feature_dirs": [str(Path(path)) for path in args.target_feature_dir],
        "protein_count": protein_count,
        "residue_count": residue_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote train-derived ProtTrans stats to {source_feature_dir / 'ProtTrans'}")
    for target in args.target_feature_dir:
        print(f"[OK] installed stats in {Path(target) / 'ProtTrans'}")


if __name__ == "__main__":
    main()
