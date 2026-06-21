#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


FEATURE_NAMES = [
    "residue_in_top1_pocket",
    "residue_in_top3_pocket",
    "nearest_pocket_distance",
    "pocket_score",
    "pocket_rank_reciprocal",
]


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            import numpy.core as _np_core

            sys.modules.setdefault("numpy._core", _np_core)
            f.seek(0)
            return pickle.load(f)


def collect_dataset(split_paths: list[Path]) -> dict[str, tuple[str, Any]]:
    dataset: dict[str, tuple[str, Any]] = {}
    for path in split_paths:
        data = _load_pickle(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a dict dataset")
        for name, value in data.items():
            if not value:
                continue
            dataset[str(name)] = (str(value[0]), value[1] if len(value) > 1 else None)
    return dataset


def _find_pdb(structure_dir: Path, name: str) -> Path | None:
    for candidate in (name, name.upper(), name.lower()):
        path = structure_dir / f"{candidate}.pdb"
        if path.exists():
            return path
    return None


def _load_length(feature_dir: Path, name: str, fallback_seq: str) -> int:
    for candidate in (name, name.upper(), name.lower()):
        path = feature_dir / "Structures" / f"{candidate}.tensor"
        if path.exists():
            return int(torch.load(path, map_location="cpu").shape[0])
    return len(fallback_seq)


def _load_ca(feature_dir: Path, name: str, length: int) -> torch.Tensor:
    for candidate in (name, name.upper(), name.lower()):
        path = feature_dir / "Structures" / f"{candidate}.tensor"
        if path.exists():
            x = torch.load(path, map_location="cpu").float()
            return torch.nan_to_num(x[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.zeros((length, 3), dtype=torch.float32)


def _parse_pdb_residue_index(line: str) -> int | None:
    try:
        raw = line[22:26].strip()
        if raw:
            return int(raw) - 1
    except ValueError:
        pass
    match = re.search(r"([A-Za-z]?)[_: -]?(\d+)", line)
    if not match:
        return None
    return int(match.group(2)) - 1


def _parse_atom_xyz(line: str) -> tuple[float, float, float] | None:
    try:
        return float(line[30:38]), float(line[38:46]), float(line[46:54])
    except ValueError:
        parts = line.split()
        for start in range(max(0, len(parts) - 6), len(parts) - 2):
            try:
                return float(parts[start]), float(parts[start + 1]), float(parts[start + 2])
            except ValueError:
                continue
    return None


def _parse_fpocket_info(info_path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not info_path.exists():
        return scores
    current_rank: int | None = None
    for line in info_path.read_text(errors="ignore").splitlines():
        rank_match = re.search(r"Pocket\s+(\d+)", line, flags=re.IGNORECASE)
        if rank_match:
            current_rank = int(rank_match.group(1))
            continue
        if current_rank is None:
            continue
        score_match = re.search(r"^\s*Score\s*:\s*([-+0-9.eE]+)", line)
        if score_match:
            try:
                scores[current_rank] = float(score_match.group(1))
            except ValueError:
                pass
    return scores


def _parse_fpocket_output(out_dir: Path) -> list[dict[str, Any]]:
    info_files = sorted(out_dir.glob("*_info.txt"))
    scores = _parse_fpocket_info(info_files[0]) if info_files else {}
    pockets: list[dict[str, Any]] = []
    for path in sorted((out_dir / "pockets").glob("pocket*_atm.pdb")):
        match = re.search(r"pocket(\d+)_", path.name)
        if not match:
            continue
        rank = int(match.group(1))
        residues: set[int] = set()
        xyz: list[tuple[float, float, float]] = []
        for line in path.read_text(errors="ignore").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            idx = _parse_pdb_residue_index(line)
            if idx is not None:
                residues.add(idx)
            coord = _parse_atom_xyz(line)
            if coord is not None:
                xyz.append(coord)
        center = torch.tensor(xyz, dtype=torch.float32).mean(dim=0) if xyz else torch.zeros(3)
        pockets.append(
            {
                "rank": rank,
                "score": float(scores.get(rank, 0.0)),
                "center": center,
                "residue_index": torch.tensor(sorted(residues), dtype=torch.long),
            }
        )
    return sorted(pockets, key=lambda item: int(item["rank"]))


def _parse_residue_tokens(text: str, length: int) -> list[int]:
    residues: set[int] = set()
    for token in re.findall(r"(?:[A-Za-z][_:])?(\d+)", text):
        idx = int(token) - 1
        if 0 <= idx < length:
            residues.add(idx)
    return sorted(residues)


def _parse_p2rank_output(out_dir: Path, pdb_path: Path, length: int) -> list[dict[str, Any]]:
    candidates = list(out_dir.glob("*predictions*.csv"))
    pockets: list[dict[str, Any]] = []
    for csv_path in sorted(candidates):
        try:
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row_id, row in enumerate(rows, start=1):
            row = {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
            residues = _parse_residue_tokens(row.get("residue_ids", ""), length)
            if not residues:
                continue
            rank = int(float(row.get("rank") or row.get("pocket") or row_id))
            score = float(row.get("score") or row.get("probability") or row.get("prob") or 0.0)
            coords = []
            for prefix in ("center_", ""):
                keys = [f"{prefix}x", f"{prefix}y", f"{prefix}z"]
                if all(key in row and row[key] not in {"", None} for key in keys):
                    try:
                        coords = [float(row[key]) for key in keys]
                        break
                    except ValueError:
                        coords = []
            pockets.append(
                {
                    "rank": rank,
                    "score": score,
                    "center": torch.tensor(coords if coords else [0.0, 0.0, 0.0], dtype=torch.float32),
                    "residue_index": torch.tensor(residues, dtype=torch.long),
                    "source_file": str(csv_path),
                }
            )
    unique: dict[int, dict[str, Any]] = {}
    for pocket in pockets:
        unique.setdefault(int(pocket["rank"]), pocket)
    return [unique[key] for key in sorted(unique)]


def _build_residue_feature(length: int, ca: torch.Tensor, pockets: list[dict[str, Any]]) -> torch.Tensor:
    feat = torch.zeros((length, len(FEATURE_NAMES)), dtype=torch.float32)
    if length <= 0 or not pockets:
        feat[:, 2] = 30.0
        return feat

    nearest = torch.full((length,), 30.0, dtype=torch.float32)
    best_score = torch.zeros((length,), dtype=torch.float32)
    best_rank = torch.zeros((length,), dtype=torch.float32)
    for pocket in pockets:
        rank = max(1, int(pocket.get("rank", 999)))
        score = float(pocket.get("score", 0.0))
        idx = pocket.get("residue_index", torch.zeros(0, dtype=torch.long)).long()
        idx = idx[(idx >= 0) & (idx < length)]
        if idx.numel() == 0:
            center = pocket.get("center", torch.zeros(3)).float().view(1, 3)
            dist = torch.linalg.norm(ca - center, dim=-1)
        else:
            dist = torch.cdist(ca.float(), ca[idx].float()).min(dim=1).values
        nearest = torch.minimum(nearest, dist.clamp(max=30.0))
        if idx.numel() > 0:
            if rank == 1:
                feat[idx, 0] = 1.0
            if rank <= 3:
                feat[idx, 1] = 1.0
            update = score >= best_score[idx]
            target = idx[update]
            best_score[target] = score
            best_rank[target] = 1.0 / float(rank)
    feat[:, 2] = nearest
    feat[:, 3] = best_score
    feat[:, 4] = best_rank
    return torch.nan_to_num(feat, nan=0.0, posinf=30.0, neginf=0.0)


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def _prepare_one(
    name: str,
    seq: str,
    *,
    feature_dir: Path,
    output_dir: Path,
    source: str,
    fpocket_bin: str,
    prank_bin: str,
    tool_conda_env: str | None,
    run_tools: bool,
    overwrite: bool,
) -> dict[str, Any]:
    structure_dir = feature_dir / "Structures"
    pdb_path = _find_pdb(structure_dir, name)
    if pdb_path is None:
        return {"name": name, "status": "missing_pdb"}

    length = _load_length(feature_dir, name, seq)
    ca = _load_ca(feature_dir, name, length)
    source_dir = output_dir / "raw" / source / name
    cache_path = output_dir / source / f"{name}.pt"
    if cache_path.exists() and not overwrite:
        return {"name": name, "status": "cached", "cache": str(cache_path)}

    source_dir.mkdir(parents=True, exist_ok=True)
    if source == "fpocket":
        out_dir = source_dir / f"{pdb_path.stem}_out"
        if run_tools and (overwrite or not out_dir.exists()):
            work_pdb = source_dir / pdb_path.name
            if not work_pdb.exists() or overwrite:
                shutil.copy2(pdb_path, work_pdb)
            cmd = shlex.split(fpocket_bin) + ["-f", str(work_pdb)]
            if tool_conda_env:
                cmd = ["conda", "run", "-n", tool_conda_env, *cmd]
            code = _run_command(cmd, source_dir, source_dir / "fpocket.log")
            if code != 0:
                return {"name": name, "status": "fpocket_failed", "log": str(source_dir / "fpocket.log")}
        pockets = _parse_fpocket_output(out_dir)
    elif source == "p2rank":
        out_dir = source_dir / "p2rank_out"
        if run_tools and (overwrite or not out_dir.exists()):
            cmd = shlex.split(prank_bin) + ["predict", "-f", str(pdb_path), "-o", str(out_dir)]
            if tool_conda_env:
                cmd = ["conda", "run", "-n", tool_conda_env, *cmd]
            code = _run_command(cmd, source_dir, source_dir / "p2rank.log")
            if code != 0:
                return {"name": name, "status": "p2rank_failed", "log": str(source_dir / "p2rank.log")}
        pockets = _parse_p2rank_output(out_dir, pdb_path, length)
    else:
        raise ValueError(f"unsupported source: {source}")

    payload = {
        "name": name,
        "source": source,
        "length": length,
        "feature_names": FEATURE_NAMES,
        "residue_feature": _build_residue_feature(length, ca, pockets),
        "pockets": pockets,
        "pocket_count": len(pockets),
        "pdb_path": str(pdb_path),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return {"name": name, "status": "ok", "cache": str(cache_path), "pocket_count": len(pockets)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external pocket tools and build PoMoG pocket feature caches.")
    parser.add_argument("--split", action="append", required=True, help="Dataset PKL split path. Can be repeated.")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=["fpocket", "p2rank"], required=True)
    parser.add_argument("--fpocket-bin", default=os.environ.get("FPOCKET_BIN", "fpocket"))
    parser.add_argument("--prank-bin", default=os.environ.get("P2RANK_PRANK", "external_tools/p2rank_2.5.1/prank"))
    parser.add_argument("--tool-conda-env", default=os.environ.get("POMOG_POCKET_TOOL_ENV", ""))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--parse-only", action="store_true", help="Do not run tools; parse existing raw outputs only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset = collect_dataset([Path(item).resolve() for item in args.split])
    names = sorted(dataset)
    if args.only:
        selected = set(args.only)
        names = [name for name in names if name in selected]
    if args.limit > 0:
        names = names[: args.limit]

    manifest = {
        "source": args.source,
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "split": args.split,
        "count": len(names),
        "feature_names": FEATURE_NAMES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"manifest_{args.source}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                _prepare_one,
                name,
                dataset[name][0],
                feature_dir=feature_dir,
                output_dir=output_dir,
                source=args.source,
                fpocket_bin=args.fpocket_bin,
                prank_bin=str(Path(args.prank_bin).resolve()),
                tool_conda_env=args.tool_conda_env or None,
                run_tools=not args.parse_only,
                overwrite=args.overwrite,
            )
            for name in names
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if idx % 25 == 0 or idx == len(futures):
                ok = sum(1 for item in results if item["status"] in {"ok", "cached"})
                print(f"[{args.source}] {idx}/{len(futures)} done ok_or_cached={ok}", flush=True)

    summary: dict[str, int] = {}
    for item in results:
        summary[item["status"]] = summary.get(item["status"], 0) + 1
    summary_path = output_dir / f"summary_{args.source}.json"
    summary_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
