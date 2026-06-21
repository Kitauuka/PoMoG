"""PoMoG inference."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pomog.config_utils import load_training_run_config
from pomog.dataset import PoMoGDataset
from pomog.metrics import evaluate_pomog
from pomog.model import PoMoG
from pomog.train.train_pomog import (
    DEFAULT_CONFIG,
    _dataset_kwargs,
    _model_forward,
    _model_kwargs,
    _unwrap,
)


def _load_pickle_compat(path: Path):
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


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", "").replace("model.", ""): value for key, value in state_dict.items()}


def _collect_ckpts(model_dir: Path) -> List[Path]:
    ckpts = sorted(model_dir.glob("fold*.ckpt")) or sorted(model_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no ckpt files found: {model_dir}")
    return ckpts


def _load_model_config(model_dir: Path) -> Dict[str, Any]:
    return load_training_run_config(model_dir, DEFAULT_CONFIG)


def _build_model(device: torch.device, model_dir: Path) -> PoMoG:
    cfg = _load_model_config(model_dir)
    return PoMoG(**_model_kwargs(cfg, augment_eps=0.0)).to(device)


def _binary_aupr(labels: list[np.ndarray], logits: list[np.ndarray]) -> float:
    if not labels or not logits:
        return 0.0
    y = np.concatenate(labels).reshape(-1)
    x = np.concatenate(logits).reshape(-1)
    if y.size == 0 or float(y.max(initial=0.0)) <= 0.0:
        return 0.0
    probs = 1.0 / (1.0 + np.exp(-x))
    precision, recall, _ = precision_recall_curve(y, probs)
    return float(auc(recall, precision))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoMoG inference")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_root", default="./outputs/test")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"test_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    if run_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(f"output dir exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_pickle_compat(Path(args.dataset_path))
    ids = list(dataset.keys())
    model_dir = Path(args.model_dir)
    cfg = _load_model_config(model_dir)
    ds = PoMoGDataset(
        dataset=dataset,
        index=ids,
        root_dir=args.feature_dir,
        mode="infer",
        **_dataset_kwargs(cfg),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpts = _collect_ckpts(model_dir)
    models: List[PoMoG] = []
    for ckpt in ckpts:
        model = _build_model(device, model_dir)
        raw = torch.load(str(ckpt), map_location=device)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        model.load_state_dict(_normalize_state_dict(state), strict=True)
        model.eval()
        models.append(model)

    full_results: Dict[str, Dict[str, np.ndarray]] = {}
    as_probs_only: Dict[str, np.ndarray] = {}
    gate_values: list[np.ndarray] = []
    gate_targets: list[np.ndarray] = []
    pocket_feature_gate_values: list[np.ndarray] = []
    pocket_feature_gate_targets: list[np.ndarray] = []
    pocket_feature_alpha_rows: list[float] = []
    pocket_logits_rows: list[np.ndarray] = []
    pocket_target_rows: list[np.ndarray] = []
    motif_logits_rows: list[np.ndarray] = []
    motif_target_rows: list[np.ndarray] = []
    p2rank_flags: list[np.ndarray] = []
    p2rank_targets: list[np.ndarray] = []
    binary_chunks: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []
    moe_weight_rows: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="PoMoG-Ensemble"):
            batch = batch.to(device)
            outputs = []
            model_gate_rows = []
            model_pocket_gate_rows = []
            pocket_logit_batch = []
            motif_logit_batch = []
            pocket_target_batch = None
            motif_target_batch = None
            for model in models:
                output = torch.sigmoid(_model_forward(model, batch, return_projection=False))
                outputs.append(output)
                target_model = _unwrap(model)
                gate = getattr(target_model, "_last_residue_gate", None)
                if gate is not None and gate.numel() > 0:
                    model_gate_rows.append(gate.float().mean(dim=-1))
                pocket_feature_gate = getattr(target_model, "_last_pocket_feature_gate", None)
                if pocket_feature_gate is not None and pocket_feature_gate.numel() > 0:
                    model_pocket_gate_rows.append(pocket_feature_gate.float().mean(dim=-1))
                    pocket_alpha = getattr(target_model, "_last_pocket_feature_alpha", None)
                    if pocket_alpha is not None:
                        pocket_feature_alpha_rows.append(float(pocket_alpha.detach().float().cpu().item()))
                moe_weights = getattr(target_model, "_last_moe_weights", None)
                if moe_weights is not None and moe_weights.numel() > 0:
                    moe_weight_rows.append(moe_weights.detach().float().cpu().numpy())
                pocket_logits = getattr(target_model, "_last_pocket_aux_logits", None)
                pocket_target = getattr(target_model, "_last_pocket_aux_target", None)
                if pocket_logits is not None and pocket_logits.numel() > 0:
                    pocket_logit_batch.append(pocket_logits.float().view(-1))
                    if pocket_target is not None:
                        pocket_target_batch = pocket_target.float().view(-1)
                motif_logits = getattr(target_model, "_last_micro_motif_aux_logits", None)
                motif_target = getattr(target_model, "_last_micro_motif_aux_target", None)
                if motif_logits is not None and motif_logits.numel() > 0:
                    motif_logit_batch.append(motif_logits.float().view(-1))
                    if motif_target is not None:
                        motif_target_batch = motif_target.float().view(-1)

            avg_probs_t = torch.stack(outputs, dim=0).mean(dim=0)
            avg_probs = avg_probs_t.cpu().numpy().reshape(-1)

            labels = batch.activate_site.detach().cpu().numpy().reshape(-1)
            if model_gate_rows:
                gate_values.append(torch.stack(model_gate_rows, dim=0).mean(dim=0).detach().cpu().numpy())
                gate_targets.append(labels)
            if model_pocket_gate_rows:
                pocket_feature_gate_values.append(torch.stack(model_pocket_gate_rows, dim=0).mean(dim=0).detach().cpu().numpy())
                pocket_feature_gate_targets.append(labels)
            if pocket_logit_batch and pocket_target_batch is not None:
                pocket_logits_rows.append(torch.stack(pocket_logit_batch, dim=0).mean(dim=0).detach().cpu().numpy())
                pocket_target_rows.append(pocket_target_batch.detach().cpu().numpy())
            if motif_logit_batch and motif_target_batch is not None:
                motif_logits_rows.append(torch.stack(motif_logit_batch, dim=0).mean(dim=0).detach().cpu().numpy())
                motif_target_rows.append(motif_target_batch.detach().cpu().numpy())
            p2rank_flags.append(batch.residue_pocket_feature[:, [0, 1, 2]].detach().cpu().numpy())
            p2rank_targets.append(labels)
            pocket_counts = getattr(batch, "pomog_pocket_counts", None)
            if pocket_counts is None:
                pocket_counts = getattr(batch, "motif" + "03_pocket_counts", None)
            if pocket_counts is not None:
                count_rows.append(pocket_counts.detach().cpu().numpy().reshape(-1, 4))

            batch_ids = batch.batch.detach().cpu().numpy()
            for sample_idx, name in enumerate(list(batch.name)):
                mask = batch_ids == sample_idx
                payload = {"probs": avg_probs[mask], "labels": labels[mask]}
                full_results[str(name)] = payload
                as_probs_only[str(name)] = avg_probs[mask]

    for protein_id, payload in full_results.items():
        probs = payload["probs"]
        labels = payload["labels"]
        seq = dataset[protein_id][0] if isinstance(dataset[protein_id], (list, tuple)) and len(dataset[protein_id]) > 0 else ""
        top_idx = np.argsort(probs)[-int(args.top_k):][::-1]
        with (report_dir / f"{protein_id}.txt").open("w", encoding="utf-8") as handle:
            handle.write("Num\tAA\tScore\tLabel\n")
            for idx in top_idx:
                aa = seq[idx] if idx < len(seq) else "X"
                label = int(labels[idx]) if idx < len(labels) else 0
                handle.write(f"{idx}\t{aa}\t{float(probs[idx]):.6f}\t{label}\n")

    with (run_dir / "prediction_results.pkl").open("wb") as handle:
        pickle.dump(full_results, handle)
    with (run_dir / "as_probs.pkl").open("wb") as handle:
        pickle.dump(as_probs_only, handle)

    metrics = evaluate_pomog(full_results, allow_threshold_search=True)
    flags = np.concatenate(p2rank_flags, axis=0) if p2rank_flags else np.zeros((0, 3), dtype=np.float32)
    flag_y = np.concatenate(p2rank_targets).reshape(-1) if p2rank_targets else np.zeros((0,), dtype=np.float32)
    pos_total = max(float((flag_y > 0.5).sum()), 1.0)
    if binary_chunks:
        binary_all = np.concatenate(binary_chunks).reshape(-1)
    else:
        all_probs = np.concatenate([row["probs"].reshape(-1) for row in full_results.values()])
        threshold = metrics.get("threshold") or metrics.get("_decision_cutoff") or 0.5
        binary_all = (all_probs >= float(threshold)).astype(int)
    gates = np.concatenate(gate_values).reshape(-1) if gate_values else np.zeros((0,), dtype=np.float32)
    gate_y = np.concatenate(gate_targets).reshape(-1) if gate_targets else np.zeros((0,), dtype=np.float32)
    pocket_gates = (
        np.concatenate(pocket_feature_gate_values).reshape(-1)
        if pocket_feature_gate_values
        else np.zeros((0,), dtype=np.float32)
    )
    pocket_gate_y = (
        np.concatenate(pocket_feature_gate_targets).reshape(-1)
        if pocket_feature_gate_targets
        else np.zeros((0,), dtype=np.float32)
    )
    counts = np.concatenate(count_rows, axis=0) if count_rows else np.zeros((0, 4), dtype=np.float32)
    moe_weights = np.concatenate(moe_weight_rows, axis=0) if moe_weight_rows else np.zeros((0, 4), dtype=np.float32)
    diagnostics = {
        "positive_residue_gate_mean": float(gates[gate_y > 0.5].mean()) if gates.size and np.any(gate_y > 0.5) else 0.0,
        "negative_residue_gate_mean": float(gates[gate_y <= 0.5].mean()) if gates.size and np.any(gate_y <= 0.5) else 0.0,
        "pocket_gate_mean": float(pocket_gates.mean()) if pocket_gates.size else 0.0,
        "pocket_gate_pos_mean": float(pocket_gates[pocket_gate_y > 0.5].mean()) if pocket_gates.size and np.any(pocket_gate_y > 0.5) else 0.0,
        "pocket_gate_neg_mean": float(pocket_gates[pocket_gate_y <= 0.5].mean()) if pocket_gates.size and np.any(pocket_gate_y <= 0.5) else 0.0,
        "pocket_alpha": float(np.mean(pocket_feature_alpha_rows)) if pocket_feature_alpha_rows else 0.0,
        "pocket_aux_aupr": _binary_aupr(pocket_target_rows, pocket_logits_rows),
        "motif_aux_aupr": _binary_aupr(motif_target_rows, motif_logits_rows),
        "p2rank_top1_positive_coverage": float(((flags[:, 0] > 0.5) & (flag_y > 0.5)).sum() / pos_total) if flags.size else 0.0,
        "p2rank_top3_positive_coverage": float(((flags[:, 1] > 0.5) & (flag_y > 0.5)).sum() / pos_total) if flags.size else 0.0,
        "p2rank_top5_positive_coverage": float(((flags[:, 2] > 0.5) & (flag_y > 0.5)).sum() / pos_total) if flags.size else 0.0,
        "false_negatives_outside_p2rank_top5": int(((binary_all == 0) & (flag_y > 0.5) & (flags[:, 2] <= 0.5)).sum()) if flags.size else 0,
        "memory_usage_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "motif_count": float(counts[:, 0].mean()) if counts.size else 0.0,
        "micro_motif_count": float(counts[:, 0].mean()) if counts.size else 0.0,
        "pocket_node_count": float(counts[:, 1].mean()) if counts.size else 0.0,
        "motif_edge_count": float(counts[:, 2].mean()) if counts.size else 0.0,
        "residue_to_pocket_edge_count": float(counts[:, 3].mean()) if counts.size else 0.0,
        "moe_base_weight_mean": float(moe_weights[:, 0].mean()) if moe_weights.size else 0.0,
        "moe_p2rank_weight_mean": float(moe_weights[:, 1].mean()) if moe_weights.size else 0.0,
        "moe_fpocket_weight_mean": float(moe_weights[:, 2].mean()) if moe_weights.size else 0.0,
        "moe_consensus_weight_mean": float(moe_weights[:, 3].mean()) if moe_weights.size else 0.0,
    }
    diagnostics["gate_difference"] = diagnostics["positive_residue_gate_mean"] - diagnostics["negative_residue_gate_mean"]
    metrics.update(diagnostics)
    for key in [
        "threshold",
        "_decision_cutoff",
        "vote_threshold",
        "threshold_mode",
        "threshold_source",
        "decision_rule",
    ]:
        metrics.pop(key, None)

    summary = {
        "task": "PoMoG_infer",
        "device": str(device),
        "dataset_path": args.dataset_path,
        "feature_dir": args.feature_dir,
        "model_dir": args.model_dir,
        "ckpts": [str(path) for path in ckpts],
        "samples": len(full_results),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] PoMoG inference output: {run_dir}")


if __name__ == "__main__":
    main()
