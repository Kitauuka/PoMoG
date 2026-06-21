"""Metrics for PoMoG residue-level active-site prediction."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_aupr(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute residue-level average precision."""
    return float(average_precision_score(labels, probs))


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute residue-level ROC-AUC, returning 0 when it is undefined."""
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.0


def compute_mcc_best_thresh(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Choose the F1-optimal threshold from the PR curve and compute MCC."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * recall * precision / (recall + precision + 1e-10)

    if thresholds.size == 0:
        decision_cutoff = 0.5
    else:
        best_idx = int(np.argmax(f1[:-1])) if f1.shape[0] > thresholds.shape[0] else int(np.argmax(f1))
        best_idx = min(best_idx, thresholds.shape[0] - 1)
        decision_cutoff = float(thresholds[best_idx])

    binary_preds = (probs >= decision_cutoff).astype(int)
    return float(matthews_corrcoef(labels, binary_preds)), decision_cutoff


def _coerce_threshold(value: Any) -> float:
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError(f"PoMoG threshold must be finite, got: {value}")
    return min(max(threshold, 0.0), 1.0)


def load_pomog_threshold(path: str | Path) -> tuple[float, str]:
    """Load a fixed threshold from a JSON artifact."""
    threshold_path = Path(path)
    with threshold_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError(f"invalid PoMoG threshold artifact: {threshold_path}")
    return _coerce_threshold(payload["threshold"]), str(threshold_path)


def load_pomog_threshold_payload(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a full threshold artifact, including optional per-fold thresholds."""
    threshold_path = Path(path)
    with threshold_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError(f"invalid PoMoG threshold artifact: {threshold_path}")
    payload["threshold"] = _coerce_threshold(payload["threshold"])
    return payload, str(threshold_path)


def resolve_pomog_threshold(
    threshold: float | None = None,
    *,
    threshold_path: str | Path | None = None,
) -> tuple[float | None, str]:
    """Resolve an explicit fixed threshold or PoMoG environment variable."""
    if threshold is not None:
        return _coerce_threshold(threshold), "argument"

    path_text = str(threshold_path or os.environ.get("POMOG_THRESHOLD_PATH", "")).strip()
    if path_text:
        return load_pomog_threshold(path_text)

    env_value = os.environ.get("POMOG_THRESHOLD")
    if env_value is not None and str(env_value).strip():
        source = os.environ.get("POMOG_THRESHOLD_SOURCE") or "environment"
        return _coerce_threshold(env_value), source

    return None, "optimized_on_input"


def _threshold_artifact_path(threshold_path: str | Path | None = None) -> Path | None:
    path_text = str(threshold_path or os.environ.get("POMOG_THRESHOLD_PATH", "")).strip()
    return Path(path_text) if path_text else None


def _fold_id_from_ckpt(path: str | Path) -> str | None:
    match = re.search(r"fold(\d+)", Path(path).stem)
    if match:
        return match.group(1)
    stem = Path(path).stem.lower()
    if stem == "holdout":
        return "holdout"
    return None


def resolve_pomog_fold_thresholds(
    ckpt_paths: list[str | Path],
    threshold_path: str | Path | None = None,
) -> tuple[list[float] | None, str | None, str]:
    """Resolve optional per-fold fixed thresholds in checkpoint order."""
    artifact_path = _threshold_artifact_path(threshold_path)
    if artifact_path is None:
        return None, None, "missing"

    payload, source = load_pomog_threshold_payload(artifact_path)
    default_threshold = _coerce_threshold(payload["threshold"])
    fold_thresholds = payload.get("fold_thresholds", [])
    if not isinstance(fold_thresholds, list) or not fold_thresholds:
        return [default_threshold for _ in ckpt_paths], source, "scalar_repeated"

    by_fold: dict[str, float] = {}
    for item in fold_thresholds:
        if not isinstance(item, dict) or item.get("threshold") is None:
            continue
        by_fold[str(item.get("fold"))] = _coerce_threshold(item["threshold"])

    thresholds: list[float] = []
    used_per_fold = False
    for ckpt_path in ckpt_paths:
        fold_id = _fold_id_from_ckpt(ckpt_path)
        if fold_id is not None and fold_id in by_fold:
            thresholds.append(by_fold[fold_id])
            used_per_fold = True
        elif len(by_fold) == 1 and len(ckpt_paths) == 1:
            thresholds.append(next(iter(by_fold.values())))
            used_per_fold = True
        else:
            thresholds.append(default_threshold)

    return thresholds, source, "per_model" if used_per_fold else "scalar_repeated"


def build_pomog_threshold_payload(
    *,
    threshold: float,
    fold_thresholds: list[dict[str, Any]],
    source_summary_path: str | Path | None = None,
    strategy: str = "cv_f1_best",
    aggregation: str = "mean_fold_decision_cutoff",
) -> dict[str, Any]:
    """Build a threshold artifact payload for optional fixed-threshold runs."""
    payload: dict[str, Any] = {
        "task": "pomog",
        "threshold": _coerce_threshold(threshold),
        "threshold_strategy": strategy,
        "aggregation": aggregation,
        "fold_thresholds": fold_thresholds,
    }
    if source_summary_path is not None:
        payload["source_summary_path"] = str(source_summary_path)
    return payload


def compute_top_k_recall(results: Dict[str, Dict[str, np.ndarray]], k: int = 5) -> float:
    """Compute per-protein Top-K recall."""
    hits = 0
    total = 0
    for res in results.values():
        probs = res["probs"].flatten()
        labels = res["labels"].flatten()
        if np.sum(labels) == 0:
            continue
        total += 1
        top_k_idx = np.argsort(probs)[-k:][::-1]
        true_idx = np.where(labels == 1)[0]
        if len(set(top_k_idx) & set(true_idx)) > 0:
            hits += 1
    return hits / total if total > 0 else 0.0


def compute_confusion_counts(labels: np.ndarray, binary_preds: np.ndarray) -> tuple[int, int, int, int]:
    """Compute binary confusion-matrix counts as tp, fp, tn, fn."""
    labels = labels.astype(int)
    binary_preds = binary_preds.astype(int)
    tp = int(np.sum((labels == 1) & (binary_preds == 1)))
    fp = int(np.sum((labels == 0) & (binary_preds == 1)))
    tn = int(np.sum((labels == 0) & (binary_preds == 0)))
    fn = int(np.sum((labels == 1) & (binary_preds == 0)))
    return tp, fp, tn, fn


def evaluate_pomog(
    results: Dict[str, Dict[str, np.ndarray]],
    top_k_list: tuple[int, ...] = (5, 10),
    threshold: float | None = None,
    threshold_path: str | Path | None = None,
    allow_threshold_search: bool = False,
) -> Dict[str, Any]:
    """Aggregate PoMoG residue-level prediction metrics."""
    all_probs = np.concatenate([r["probs"].flatten() for r in results.values()])
    all_labels = np.concatenate([r["labels"].flatten() for r in results.values()])
    precomputed_binary_chunks = [
        r["binary_preds"].flatten()
        for r in results.values()
        if "binary_preds" in r
    ]
    if precomputed_binary_chunks and len(precomputed_binary_chunks) != len(results):
        raise ValueError("binary_preds must be present for every PoMoG result or for none")
    if precomputed_binary_chunks:
        binary_preds = np.concatenate(precomputed_binary_chunks).astype(int)
        mcc = float(matthews_corrcoef(all_labels, binary_preds))
        threshold_source = os.environ.get("POMOG_THRESHOLD_SOURCE") or "precomputed_binary"
        tp, fp, tn, fn = compute_confusion_counts(all_labels, binary_preds)
        metrics: Dict[str, Any] = {
            "aupr": compute_aupr(all_labels, all_probs),
            "auc": compute_auc(all_labels, all_probs),
            "mcc": mcc,
            "threshold": None,
            "_decision_cutoff": None,
            "vote_threshold": 0.5,
            "threshold_mode": "precomputed_binary",
            "threshold_source": threshold_source,
            "decision_rule": "precomputed_binary_preds",
            "precision": float(precision_score(all_labels, binary_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, binary_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, binary_preds, zero_division=0)),
            "total_residues": int(all_labels.shape[0]),
            "positive_residues": int(all_labels.sum()),
            "predicted_positive_residues": int(binary_preds.sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        for k in top_k_list:
            metrics[f"top{k}_recall"] = compute_top_k_recall(results, k=k)
        return metrics

    fixed_threshold, threshold_source = resolve_pomog_threshold(
        threshold,
        threshold_path=threshold_path,
    )
    if fixed_threshold is None:
        env_allows_search = os.environ.get("POMOG_ALLOW_THRESHOLD_SEARCH", "").strip() in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
        if not (allow_threshold_search or env_allows_search):
            raise ValueError(
                "PoMoG evaluation requires a fixed threshold. "
                "Pass threshold/threshold_path or use allow_threshold_search for diagnostic runs."
            )
        mcc, used_threshold = compute_mcc_best_thresh(all_labels, all_probs)
        threshold_mode = "optimized_on_evaluation_data"
    else:
        used_threshold = fixed_threshold
        binary_for_mcc = (all_probs >= used_threshold).astype(int)
        mcc = float(matthews_corrcoef(all_labels, binary_for_mcc))
        threshold_mode = "fixed"

    binary_preds = (all_probs >= used_threshold).astype(int)
    tp, fp, tn, fn = compute_confusion_counts(all_labels, binary_preds)
    metrics: Dict[str, Any] = {
        "aupr": compute_aupr(all_labels, all_probs),
        "auc": compute_auc(all_labels, all_probs),
        "mcc": mcc,
        "threshold": used_threshold,
        "_decision_cutoff": used_threshold,
        "threshold_mode": threshold_mode,
        "threshold_source": threshold_source,
        "precision": float(precision_score(all_labels, binary_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, binary_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, binary_preds, zero_division=0)),
        "total_residues": int(all_labels.shape[0]),
        "positive_residues": int(all_labels.sum()),
        "predicted_positive_residues": int(binary_preds.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    for k in top_k_list:
        metrics[f"top{k}_recall"] = compute_top_k_recall(results, k=k)
    return metrics
