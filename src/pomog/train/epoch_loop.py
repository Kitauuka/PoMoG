from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pomog.data.structure_prior_cache import DETECTOR_NAMES
from pomog.metrics import compute_mcc_best_thresh
from pomog.train.auxiliary_losses import _aux_losses, _binary_aupr
from pomog.train.runtime import _amp_mode, _autocast, _model_forward, _unwrap
from pomog.train.train_utils import _compute_loss_components, _confidence_sample_weight


def _update_topk(stats: dict[int, list[int]], probs: np.ndarray, labels: np.ndarray, batch_index: np.ndarray) -> None:
    for sample_id in np.unique(batch_index):
        mask = batch_index == sample_id
        sample_probs = probs[mask]
        sample_labels = labels[mask]
        if sample_labels.size == 0 or sample_labels.sum() <= 0:
            continue
        for k, row in stats.items():
            top_idx = np.argsort(sample_probs)[-int(k):]
            row[0] += int(np.intersect1d(top_idx, np.where(sample_labels > 0.5)[0]).size > 0)
            row[1] += 1


def _train_epoch(model, loader, optimizer, criterion, device, config: Dict[str, Any], epoch_idx: int):
    model.train()
    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice": 0.0,
        "focal": 0.0,
        "contrastive": 0.0,
        "contrastive_raw": 0.0,
        "contrastive_w": 0.0,
        "contrastive_sampled": 0.0,
        "contrastive_valid_anchors": 0.0,
        "contrastive_types": 0.0,
        "contrastive_positive": 0.0,
        "contrastive_negative": 0.0,
        "contrastive_pos_anchors": 0.0,
        "contrastive_neg_anchors": 0.0,
        "aux": 0.0,
        "pocket_aux": 0.0,
        "micro_motif_aux": 0.0,
        "consistency": 0.0,
        "ranking": 0.0,
        "gate_margin": 0.0,
        "rank_same_pocket_negatives": 0.0,
        "rank_high_pocket_negatives": 0.0,
        "rank_easy_negatives": 0.0,
        "rank_same_margin": 0.0,
        "rank_high_margin": 0.0,
        "rank_easy_margin": 0.0,
        "rank_loss_same_pocket": 0.0,
        "rank_loss_high_pocket": 0.0,
        "rank_loss_nearby": 0.0,
        "rank_loss_easy": 0.0,
        "n_same_pocket_pairs": 0.0,
        "n_high_pocket_pairs": 0.0,
        "n_nearby_pairs": 0.0,
        "n_easy_pairs": 0.0,
        "margin_same_pocket_mean": 0.0,
        "margin_high_pocket_mean": 0.0,
        "margin_nearby_mean": 0.0,
        "margin_easy_mean": 0.0,
        "rescue_rank_loss": 0.0,
        "n_out_pocket_pos": 0.0,
        "n_rescue_pairs": 0.0,
        "rescue_margin_mean": 0.0,
        "hardneg_plus_same_pocket_negatives": 0.0,
        "hardneg_plus_consensus_negatives": 0.0,
        "hardneg_plus_local_negatives": 0.0,
        "hardneg_plus_easy_negatives": 0.0,
        "hardneg_plus_margin": 0.0,
        "dual_route_pocket": 0.0,
        "dual_route_rescue": 0.0,
        "structure_prior_moe_entropy": 0.0,
        "neg_suppress_loss": 0.0,
        "neg_suppress_protect_positive": 0.0,
        "dual_delta_rescue_loss": 0.0,
        "dual_delta_rescue_protect_negative": 0.0,
        "prior_logit_adapter_l2": 0.0,
    }
    fetch_seconds = 0.0
    compute_seconds = 0.0
    steps = 0
    amp_mode = _amp_mode(config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_mode == "fp16")
    loader_iter = iter(loader)
    while True:
        fetch_start = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        fetch_seconds += time.time() - fetch_start
        compute_start = time.time()
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(amp_mode):
            pred, projection, _ = _model_forward(model, batch, return_projection=True)
            target = batch.activate_site.float().unsqueeze(1)
            loss, loss_comp = _compute_loss_components(
                pred,
                target,
                projection,
                batch.seq,
                criterion,
                config,
                epoch_idx=epoch_idx,
                random_sampling=True,
                sample_weight=_confidence_sample_weight(batch, config),
            )
            aux_loss, aux_stats = _aux_losses(model, batch, pred, target, config, epoch_idx=epoch_idx)
            loss = loss + aux_loss
        if amp_mode == "fp16":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        compute_seconds += time.time() - compute_start
        for key in totals:
            if key == "loss":
                totals[key] += float(loss.item())
            elif key == "aux":
                totals[key] += float(aux_loss.detach().cpu())
            elif key in aux_stats:
                totals[key] += float(aux_stats[key])
            else:
                totals[key] += float(loss_comp.get(key, 0.0))
        steps += 1
    denom = max(steps, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["fetch_seconds"] = fetch_seconds
    metrics["compute_seconds"] = compute_seconds
    metrics["memory_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    return metrics


def _eval_epoch(model, loader, criterion, device, config: Dict[str, Any], epoch_idx: int):
    model.eval()
    preds, targets = [], []
    loss_rows = []
    gate_values, gate_targets = [], []
    pocket_feature_gate_values, pocket_feature_gate_targets = [], []
    pocket_feature_alpha_rows = []
    route_gate_values, route_gate_targets = [], []
    neg_gate_values, neg_delta_values, final_delta_values = [], [], []
    rescue_gate_values, rescue_delta_values, prior_delta_values = [], [], []
    correction_targets = []
    pocket_logits_rows, pocket_target_rows = [], []
    micro_logits_rows, micro_target_rows = [], []
    p2rank_flags, p2rank_targets = [], []
    moe_weight_rows = []
    structure_prior_moe_rows = []
    structure_prior_missing_rows = []
    structure_prior_count_rows = []
    count_rows = []
    topk_stats = {5: [0, 0], 10: [0, 0]}
    fetch_seconds = 0.0
    compute_seconds = 0.0
    amp_mode = _amp_mode(config, device)
    with torch.no_grad():
        loader_iter = iter(loader)
        while True:
            fetch_start = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            fetch_seconds += time.time() - fetch_start
            compute_start = time.time()
            batch = batch.to(device)
            with _autocast(amp_mode):
                pred, projection, _ = _model_forward(model, batch, return_projection=True)
                target = batch.activate_site.float().unsqueeze(1)
                loss, loss_comp = _compute_loss_components(
                    pred,
                    target,
                    projection,
                    batch.seq,
                    criterion,
                    config,
                    epoch_idx=epoch_idx,
                    random_sampling=False,
                    sample_weight=_confidence_sample_weight(batch, config),
                )
                aux_loss, aux_stats = _aux_losses(model, batch, pred, target, config, epoch_idx=epoch_idx)
                loss = loss + aux_loss
            loss_comp["loss"] = float(loss.detach().cpu())
            loss_comp["aux"] = float(aux_loss.detach().cpu())
            loss_comp.update(aux_stats)
            loss_rows.append(loss_comp)
            prob_np = torch.sigmoid(pred.float()).cpu().numpy().reshape(-1)
            target_np = target.cpu().numpy().reshape(-1)
            preds.append(prob_np)
            targets.append(target_np)
            _update_topk(topk_stats, prob_np, target_np, batch.batch.detach().cpu().numpy().reshape(-1))

            target_model = _unwrap(model)
            gate = getattr(target_model, "_last_residue_gate", None)
            if gate is not None and gate.numel() > 0:
                gate_values.append(gate.float().mean(dim=-1).detach().cpu().numpy())
                gate_targets.append(target_np)
            pocket_feature_gate = getattr(target_model, "_last_pocket_feature_gate", None)
            if pocket_feature_gate is not None and pocket_feature_gate.numel() > 0:
                pocket_feature_gate_values.append(pocket_feature_gate.float().mean(dim=-1).detach().cpu().numpy())
                pocket_feature_gate_targets.append(target_np)
                pocket_alpha = getattr(target_model, "_last_pocket_feature_alpha", None)
                if pocket_alpha is not None:
                    pocket_feature_alpha_rows.append(float(pocket_alpha.detach().float().cpu().item()))
            pocket_logits = getattr(target_model, "_last_pocket_aux_logits", None)
            pocket_target = getattr(target_model, "_last_pocket_aux_target", None)
            if pocket_logits is not None and pocket_target is not None and pocket_logits.numel() > 0:
                pocket_logits_rows.append(pocket_logits.detach().float().cpu().numpy())
                pocket_target_rows.append(pocket_target.detach().float().cpu().numpy())
            micro_logits = getattr(target_model, "_last_micro_motif_aux_logits", None)
            micro_target = getattr(target_model, "_last_micro_motif_aux_target", None)
            if micro_logits is not None and micro_target is not None and micro_logits.numel() > 0:
                micro_logits_rows.append(micro_logits.detach().float().cpu().numpy())
                micro_target_rows.append(micro_target.detach().float().cpu().numpy())
            if hasattr(batch, "residue_pocket_feature"):
                p2rank_flags.append(batch.residue_pocket_feature[:, [0, 1, 2]].detach().float().cpu().numpy())
                p2rank_targets.append(target_np)
            if hasattr(batch, "pomog_pocket_counts"):
                count_rows.append(batch.pomog_pocket_counts.detach().float().cpu().numpy().reshape(-1, 4))
            moe_weights = getattr(target_model, "_last_moe_weights", None)
            if moe_weights is not None and moe_weights.numel() > 0:
                moe_weight_rows.append(moe_weights.detach().float().cpu().numpy())
            route_gate = getattr(target_model, "_last_dual_route_gate", None)
            if route_gate is not None and route_gate.numel() > 0:
                route_gate_values.append(route_gate.detach().float().cpu().numpy().reshape(-1))
                route_gate_targets.append(target_np)
            neg_gate = getattr(target_model, "_last_neg_suppress_gate", None)
            neg_delta = getattr(target_model, "_last_neg_suppress_amount", None)
            final_delta = getattr(target_model, "_last_final_delta_logits", None)
            rescue_gate = getattr(target_model, "_last_rescue_boost_gate", None)
            rescue_delta = getattr(target_model, "_last_rescue_boost_amount", None)
            prior_delta = getattr(target_model, "_last_prior_logit_delta", None)
            if any(x is not None and x.numel() > 0 for x in (neg_gate, neg_delta, final_delta, rescue_gate, rescue_delta, prior_delta)):
                correction_targets.append(target_np)
                if neg_gate is not None and neg_gate.numel() > 0:
                    neg_gate_values.append(neg_gate.detach().float().cpu().numpy().reshape(-1))
                if neg_delta is not None and neg_delta.numel() > 0:
                    neg_delta_values.append(neg_delta.detach().float().cpu().numpy().reshape(-1))
                if final_delta is not None and final_delta.numel() > 0:
                    final_delta_values.append(final_delta.detach().float().cpu().numpy().reshape(-1))
                if rescue_gate is not None and rescue_gate.numel() > 0:
                    rescue_gate_values.append(rescue_gate.detach().float().cpu().numpy().reshape(-1))
                if rescue_delta is not None and rescue_delta.numel() > 0:
                    rescue_delta_values.append(rescue_delta.detach().float().cpu().numpy().reshape(-1))
                if prior_delta is not None and prior_delta.numel() > 0:
                    prior_delta_values.append(prior_delta.detach().float().cpu().numpy().reshape(-1))
            structure_prior_weights = getattr(target_model, "_last_structure_prior_moe_v2_weights", None)
            if structure_prior_weights is not None and structure_prior_weights.numel() > 0:
                structure_prior_moe_rows.append(structure_prior_weights.detach().float().cpu().numpy())
            if hasattr(batch, "structure_prior_detector_missing"):
                structure_prior_missing_rows.append(
                    batch.structure_prior_detector_missing.detach().float().cpu().numpy().reshape(-1, len(DETECTOR_NAMES))
                )
            if hasattr(batch, "structure_prior_counts"):
                structure_prior_count_rows.append(batch.structure_prior_counts.detach().float().cpu().numpy().reshape(-1, 5))
            compute_seconds += time.time() - compute_start

    flat_preds = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.float32)
    flat_targets = np.concatenate(targets) if targets else np.zeros((0,), dtype=np.float32)
    try:
        val_auc = roc_auc_score(flat_targets, flat_preds)
        precision, recall, _ = precision_recall_curve(flat_targets, flat_preds)
        val_aupr = auc(recall, precision)
    except ValueError:
        val_auc = val_aupr = 0.0
    val_mcc, val_best_thresh = compute_mcc_best_thresh(flat_targets, flat_preds)
    val_binary_preds = (flat_preds >= val_best_thresh).astype(int)
    val_f1 = float(f1_score(flat_targets, val_binary_preds, zero_division=0))
    val_precision = float(precision_score(flat_targets, val_binary_preds, zero_division=0))
    val_recall = float(recall_score(flat_targets, val_binary_preds, zero_division=0))
    val_false_positive = int(((val_binary_preds == 1) & (flat_targets < 0.5)).sum())

    def mean_metric(key: str) -> float:
        return float(np.mean([row.get(key, 0.0) for row in loss_rows]) if loss_rows else 0.0)

    if gate_values:
        gates = np.concatenate(gate_values).reshape(-1)
        gate_y = np.concatenate(gate_targets).reshape(-1)
        pos_gate = float(gates[gate_y > 0.5].mean()) if np.any(gate_y > 0.5) else 0.0
        neg_gate = float(gates[gate_y <= 0.5].mean()) if np.any(gate_y <= 0.5) else 0.0
    else:
        pos_gate = neg_gate = 0.0
    if pocket_feature_gate_values:
        pocket_gates = np.concatenate(pocket_feature_gate_values).reshape(-1)
        pocket_gate_y = np.concatenate(pocket_feature_gate_targets).reshape(-1)
        pocket_gate_mean = float(pocket_gates.mean()) if pocket_gates.size else 0.0
        pocket_gate_pos = float(pocket_gates[pocket_gate_y > 0.5].mean()) if np.any(pocket_gate_y > 0.5) else 0.0
        pocket_gate_neg = float(pocket_gates[pocket_gate_y <= 0.5].mean()) if np.any(pocket_gate_y <= 0.5) else 0.0
    else:
        pocket_gate_mean = pocket_gate_pos = pocket_gate_neg = 0.0
    pocket_gate_alpha = float(np.mean(pocket_feature_alpha_rows)) if pocket_feature_alpha_rows else 0.0
    if route_gate_values:
        route_gate = np.concatenate(route_gate_values).reshape(-1)
        route_y = np.concatenate(route_gate_targets).reshape(-1)
        pocket_route_pos = float(route_gate[route_y > 0.5].mean()) if np.any(route_y > 0.5) else 0.0
        pocket_route_neg = float(route_gate[route_y <= 0.5].mean()) if np.any(route_y <= 0.5) else 0.0
    else:
        pocket_route_pos = pocket_route_neg = 0.0
    corr_y = np.concatenate(correction_targets).reshape(-1) if correction_targets else np.zeros((0,), dtype=np.float32)

    def pos_neg_mean(rows: list[np.ndarray]) -> tuple[float, float]:
        if not rows or corr_y.size == 0:
            return 0.0, 0.0
        values = np.concatenate(rows).reshape(-1)
        if values.shape[0] != corr_y.shape[0]:
            return 0.0, 0.0
        pos = float(values[corr_y > 0.5].mean()) if np.any(corr_y > 0.5) else 0.0
        neg = float(values[corr_y <= 0.5].mean()) if np.any(corr_y <= 0.5) else 0.0
        return pos, neg

    neg_gate_pos, neg_gate_neg = pos_neg_mean(neg_gate_values)
    neg_delta_pos, neg_delta_neg = pos_neg_mean(neg_delta_values)
    final_delta_pos, final_delta_neg = pos_neg_mean(final_delta_values)
    rescue_gate_pos, rescue_gate_neg = pos_neg_mean(rescue_gate_values)
    rescue_delta_pos, rescue_delta_neg = pos_neg_mean(rescue_delta_values)
    prior_delta_pos, prior_delta_neg = pos_neg_mean(prior_delta_values)
    if p2rank_flags:
        flags = np.concatenate(p2rank_flags, axis=0)
        flag_y = np.concatenate(p2rank_targets).reshape(-1)
        pos_total = max(float((flag_y > 0.5).sum()), 1.0)
        p2rank_top1 = float(((flags[:, 0] > 0.5) & (flag_y > 0.5)).sum() / pos_total)
        p2rank_top3 = float(((flags[:, 1] > 0.5) & (flag_y > 0.5)).sum() / pos_total)
        p2rank_top5 = float(((flags[:, 2] > 0.5) & (flag_y > 0.5)).sum() / pos_total)
        false_neg_outside_top5 = int(((val_binary_preds == 0) & (flat_targets > 0.5) & (flags[:, 2] <= 0.5)).sum())
    else:
        p2rank_top1 = p2rank_top3 = p2rank_top5 = 0.0
        false_neg_outside_top5 = 0
    counts = np.concatenate(count_rows, axis=0) if count_rows else np.zeros((0, 4), dtype=np.float32)
    moe_weights = np.concatenate(moe_weight_rows, axis=0) if moe_weight_rows else np.zeros((0, 4), dtype=np.float32)
    structure_prior_weights = np.concatenate(structure_prior_moe_rows, axis=0) if structure_prior_moe_rows else np.zeros((0, 6), dtype=np.float32)
    structure_prior_missing = (
        np.concatenate(structure_prior_missing_rows, axis=0)
        if structure_prior_missing_rows
        else np.zeros((0, len(DETECTOR_NAMES)), dtype=np.float32)
    )
    structure_prior_counts = np.concatenate(structure_prior_count_rows, axis=0) if structure_prior_count_rows else np.zeros((0, 5), dtype=np.float32)

    return {
        "auc": float(val_auc),
        "aupr": float(val_aupr),
        "f1": val_f1,
        "mcc": float(val_mcc),
        "precision": val_precision,
        "recall": val_recall,
        "false_positive": val_false_positive,
        "_decision_cutoff": float(val_best_thresh),
        "loss": mean_metric("loss"),
        "bce": mean_metric("bce"),
        "dice": mean_metric("dice"),
        "focal": mean_metric("focal"),
        "contrastive": mean_metric("contrastive"),
        "contrastive_raw": mean_metric("contrastive_raw"),
        "contrastive_w": mean_metric("contrastive_w"),
        "contrastive_sampled": mean_metric("contrastive_sampled"),
        "contrastive_valid_anchors": mean_metric("contrastive_valid_anchors"),
        "contrastive_types": mean_metric("contrastive_types"),
        "contrastive_positive": mean_metric("contrastive_positive"),
        "contrastive_negative": mean_metric("contrastive_negative"),
        "contrastive_pos_anchors": mean_metric("contrastive_pos_anchors"),
        "contrastive_neg_anchors": mean_metric("contrastive_neg_anchors"),
        "aux": mean_metric("aux"),
        "pocket_aux": mean_metric("pocket_aux"),
        "micro_motif_aux": mean_metric("micro_motif_aux"),
        "consistency": mean_metric("consistency"),
        "ranking": mean_metric("ranking"),
        "gate_margin": mean_metric("gate_margin"),
        "rank_same_pocket_negatives": mean_metric("rank_same_pocket_negatives"),
        "rank_high_pocket_negatives": mean_metric("rank_high_pocket_negatives"),
        "rank_easy_negatives": mean_metric("rank_easy_negatives"),
        "rank_same_margin": mean_metric("rank_same_margin"),
        "rank_high_margin": mean_metric("rank_high_margin"),
        "rank_easy_margin": mean_metric("rank_easy_margin"),
        "rank_loss_same_pocket": mean_metric("rank_loss_same_pocket"),
        "rank_loss_high_pocket": mean_metric("rank_loss_high_pocket"),
        "rank_loss_nearby": mean_metric("rank_loss_nearby"),
        "rank_loss_easy": mean_metric("rank_loss_easy"),
        "n_same_pocket_pairs": mean_metric("n_same_pocket_pairs"),
        "n_high_pocket_pairs": mean_metric("n_high_pocket_pairs"),
        "n_nearby_pairs": mean_metric("n_nearby_pairs"),
        "n_easy_pairs": mean_metric("n_easy_pairs"),
        "margin_same_pocket_mean": mean_metric("margin_same_pocket_mean"),
        "margin_high_pocket_mean": mean_metric("margin_high_pocket_mean"),
        "margin_nearby_mean": mean_metric("margin_nearby_mean"),
        "margin_easy_mean": mean_metric("margin_easy_mean"),
        "rescue_rank_loss": mean_metric("rescue_rank_loss"),
        "n_out_pocket_pos": mean_metric("n_out_pocket_pos"),
        "n_rescue_pairs": mean_metric("n_rescue_pairs"),
        "rescue_margin_mean": mean_metric("rescue_margin_mean"),
        "hardneg_plus_same_pocket_negatives": mean_metric("hardneg_plus_same_pocket_negatives"),
        "hardneg_plus_consensus_negatives": mean_metric("hardneg_plus_consensus_negatives"),
        "hardneg_plus_local_negatives": mean_metric("hardneg_plus_local_negatives"),
        "hardneg_plus_easy_negatives": mean_metric("hardneg_plus_easy_negatives"),
        "hardneg_plus_margin": mean_metric("hardneg_plus_margin"),
        "dual_route_pocket": mean_metric("dual_route_pocket"),
        "dual_route_rescue": mean_metric("dual_route_rescue"),
        "structure_prior_moe_entropy": mean_metric("structure_prior_moe_entropy"),
        "neg_suppress_loss": mean_metric("neg_suppress_loss"),
        "neg_suppress_protect_positive": mean_metric("neg_suppress_protect_positive"),
        "dual_delta_rescue_loss": mean_metric("dual_delta_rescue_loss"),
        "dual_delta_rescue_protect_negative": mean_metric("dual_delta_rescue_protect_negative"),
        "prior_logit_adapter_l2": mean_metric("prior_logit_adapter_l2"),
        "top5_recall": float(topk_stats[5][0] / max(topk_stats[5][1], 1)),
        "top10_recall": float(topk_stats[10][0] / max(topk_stats[10][1], 1)),
        "positive_residue_gate_mean": pos_gate,
        "negative_residue_gate_mean": neg_gate,
        "gate_difference": pos_gate - neg_gate,
        "pocket_gate_mean": pocket_gate_mean,
        "pocket_gate_pos_mean": pocket_gate_pos,
        "pocket_gate_neg_mean": pocket_gate_neg,
        "pocket_alpha": pocket_gate_alpha,
        "pocket_route_gate_mean_pos": pocket_route_pos,
        "pocket_route_gate_mean_neg": pocket_route_neg,
        "rescue_route_gate_mean_pos": 1.0 - pocket_route_pos if route_gate_values else 0.0,
        "rescue_route_gate_mean_neg": 1.0 - pocket_route_neg if route_gate_values else 0.0,
        "neg_suppress_gate_mean_pos": neg_gate_pos,
        "neg_suppress_gate_mean_neg": neg_gate_neg,
        "neg_suppress_delta_logit_pos": neg_delta_pos,
        "neg_suppress_delta_logit_neg": neg_delta_neg,
        "rescue_boost_gate_mean_pos": rescue_gate_pos,
        "rescue_boost_gate_mean_neg": rescue_gate_neg,
        "rescue_boost_delta_logit_pos": rescue_delta_pos,
        "rescue_boost_delta_logit_neg": rescue_delta_neg,
        "prior_logit_delta_pos": prior_delta_pos,
        "prior_logit_delta_neg": prior_delta_neg,
        "final_delta_logit_pos": final_delta_pos,
        "final_delta_logit_neg": final_delta_neg,
        "pocket_aux_aupr": _binary_aupr(pocket_target_rows, pocket_logits_rows),
        "motif_aux_aupr": _binary_aupr(micro_target_rows, micro_logits_rows),
        "p2rank_top1_positive_coverage": p2rank_top1,
        "p2rank_top3_positive_coverage": p2rank_top3,
        "p2rank_top5_positive_coverage": p2rank_top5,
        "false_negatives_outside_p2rank_top5": false_neg_outside_top5,
        "memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "motif_count": float(counts[:, 0].mean()) if counts.size else 0.0,
        "micro_motif_count": float(counts[:, 0].mean()) if counts.size else 0.0,
        "pocket_node_count": float(counts[:, 1].mean()) if counts.size else 0.0,
        "motif_edge_count": float(counts[:, 2].mean()) if counts.size else 0.0,
        "residue_to_pocket_edge_count": float(counts[:, 3].mean()) if counts.size else 0.0,
        "moe_base_weight_mean": float(moe_weights[:, 0].mean()) if moe_weights.size else 0.0,
        "moe_p2rank_weight_mean": float(moe_weights[:, 1].mean()) if moe_weights.size else 0.0,
        "moe_fpocket_weight_mean": float(moe_weights[:, 2].mean()) if moe_weights.size else 0.0,
        "moe_consensus_weight_mean": float(moe_weights[:, 3].mean()) if moe_weights.size else 0.0,
        "structure_prior_missing_rate": float(structure_prior_missing.mean()) if structure_prior_missing.size else 0.0,
        "structure_prior_pocket_count": float(structure_prior_counts[:, 0].mean()) if structure_prior_counts.size else 0.0,
        "structure_prior_residue_to_pocket_edge_count": float(structure_prior_counts[:, 1].mean()) if structure_prior_counts.size else 0.0,
        "structure_prior_mask_observed_fraction": float(structure_prior_counts[:, 2].mean()) if structure_prior_counts.size else 0.0,
        "structure_prior_moe_base_weight_mean": float(structure_prior_weights[:, 0].mean()) if structure_prior_weights.size else 0.0,
        "structure_prior_moe_p2rank_weight_mean": float(structure_prior_weights[:, 1].mean()) if structure_prior_weights.size else 0.0,
        "structure_prior_moe_fpocket_weight_mean": float(structure_prior_weights[:, 2].mean()) if structure_prior_weights.size else 0.0,
        "structure_prior_moe_castp_weight_mean": float(structure_prior_weights[:, 3].mean()) if structure_prior_weights.size else 0.0,
        "structure_prior_moe_caver_weight_mean": float(structure_prior_weights[:, 4].mean()) if structure_prior_weights.size else 0.0,
        "structure_prior_moe_lee_weight_mean": float(structure_prior_weights[:, 5].mean()) if structure_prior_weights.size else 0.0,
        "fetch_seconds": fetch_seconds,
        "compute_seconds": compute_seconds,
    }
