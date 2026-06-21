from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from pomog.models.modules.pocket_ranking_loss import hard_negative_plus_loss, in_out_pocket_targets
from pomog.train.losses.pocket_ranking import (
    out_of_pocket_rescue_ranking_loss,
    pocket_hard_negative_ranking_loss,
)
from pomog.train.runtime import _unwrap


def _binary_aupr(labels: list[np.ndarray], logits: list[np.ndarray]) -> float:
    if not labels or not logits:
        return 0.0
    y = np.concatenate(labels).reshape(-1)
    p = 1.0 / (1.0 + np.exp(-np.concatenate(logits).reshape(-1)))
    if y.size == 0 or float(y.max(initial=0.0)) <= 0.0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, p)
    return float(auc(recall, precision))


def _pocket_probs_from_residue(residue_logits: torch.Tensor, pocket_member_index: torch.Tensor, pocket_count: int) -> torch.Tensor:
    if pocket_count <= 0 or pocket_member_index is None or pocket_member_index.numel() == 0:
        return residue_logits.new_zeros((pocket_count,))
    probs = torch.sigmoid(residue_logits.float().view(-1)).clamp(min=1e-6, max=1.0 - 1e-6)
    member = pocket_member_index.long().to(residue_logits.device)
    residues = member[0].clamp(min=0, max=max(int(probs.numel()) - 1, 0))
    pockets = member[1].clamp(min=0, max=max(int(pocket_count) - 1, 0))
    log_not = torch.log1p(-probs[residues])
    summed = scatter_add(log_not, pockets, dim=0, dim_size=pocket_count)
    return (1.0 - torch.exp(summed)).clamp(min=0.0, max=1.0)


def _ranking_loss(
    residue_logits: torch.Tensor,
    residue_labels: torch.Tensor,
    pocket_member_index: torch.Tensor,
    *,
    max_pos: int,
    max_neg: int,
) -> torch.Tensor:
    if pocket_member_index is None or pocket_member_index.numel() == 0:
        return residue_logits.new_zeros(())
    logits = residue_logits.view(-1)
    labels = residue_labels.float().view(-1)
    member = pocket_member_index.long().to(residue_logits.device)
    residues = member[0].clamp(min=0, max=max(int(logits.numel()) - 1, 0))
    pockets = member[1]
    losses = []
    for pocket_id in torch.unique(pockets).tolist():
        idx = residues[pockets == int(pocket_id)]
        if idx.numel() <= 1:
            continue
        pos = idx[labels[idx] > 0.5][: max(1, int(max_pos))]
        neg = idx[labels[idx] <= 0.5][: max(1, int(max_neg))]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = logits[pos].view(-1, 1) - logits[neg].view(1, -1)
        losses.append(F.softplus(-diff).mean())
    return torch.stack(losses).mean() if losses else residue_logits.new_zeros(())


def _hard_negative_ranking_loss(
    residue_logits: torch.Tensor,
    residue_labels: torch.Tensor,
    pocket_member_index: torch.Tensor,
    residue_pocket_feature: torch.Tensor,
    batch_id: torch.Tensor,
    config: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    stats = {
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
    logits = residue_logits.view(-1)
    labels = residue_labels.float().view(-1).to(logits.device)
    batch_id = batch_id.long().view(-1).to(logits.device)
    features = residue_pocket_feature.float().to(logits.device) if residue_pocket_feature is not None else logits.new_zeros((logits.numel(), 0))
    member = pocket_member_index.long().to(logits.device) if pocket_member_index is not None else logits.new_zeros((2, 0), dtype=torch.long)
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_zeros(()), stats

    if features.numel() > 0 and features.shape[-1] >= 20:
        top5_mask = (features[:, 2] > 0.5) | (features[:, 5] > 0.5)
        high_mask = (features[:, 1] > 0.5) | (features[:, 4] > 0.5)
    elif features.numel() > 0 and features.shape[-1] >= 8:
        top5_mask = (features[:, 2] > 0.5) | (features[:, 7] > 0.5)
        high_mask = (features[:, 1] > 0.5) | (features[:, 6] > 0.5)
    elif features.numel() > 0 and features.shape[-1] >= 3:
        top5_mask = features[:, 2] > 0.5
        high_mask = features[:, 1] > 0.5
    else:
        top5_mask = torch.zeros_like(labels, dtype=torch.bool)
        high_mask = torch.zeros_like(labels, dtype=torch.bool)

    n_same = max(0, int(config.get("n_same_pocket_neg", 8)))
    n_high = max(0, int(config.get("n_high_pocket_neg", 8)))
    n_easy = max(0, int(config.get("n_easy_neg", 8)))
    high_weight = float(config.get("high_pocket_rank_weight", 0.5))
    easy_weight = float(config.get("easy_rank_weight", 0.25))
    losses: list[torch.Tensor] = []
    same_margins: list[torch.Tensor] = []
    high_margins: list[torch.Tensor] = []
    easy_margins: list[torch.Tensor] = []

    for protein_id in torch.unique(batch_id).tolist():
        protein_mask = batch_id == int(protein_id)
        protein_idx = torch.where(protein_mask)[0]
        pos_idx = protein_idx[labels[protein_idx] > 0.5]
        if pos_idx.numel() == 0:
            continue
        neg_idx = protein_idx[labels[protein_idx] <= 0.5]
        if neg_idx.numel() == 0:
            continue
        high_neg = neg_idx[high_mask[neg_idx]]
        easy_neg = neg_idx[~top5_mask[neg_idx]]
        for pos in pos_idx.tolist():
            pos_tensor = torch.tensor(int(pos), device=logits.device, dtype=torch.long)
            pos_score = logits[pos_tensor]
            same_neg = logits.new_zeros((0,), dtype=torch.long)
            if member.numel() > 0:
                pocket_ids = member[1, member[0] == int(pos)]
                if pocket_ids.numel() > 0:
                    same_mask = torch.zeros((int(member.shape[1]),), device=logits.device, dtype=torch.bool)
                    for pocket_id in pocket_ids.unique().tolist():
                        same_mask |= member[1] == int(pocket_id)
                    same_res = member[0, same_mask].unique(sorted=True)
                    same_neg = same_res[labels[same_res] <= 0.5]
            if same_neg.numel() > 0 and n_same > 0:
                chosen = same_neg[:n_same]
                margin = pos_score - logits[chosen]
                losses.append(F.softplus(-margin).mean())
                same_margins.append(margin.detach().mean())
                stats["rank_same_pocket_negatives"] += float(chosen.numel())
            if high_neg.numel() > 0 and n_high > 0:
                chosen = high_neg[:n_high]
                margin = pos_score - logits[chosen]
                losses.append(high_weight * F.softplus(-margin).mean())
                high_margins.append(margin.detach().mean())
                stats["rank_high_pocket_negatives"] += float(chosen.numel())
            if easy_neg.numel() > 0 and n_easy > 0:
                chosen = easy_neg[:n_easy]
                margin = pos_score - logits[chosen]
                losses.append(easy_weight * F.softplus(-margin).mean())
                easy_margins.append(margin.detach().mean())
                stats["rank_easy_negatives"] += float(chosen.numel())

    if same_margins:
        stats["rank_same_margin"] = float(torch.stack(same_margins).mean().cpu())
    if high_margins:
        stats["rank_high_margin"] = float(torch.stack(high_margins).mean().cpu())
    if easy_margins:
        stats["rank_easy_margin"] = float(torch.stack(easy_margins).mean().cpu())
    return (torch.stack(losses).mean() if losses else logits.new_zeros(())), stats


def _in_any_pocket_from_features(residue_pocket_feature: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor:
    if residue_pocket_feature is None or residue_pocket_feature.numel() == 0:
        return torch.zeros((int(logits.numel()),), device=logits.device, dtype=torch.bool)
    feat = residue_pocket_feature.float().to(logits.device)
    if int(feat.shape[0]) != int(logits.numel()):
        fixed = logits.new_zeros((int(logits.numel()), int(feat.shape[-1])))
        take = min(int(logits.numel()), int(feat.shape[0]))
        if take > 0:
            fixed[:take] = feat[:take]
        feat = fixed
    if feat.shape[-1] >= 20:
        return (feat[:, 2] > 0.5) | (feat[:, 5] > 0.5)
    if feat.shape[-1] >= 8:
        return (feat[:, 2] > 0.5) | (feat[:, 7] > 0.5)
    if feat.shape[-1] >= 3:
        return feat[:, 2] > 0.5
    return torch.zeros((int(logits.numel()),), device=logits.device, dtype=torch.bool)


def _dssp_rsa_proxy_from_features(residue_pocket_feature: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor | None:
    if residue_pocket_feature is None or residue_pocket_feature.numel() == 0:
        return None
    feat = residue_pocket_feature.float().to(logits.device)
    if feat.shape[-1] >= 20:
        return feat[:, [14, 15]]
    if feat.shape[-1] >= 10:
        return feat[:, [8, 9]]
    return None


def _scheduled_loss_weight(config: Dict[str, Any], key: str, epoch_idx: int | None) -> float:
    base = float(config.get(key, 0.0))
    if not bool(config.get("pomog_aux_schedule_enabled", False)) or epoch_idx is None:
        return base
    start = config.get(f"{key}_start")
    if start is None:
        return base
    end_value = config.get(f"{key}_end")
    end = base if end_value is None else float(end_value)
    schedule_epochs = max(1, int(config.get("pomog_aux_schedule_epochs", 30)))
    progress = min(max(float(epoch_idx) / float(schedule_epochs), 0.0), 1.0)
    mode = str(config.get("pomog_aux_schedule_mode", "cosine_decay")).lower()
    if mode == "linear_decay":
        factor = 1.0 - progress
    else:
        factor = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    return end + (float(start) - end) * factor


def _aux_losses(
    model: nn.Module,
    batch,
    residue_logits: torch.Tensor,
    residue_labels: torch.Tensor,
    config: Dict[str, Any],
    epoch_idx: int | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    target_model = _unwrap(model)
    device = residue_logits.device
    total = residue_logits.new_zeros(())
    stats: Dict[str, float] = {
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

    pocket_logits = getattr(target_model, "_last_pocket_aux_logits", None)
    pocket_target = getattr(target_model, "_last_pocket_aux_target", None)
    pocket_w = _scheduled_loss_weight(config, "pocket_aux_weight", epoch_idx)
    if pocket_w > 0 and pocket_logits is not None and pocket_target is not None and pocket_logits.numel() > 0:
        loss = F.binary_cross_entropy_with_logits(pocket_logits, pocket_target.to(pocket_logits.device))
        total = total + pocket_w * loss
        stats["pocket_aux"] = float(loss.detach().cpu())

    micro_logits = getattr(target_model, "_last_micro_motif_aux_logits", None)
    micro_target = getattr(target_model, "_last_micro_motif_aux_target", None)
    micro_w = _scheduled_loss_weight(config, "micro_motif_aux_weight", epoch_idx)
    if micro_w > 0 and micro_logits is not None and micro_target is not None and micro_logits.numel() > 0:
        loss = F.binary_cross_entropy_with_logits(micro_logits, micro_target.to(micro_logits.device))
        total = total + micro_w * loss
        stats["micro_motif_aux"] = float(loss.detach().cpu())

    consistency_w = _scheduled_loss_weight(config, "consistency_weight", epoch_idx)
    if consistency_w > 0 and pocket_logits is not None and pocket_logits.numel() > 0:
        p_res = _pocket_probs_from_residue(residue_logits, batch.pocket_member_index, int(pocket_logits.shape[0]))
        p_pocket = torch.sigmoid(pocket_logits.float().view(-1))
        loss = F.mse_loss(p_pocket, p_res.to(p_pocket.device))
        total = total + consistency_w * loss
        stats["consistency"] = float(loss.detach().cpu())

    ranking_w = _scheduled_loss_weight(config, "ranking_weight", epoch_idx)
    if bool(config.get("use_pocket_hardneg2", False)) and "pocket_rank_loss_weight" in config:
        ranking_w = _scheduled_loss_weight(config, "pocket_rank_loss_weight", epoch_idx)
    if ranking_w > 0:
        ranking_mode = str(config.get("ranking_mode", "same_pocket")).lower()
        warmup_epochs = int(config.get("ranking_warmup_epochs", 0))
        if epoch_idx is not None and epoch_idx < warmup_epochs:
            loss = residue_logits.new_zeros(())
            detail_stats: Dict[str, float] = {}
        elif bool(config.get("use_pocket_hardneg2", False)) or ranking_mode == "pocket_hardneg2":
            features = getattr(batch, "residue_pocket_feature", None)
            logits_flat = residue_logits.view(-1)
            in_any_pocket = getattr(batch, "in_any_pocket", None)
            if in_any_pocket is None:
                in_any_pocket = _in_any_pocket_from_features(features, logits_flat)
            loss, detail_stats = pocket_hard_negative_ranking_loss(
                residue_logits,
                residue_labels,
                batch.batch,
                in_any_pocket,
                p2rank_membership=getattr(batch, "p2rank_membership", getattr(batch, "pocket_member_index", None)),
                fpocket_membership=getattr(batch, "fpocket_membership", None),
                coords_ca=batch.X[:, 1] if hasattr(batch, "X") else None,
                same_pocket_neg_per_pos=int(config.get("same_pocket_neg_per_pos", config.get("n_same_pocket_neg", 8))),
                high_pocket_neg_per_pos=int(config.get("high_pocket_neg_per_pos", config.get("n_high_pocket_neg", 8))),
                nearby_neg_per_pos=int(config.get("nearby_neg_per_pos", 8)),
                easy_neg_per_pos=int(config.get("easy_neg_per_pos", config.get("n_easy_neg", 8))),
                radius=float(config.get("nearby_rank_radius", config.get("radius", 10.0))),
                weights={
                    "same_pocket": float(config.get("same_pocket_rank_weight", 1.0)),
                    "high_pocket": float(config.get("high_pocket_rank_weight", 0.5)),
                    "nearby": float(config.get("nearby_rank_weight", 0.5)),
                    "easy": float(config.get("easy_rank_weight", 0.25)),
                },
            )
        elif ranking_mode in {"inpocket_hardneg", "in_pocket_hardneg", "hard_negative", "hardneg"}:
            loss, detail_stats = _hard_negative_ranking_loss(
                residue_logits,
                residue_labels,
                batch.pocket_member_index,
                getattr(batch, "residue_pocket_feature", None),
                batch.batch,
                config,
            )
        elif ranking_mode in {"hardneg_plus", "hard_negative_plus", "pomog_hardneg_plus", "rescuerank", "rescue_rank", "pomog_rescuerank"}:
            prior_feat = getattr(batch, "structure_prior_feat", None)
            if prior_feat is None:
                prior_feat = getattr(batch, "residue_pocket_feature", None)
            loss, detail_stats = hard_negative_plus_loss(
                residue_logits,
                residue_labels,
                prior_feat,
                batch.batch,
                getattr(batch, "edge_index", None),
                max_pos=int(config.get("ranking_max_pos", 16)),
                max_neg=int(config.get("ranking_max_neg", 32)),
                same_pocket_weight=float(config.get("hardneg_plus_same_pocket_weight", 1.0)),
                consensus_weight=float(config.get("hardneg_plus_consensus_weight", 0.5)),
                local_weight=float(config.get("hardneg_plus_local_weight", 0.5)),
                easy_weight=float(config.get("hardneg_plus_easy_weight", 0.25)),
            )
        else:
            loss = _ranking_loss(
                residue_logits,
                residue_labels,
                batch.pocket_member_index,
                max_pos=int(config.get("ranking_max_pos", 16)),
                max_neg=int(config.get("ranking_max_neg", 32)),
            )
            detail_stats = {}
        total = total + ranking_w * loss
        stats["ranking"] = float(loss.detach().cpu())
        stats.update({key: float(value) for key, value in detail_stats.items()})
    rescue_w = float(config.get("rescue_rank_loss_weight", 0.0))
    if bool(config.get("use_out_pocket_rescue_rank", False)) and rescue_w > 0.0:
        rescue_warmup = int(config.get("rescue_ranking_warmup_epochs", 0))
        if epoch_idx is not None and epoch_idx < rescue_warmup:
            loss = residue_logits.new_zeros(())
            detail_stats = {}
        else:
            features = getattr(batch, "residue_pocket_feature", None)
            logits_flat = residue_logits.view(-1)
            in_any_pocket = getattr(batch, "in_any_pocket", None)
            if in_any_pocket is None:
                in_any_pocket = _in_any_pocket_from_features(features, logits_flat)
            loss, detail_stats = out_of_pocket_rescue_ranking_loss(
                residue_logits,
                residue_labels,
                batch.batch,
                in_any_pocket,
                dssp_or_rsa_feat=_dssp_rsa_proxy_from_features(features, logits_flat),
                coords_ca=batch.X[:, 1] if hasattr(batch, "X") else None,
                neg_per_pos=int(config.get("rescue_neg_per_pos", 8)),
                radius=float(config.get("rescue_radius", 14.0)),
            )
        total = total + rescue_w * loss
        stats["rescue_rank_loss"] = float(loss.detach().cpu())
        stats.update({key: float(value) for key, value in detail_stats.items()})
    gate_w = _scheduled_loss_weight(config, "gate_margin_weight", epoch_idx)
    if gate_w > 0:
        gate = getattr(target_model, "_last_residue_gate", None)
        if gate is not None and gate.numel() > 0:
            gate_mean = gate.float().mean(dim=-1).view(-1)
            labels = residue_labels.float().view(-1).to(gate_mean.device)
            pos = gate_mean[labels > 0.5]
            neg = gate_mean[labels <= 0.5]
            if pos.numel() > 0 and neg.numel() > 0:
                margin = gate_mean.new_tensor(float(config.get("gate_margin", 0.0)))
                loss = F.relu(margin - (pos.mean() - neg.mean()))
                total = total + gate_w * loss
                stats["gate_margin"] = float(loss.detach().cpu())
    prior_feat = getattr(batch, "structure_prior_feat", None)
    pocket_route_logits = getattr(target_model, "_last_pocket_route_logits", None)
    rescue_route_logits = getattr(target_model, "_last_rescue_route_logits", None)
    if prior_feat is not None and pocket_route_logits is not None and rescue_route_logits is not None:
        route_targets = in_out_pocket_targets(residue_labels, prior_feat.to(residue_logits.device))
        pocket_w = float(config.get("dual_route_pocket_weight", 0.0))
        if pocket_w > 0.0:
            pocket_mask = route_targets["in_pocket_positive"] | route_targets["covered_negative"]
            if bool(pocket_mask.any()):
                loss = F.binary_cross_entropy_with_logits(
                    pocket_route_logits.float().view(-1)[pocket_mask],
                    residue_labels.float().view(-1).to(pocket_route_logits.device)[pocket_mask],
                )
                total = total + pocket_w * loss
                stats["dual_route_pocket"] = float(loss.detach().cpu())
        rescue_w = float(config.get("dual_route_rescue_weight", 0.0))
        if rescue_w > 0.0:
            rescue_mask = route_targets["out_pocket_positive"] | route_targets["out_pocket_negative"]
            if bool(rescue_mask.any()):
                loss = F.binary_cross_entropy_with_logits(
                    rescue_route_logits.float().view(-1)[rescue_mask],
                    residue_labels.float().view(-1).to(rescue_route_logits.device)[rescue_mask],
                )
                total = total + rescue_w * loss
                stats["dual_route_rescue"] = float(loss.detach().cpu())
    moe_head = getattr(target_model, "structure_prior_moe_v2_head", None)
    entropy_loss = getattr(moe_head, "_last_entropy_loss", None) if moe_head is not None else None
    if entropy_loss is not None:
        total = total + entropy_loss
        stats["structure_prior_moe_entropy"] = float(entropy_loss.detach().cpu())
    labels_flat = residue_labels.float().view(-1).to(residue_logits.device)
    pos_mask = labels_flat > 0.5
    neg_mask = labels_flat <= 0.5
    neg_amount = getattr(target_model, "_last_neg_suppress_amount", None)
    if neg_amount is not None and neg_amount.numel() > 0:
        neg_amount = neg_amount.float().view(-1).to(residue_logits.device)
        suppress_w = float(config.get("pomog_neg_suppress_loss_weight", 0.0))
        if suppress_w > 0.0 and bool(neg_mask.any()):
            loss = F.softplus(-neg_amount[neg_mask]).mean()
            total = total + suppress_w * loss
            stats["neg_suppress_loss"] = float(loss.detach().cpu())
        protect_w = float(config.get("pomog_neg_suppress_protect_positive_weight", 0.0))
        if protect_w > 0.0 and bool(pos_mask.any()):
            loss = neg_amount[pos_mask].abs().mean()
            total = total + protect_w * loss
            stats["neg_suppress_protect_positive"] = float(loss.detach().cpu())
    rescue_amount = getattr(target_model, "_last_rescue_boost_amount", None)
    if rescue_amount is not None and rescue_amount.numel() > 0:
        rescue_amount = rescue_amount.float().view(-1).to(residue_logits.device)
        prior_feat = getattr(batch, "structure_prior_feat", None)
        if prior_feat is not None:
            route_targets = in_out_pocket_targets(residue_labels, prior_feat.to(residue_logits.device))
            rescue_pos_mask = route_targets["out_pocket_positive"]
        else:
            rescue_pos_mask = pos_mask
        rescue_w = float(config.get("pomog_dual_delta_rescue_loss_weight", 0.0))
        if rescue_w > 0.0 and bool(rescue_pos_mask.any()):
            loss = F.softplus(-rescue_amount[rescue_pos_mask]).mean()
            total = total + rescue_w * loss
            stats["dual_delta_rescue_loss"] = float(loss.detach().cpu())
        rescue_protect_w = float(config.get("pomog_dual_delta_rescue_protect_negative_weight", 0.0))
        if rescue_protect_w > 0.0 and bool(neg_mask.any()):
            loss = F.relu(rescue_amount[neg_mask]).mean()
            total = total + rescue_protect_w * loss
            stats["dual_delta_rescue_protect_negative"] = float(loss.detach().cpu())
    prior_delta = getattr(target_model, "_last_prior_logit_delta", None)
    prior_l2_w = float(config.get("pomog_prior_logit_adapter_l2_weight", 0.0))
    if prior_l2_w > 0.0 and prior_delta is not None and prior_delta.numel() > 0:
        loss = prior_delta.float().pow(2).mean()
        total = total + prior_l2_w * loss
        stats["prior_logit_adapter_l2"] = float(loss.detach().cpu())
    return total, stats
