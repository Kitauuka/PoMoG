from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from pomog.data.structure_prior_cache import FEATURE_INDEX


def coverage_mask_from_prior(prior_feat: torch.Tensor | None) -> torch.Tensor:
    if prior_feat is None or prior_feat.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    feat = prior_feat.float()
    cols = [
        FEATURE_INDEX["p2rank_in_top5"],
        FEATURE_INDEX["fpocket_in_top5"],
        FEATURE_INDEX["caver_tunnel_lining"],
        FEATURE_INDEX["pykvfinder_cavity"],
        FEATURE_INDEX["mole_tunnel_lining"],
        FEATURE_INDEX["castp_surface_pocket"],
        FEATURE_INDEX["castp_interior_cavity"],
        FEATURE_INDEX["castp_channel"],
        FEATURE_INDEX["ghecom_pocket"],
    ]
    present = [idx for idx in cols if idx < int(feat.shape[-1])]
    if not present:
        return torch.zeros((int(feat.shape[0]),), device=feat.device, dtype=torch.bool)
    return (feat[:, present] > 0.5).any(dim=-1)


def hard_negative_plus_loss(
    residue_logits: torch.Tensor,
    residue_labels: torch.Tensor,
    prior_feat: torch.Tensor | None,
    batch_id: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    *,
    local_radius_edges: bool = True,
    max_pos: int = 16,
    max_neg: int = 32,
    same_pocket_weight: float = 1.0,
    consensus_weight: float = 0.5,
    local_weight: float = 0.5,
    easy_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = residue_logits.float().view(-1)
    labels = residue_labels.float().view(-1).to(logits.device)
    batch_id = batch_id.long().view(-1).to(logits.device)
    feat = prior_feat.float().to(logits.device) if prior_feat is not None and prior_feat.numel() > 0 else logits.new_zeros((logits.numel(), 0))
    covered = coverage_mask_from_prior(feat).to(logits.device) if feat.numel() else torch.zeros_like(labels, dtype=torch.bool)
    if feat.numel() and feat.shape[-1] > FEATURE_INDEX["detector_count"]:
        consensus = feat[:, FEATURE_INDEX["detector_count"]] >= 2.0
    else:
        consensus = covered
    stats = {
        "hardneg_plus_same_pocket_negatives": 0.0,
        "hardneg_plus_consensus_negatives": 0.0,
        "hardneg_plus_local_negatives": 0.0,
        "hardneg_plus_easy_negatives": 0.0,
        "hardneg_plus_margin": 0.0,
    }
    if logits.numel() == 0 or labels.numel() == 0:
        return logits.new_zeros(()), stats
    local_neighbors: dict[int, torch.Tensor] = {}
    if local_radius_edges and edge_index is not None and edge_index.numel() > 0:
        src, dst = edge_index.long().to(logits.device)
        for pos in torch.where(labels > 0.5)[0].tolist():
            neighbors = torch.cat([src[dst == int(pos)], dst[src == int(pos)]], dim=0).unique(sorted=True)
            local_neighbors[int(pos)] = neighbors[labels[neighbors] <= 0.5]
    losses: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    for protein_id in torch.unique(batch_id).tolist():
        protein_idx = torch.where(batch_id == int(protein_id))[0]
        pos_idx = protein_idx[labels[protein_idx] > 0.5][: max(1, int(max_pos))]
        neg_idx = protein_idx[labels[protein_idx] <= 0.5]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        covered_neg = neg_idx[covered[neg_idx]][: max(1, int(max_neg))]
        consensus_neg = neg_idx[consensus[neg_idx]][: max(1, int(max_neg))]
        easy_neg = neg_idx[~covered[neg_idx]][: max(1, int(max_neg))]
        for pos in pos_idx.tolist():
            pos_score = logits[int(pos)]
            for chosen, weight, key in (
                (covered_neg, same_pocket_weight, "hardneg_plus_same_pocket_negatives"),
                (consensus_neg, consensus_weight, "hardneg_plus_consensus_negatives"),
                (local_neighbors.get(int(pos), logits.new_zeros((0,), dtype=torch.long))[: max(1, int(max_neg))], local_weight, "hardneg_plus_local_negatives"),
                (easy_neg, easy_weight, "hardneg_plus_easy_negatives"),
            ):
                if chosen.numel() == 0 or float(weight) <= 0.0:
                    continue
                margin = pos_score - logits[chosen.long()]
                losses.append(float(weight) * F.softplus(-margin).mean())
                margins.append(margin.detach().mean())
                stats[key] += float(chosen.numel())
    if margins:
        stats["hardneg_plus_margin"] = float(torch.stack(margins).mean().detach().cpu())
    return (torch.stack(losses).mean() if losses else logits.new_zeros(())), stats


def in_out_pocket_targets(labels: torch.Tensor, prior_feat: torch.Tensor | None) -> dict[str, torch.Tensor]:
    y = labels.float().view(-1)
    covered = coverage_mask_from_prior(prior_feat).to(y.device)
    if covered.numel() == 0:
        covered = torch.zeros_like(y, dtype=torch.bool)
    return {
        "in_pocket_positive": (y > 0.5) & covered,
        "out_pocket_positive": (y > 0.5) & ~covered,
        "covered_negative": (y <= 0.5) & covered,
        "out_pocket_negative": (y <= 0.5) & ~covered,
    }
