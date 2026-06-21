from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def _as_flat_bool(value: Tensor | None, like: Tensor) -> Tensor:
    if value is None or value.numel() == 0:
        return torch.zeros_like(like, dtype=torch.bool)
    out = value.to(like.device).view(-1)
    if out.numel() != like.numel():
        fixed = torch.zeros_like(like, dtype=torch.bool)
        take = min(int(out.numel()), int(like.numel()))
        fixed[:take] = out[:take].bool()
        return fixed
    return out.bool()


def _same_pocket_negatives(
    pos: int,
    labels: Tensor,
    membership: Tensor | None,
) -> Tensor:
    if membership is None or membership.numel() == 0:
        return labels.new_zeros((0,), dtype=torch.long)
    member = membership.to(labels.device)
    if member.ndim == 2 and int(member.shape[0]) == 2:
        residues = member[0].long()
        pocket_ids = member[1].long()
        ids = pocket_ids[residues == int(pos)]
        if ids.numel() == 0:
            return labels.new_zeros((0,), dtype=torch.long)
        same_mask = torch.zeros((int(member.shape[1]),), device=labels.device, dtype=torch.bool)
        for pocket_id in ids.unique().tolist():
            same_mask |= pocket_ids == int(pocket_id)
        same_res = residues[same_mask].unique(sorted=True)
        return same_res[labels[same_res] <= 0.5]
    if member.ndim == 2 and int(member.shape[0]) == labels.numel():
        row = member[int(pos)].bool()
        if not bool(row.any()):
            return labels.new_zeros((0,), dtype=torch.long)
        same = (member.bool() & row.view(1, -1)).any(dim=-1)
        same[int(pos)] = False
        idx = torch.where(same)[0]
        return idx[labels[idx] <= 0.5]
    return labels.new_zeros((0,), dtype=torch.long)


def _top_by_logit(indices: Tensor, logits: Tensor, limit: int) -> Tensor:
    if indices.numel() == 0 or int(limit) <= 0:
        return indices.new_zeros((0,))
    indices = indices.long()
    k = min(int(limit), int(indices.numel()))
    order = torch.argsort(logits[indices], descending=True)[:k]
    return indices[order]


def _nearby_negatives(pos: int, protein_neg: Tensor, coords_ca: Tensor | None, radius: float) -> Tensor:
    if coords_ca is None or coords_ca.numel() == 0 or protein_neg.numel() == 0:
        return protein_neg.new_zeros((0,))
    coords = coords_ca.float().to(protein_neg.device)
    if int(pos) >= int(coords.shape[0]):
        return protein_neg.new_zeros((0,))
    dist = torch.linalg.norm(coords[protein_neg.long()] - coords[int(pos)].view(1, 3), dim=-1)
    return protein_neg[dist <= float(radius)]


def _component_stats(losses: list[Tensor], margins: list[Tensor], pair_count: int, logits: Tensor) -> tuple[Tensor, float, float, int]:
    if losses:
        loss = torch.stack(losses).mean()
        loss_value = float(loss.detach().cpu())
    else:
        loss = logits.new_zeros(())
        loss_value = 0.0
    margin_value = float(torch.stack(margins).mean().detach().cpu()) if margins else 0.0
    return loss, loss_value, margin_value, int(pair_count)


def pocket_hard_negative_ranking_loss(
    logits: Tensor,
    labels: Tensor,
    batch_index: Tensor,
    in_any_pocket: Tensor,
    p2rank_membership: Tensor | None = None,
    fpocket_membership: Tensor | None = None,
    coords_ca: Tensor | None = None,
    same_pocket_neg_per_pos: int = 8,
    high_pocket_neg_per_pos: int = 8,
    easy_neg_per_pos: int = 8,
    nearby_neg_per_pos: int = 8,
    radius: float = 10.0,
    margin_mode: str = "softplus",
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    del margin_mode
    score = logits.float().view(-1)
    y = labels.float().view(-1).to(score.device)
    batch = batch_index.long().view(-1).to(score.device)
    in_pocket = _as_flat_bool(in_any_pocket, y)
    p2_member = p2rank_membership.to(score.device) if p2rank_membership is not None and p2rank_membership.numel() > 0 else None
    fp_member = fpocket_membership.to(score.device) if fpocket_membership is not None and fpocket_membership.numel() > 0 else None
    coords = coords_ca.float().to(score.device) if coords_ca is not None and coords_ca.numel() > 0 else None
    weights = dict(weights or {})
    weights.setdefault("same_pocket", 1.0)
    weights.setdefault("high_pocket", 0.5)
    weights.setdefault("nearby", 0.5)
    weights.setdefault("easy", 0.25)

    buckets: dict[str, list[Tensor]] = {key: [] for key in ("same_pocket", "high_pocket", "nearby", "easy")}
    margins: dict[str, list[Tensor]] = {key: [] for key in buckets}
    pairs = {key: 0 for key in buckets}
    if score.numel() == 0 or y.numel() == 0:
        return score.new_zeros(()), _hardneg_stats(score, buckets, margins, pairs)

    for protein_id in torch.unique(batch).tolist():
        protein_idx = torch.where(batch == int(protein_id))[0]
        pos_idx = protein_idx[y[protein_idx] > 0.5]
        neg_idx = protein_idx[y[protein_idx] <= 0.5]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        high_neg_base = _top_by_logit(neg_idx[in_pocket[neg_idx]], score, int(high_pocket_neg_per_pos))
        easy_neg_base = _top_by_logit(neg_idx[~in_pocket[neg_idx]], score, int(easy_neg_per_pos))
        for pos in pos_idx.tolist():
            pos_score = score[int(pos)]
            same_neg = torch.cat(
                [
                    _same_pocket_negatives(int(pos), y, p2_member),
                    _same_pocket_negatives(int(pos), y, fp_member),
                ],
                dim=0,
            ).unique(sorted=True)
            same_neg = same_neg[batch[same_neg] == int(protein_id)]
            nearby_neg = _top_by_logit(
                _nearby_negatives(int(pos), neg_idx, coords, radius),
                score,
                int(nearby_neg_per_pos),
            )
            for key, chosen in (
                ("same_pocket", _top_by_logit(same_neg, score, int(same_pocket_neg_per_pos))),
                ("high_pocket", high_neg_base),
                ("nearby", nearby_neg),
                ("easy", easy_neg_base),
            ):
                if chosen.numel() == 0:
                    continue
                margin = pos_score - score[chosen.long()]
                buckets[key].append(F.softplus(-margin).mean())
                margins[key].append(margin.detach().mean())
                pairs[key] += int(chosen.numel())

    total = score.new_zeros(())
    for key, rows in buckets.items():
        if rows:
            total = total + float(weights.get(key, 0.0)) * torch.stack(rows).mean()
    return total, _hardneg_stats(score, buckets, margins, pairs)


def _hardneg_stats(
    logits: Tensor,
    buckets: dict[str, list[Tensor]],
    margins: dict[str, list[Tensor]],
    pairs: dict[str, int],
) -> dict[str, float]:
    same_loss, same_value, same_margin, same_pairs = _component_stats(
        buckets.get("same_pocket", []), margins.get("same_pocket", []), pairs.get("same_pocket", 0), logits
    )
    del same_loss
    high_loss, high_value, high_margin, high_pairs = _component_stats(
        buckets.get("high_pocket", []), margins.get("high_pocket", []), pairs.get("high_pocket", 0), logits
    )
    del high_loss
    nearby_loss, nearby_value, nearby_margin, nearby_pairs = _component_stats(
        buckets.get("nearby", []), margins.get("nearby", []), pairs.get("nearby", 0), logits
    )
    del nearby_loss
    easy_loss, easy_value, easy_margin, easy_pairs = _component_stats(
        buckets.get("easy", []), margins.get("easy", []), pairs.get("easy", 0), logits
    )
    del easy_loss
    return {
        "rank_loss_same_pocket": same_value,
        "rank_loss_high_pocket": high_value,
        "rank_loss_nearby": nearby_value,
        "rank_loss_easy": easy_value,
        "n_same_pocket_pairs": float(same_pairs),
        "n_high_pocket_pairs": float(high_pairs),
        "n_nearby_pairs": float(nearby_pairs),
        "n_easy_pairs": float(easy_pairs),
        "margin_same_pocket_mean": same_margin,
        "margin_high_pocket_mean": high_margin,
        "margin_nearby_mean": nearby_margin,
        "margin_easy_mean": easy_margin,
        "rank_same_pocket_negatives": float(same_pairs),
        "rank_high_pocket_negatives": float(high_pairs),
        "rank_easy_negatives": float(easy_pairs),
        "rank_same_margin": same_margin,
        "rank_high_margin": high_margin,
        "rank_easy_margin": easy_margin,
    }


def out_of_pocket_rescue_ranking_loss(
    logits: Tensor,
    labels: Tensor,
    batch_index: Tensor,
    in_any_pocket: Tensor,
    dssp_or_rsa_feat: Tensor | None = None,
    coords_ca: Tensor | None = None,
    neg_per_pos: int = 8,
    radius: float = 14.0,
) -> tuple[Tensor, dict[str, float]]:
    score = logits.float().view(-1)
    y = labels.float().view(-1).to(score.device)
    batch = batch_index.long().view(-1).to(score.device)
    in_pocket = _as_flat_bool(in_any_pocket, y)
    coords = coords_ca.float().to(score.device) if coords_ca is not None and coords_ca.numel() > 0 else None
    feat = dssp_or_rsa_feat.float().to(score.device) if dssp_or_rsa_feat is not None and dssp_or_rsa_feat.numel() > 0 else None

    losses: list[Tensor] = []
    margins: list[Tensor] = []
    pair_count = 0
    out_pos_total = int(((y > 0.5) & ~in_pocket).sum().item()) if y.numel() else 0
    if score.numel() == 0 or y.numel() == 0:
        return score.new_zeros(()), _rescue_stats(score, losses, margins, pair_count, out_pos_total)

    for protein_id in torch.unique(batch).tolist():
        protein_idx = torch.where(batch == int(protein_id))[0]
        out_pos = protein_idx[(y[protein_idx] > 0.5) & ~in_pocket[protein_idx]]
        out_neg = protein_idx[(y[protein_idx] <= 0.5) & ~in_pocket[protein_idx]]
        if out_pos.numel() == 0 or out_neg.numel() == 0:
            continue
        high_neg = _top_by_logit(out_neg, score, int(neg_per_pos))
        for pos in out_pos.tolist():
            candidates = [high_neg]
            local = _top_by_logit(_nearby_negatives(int(pos), out_neg, coords, radius), score, int(neg_per_pos))
            candidates.append(local)
            if feat is not None and int(pos) < int(feat.shape[0]):
                row = feat[int(pos)].view(-1)
                same_feat = torch.zeros_like(out_neg, dtype=torch.bool)
                if row.numel() > 0:
                    comp_cols = min(int(row.numel()), int(feat.shape[-1]), 2)
                    diff = (feat[out_neg.long(), :comp_cols] - row[:comp_cols].view(1, -1)).abs()
                    same_feat = (diff <= 1e-4).all(dim=-1)
                candidates.append(_top_by_logit(out_neg[same_feat], score, int(neg_per_pos)))
            candidates.append(_top_by_logit(out_neg, score, int(neg_per_pos)))
            chosen = torch.cat([row.long() for row in candidates if row.numel() > 0], dim=0).unique(sorted=True)
            chosen = _top_by_logit(chosen, score, int(neg_per_pos))
            if chosen.numel() == 0:
                continue
            margin = score[int(pos)] - score[chosen.long()]
            losses.append(F.softplus(-margin).mean())
            margins.append(margin.detach().mean())
            pair_count += int(chosen.numel())
    return (torch.stack(losses).mean() if losses else score.new_zeros(())), _rescue_stats(
        score,
        losses,
        margins,
        pair_count,
        out_pos_total,
    )


def _rescue_stats(logits: Tensor, losses: list[Tensor], margins: list[Tensor], pair_count: int, pos_count: int) -> dict[str, float]:
    value = float(torch.stack(losses).mean().detach().cpu()) if losses else 0.0
    margin = float(torch.stack(margins).mean().detach().cpu()) if margins else 0.0
    return {
        "rescue_rank_loss": value,
        "n_out_pocket_pos": float(pos_count),
        "n_rescue_pairs": float(pair_count),
        "rescue_margin_mean": margin,
    }
