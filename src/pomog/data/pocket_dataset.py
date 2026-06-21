from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.utils.data as data
from torch_geometric.data import Data

from pomog.data.residue_dataset import PoMoGDataset
from pomog.data.site_segments import cache_digest, cache_file, dssp8_index, rsa3_bin
from pomog.data.structure_prior_cache import (
    STRUCTURE_PRIOR_EDGE_ATTR_NAMES,
    STRUCTURE_PRIOR_FEATURE_DIM,
    detector_missing_tensor,
    empty_structure_prior_payload,
    load_structure_prior_for_protein,
)


POMOG_POCKET_RESIDUE_FEAT_DIM = 14
POMOG_POCKET_RESIDUE_FEAT_DIM_V2 = 20
POMOG_POCKET_NODE_RAW_DIM = 16
POMOG_POCKET_GRAPH_EDGE_DIM = 8
POMOG_POCKET_CACHE_VERSION = 1

_MICRO_NODE = 0
_P2RANK_NODE = 1
_EDGE_SEQ = 0
_EDGE_CENTROID = 1
_EDGE_CONTACT = 2
_EDGE_OVERLAP = 3
_EDGE_POCKET_RADIUS = 4
_EDGE_POCKET_KNN = 5


class PoMoGPocketData(Data):
    """PyG Data for PoMoG pocket-guided micro-motif branches."""

    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = int(self.seq.shape[0]) if hasattr(self, "seq") else 0
        num_micro = int(self.micro_motif_type.shape[0]) if hasattr(self, "micro_motif_type") else 0
        num_pockets = int(self.pocket_node_feature.shape[0]) if hasattr(self, "pocket_node_feature") else 0
        num_context = int(self.context_node_type.shape[0]) if hasattr(self, "context_node_type") else 0
        if key == "micro_motif_assignment":
            return num_micro
        if key == "micro_motif_member_index":
            return torch.tensor([[num_nodes], [num_micro]])
        if key == "pocket_member_index":
            return torch.tensor([[num_nodes], [num_pockets]])
        if key in {"context_edge_index", "context_node_member_index"}:
            return torch.tensor([[num_nodes], [num_context]])
        if key == "context_graph_edge_index":
            return num_context
        if key == "structure_prior_residue_to_pocket_index":
            num_prior_pockets = int(self.structure_prior_pocket_feature.shape[0]) if hasattr(self, "structure_prior_pocket_feature") else 0
            return torch.tensor([[num_nodes], [num_prior_pockets]])
        if key in {"micro_motif_batch", "pocket_batch", "context_node_batch"}:
            return 1
        return super().__inc__(key, value, *args, **kwargs)


def pomog_pocket_cache_fingerprint(
    *,
    radius: float,
    p2rank_cache_dir: str | None,
    fpocket_cache_dir: str | None,
    pocket_top_k: int,
    pocket_link_radius: float,
    pocket_context_top_k: int,
    partition_method: str,
    rsa_low: float,
    rsa_high: float,
    micro_knn: int,
    micro_contact_radius: float,
    micro_merge_ratio_threshold: float,
    merge_short_motifs: bool,
    build_micro_motifs: bool,
    build_pocket_nodes: bool,
    pocket_to_pocket_knn: int,
    residue_pocket_feature_dim: int = POMOG_POCKET_RESIDUE_FEAT_DIM,
) -> str:
    payload = {
        "cache": "pomog_pocket_guided",
        "version": POMOG_POCKET_CACHE_VERSION,
        "radius": round(float(radius), 6),
        "p2rank_cache_dir": str(p2rank_cache_dir or ""),
        "fpocket_cache_dir": str(fpocket_cache_dir or ""),
        "pocket_top_k": int(pocket_top_k),
        "pocket_link_radius": round(float(pocket_link_radius), 6),
        "pocket_context_top_k": int(pocket_context_top_k),
        "partition_method": str(partition_method).lower(),
        "rsa_low": round(float(rsa_low), 6),
        "rsa_high": round(float(rsa_high), 6),
        "micro_knn": int(micro_knn),
        "micro_contact_radius": round(float(micro_contact_radius), 6),
        "micro_merge_ratio_threshold": round(float(micro_merge_ratio_threshold), 6),
        "merge_short_motifs": bool(merge_short_motifs),
        "build_micro_motifs": bool(build_micro_motifs),
        "build_pocket_nodes": bool(build_pocket_nodes),
        "pocket_to_pocket_knn": int(pocket_to_pocket_knn),
        "schema": "dssp8_rsa3_micro_p2rank_sparse_graph_v1",
    }
    if int(residue_pocket_feature_dim) > POMOG_POCKET_RESIDUE_FEAT_DIM:
        payload["residue_pocket_feature_dim"] = int(residue_pocket_feature_dim)
        payload["schema"] = "dssp8_rsa3_micro_p2rank_fpocket_consensus_sparse_graph_v2"
    return cache_digest(payload)


def _default_local_cache_root() -> Path:
    root = os.environ.get("POMOG_LOCAL_CACHE_DIR", "").strip()
    if root:
        return Path(root).expanduser().resolve()
    user = os.environ.get("USER") or "unknown"
    return Path.cwd().resolve() / "tmp" / "pomog_cache" / user


def resolve_pomog_pocket_cache_dir(root_dir: str, *, fingerprint: str, explicit_dir: str | None = None) -> Path:
    if explicit_dir:
        return Path(explicit_dir).expanduser().resolve()
    root_key = cache_digest({"feature_root": str(Path(root_dir).expanduser().resolve())})[:12]
    return _default_local_cache_root() / "pomog_pocket_guided" / root_key / fingerprint


def _pocket_cache_candidates(cache_dir: str | None, source: str, name: str) -> list[Path]:
    if not cache_dir:
        return []
    root = Path(cache_dir).expanduser().resolve()
    source = str(source).lower()
    stems = [name, name.upper(), name.lower()]
    return [root / source / f"{stem}.pt" for stem in stems] + [root / f"{stem}.pt" for stem in stems]


def _load_pocket_payload(cache_dir: str | None, source: str, name: str, length: int) -> dict[str, Any] | None:
    for path in _pocket_cache_candidates(cache_dir, source, name):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if int(payload.get("length", length)) != int(length):
            continue
        return payload
    return None


def _as_float_tensor(value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except Exception:
        tensor = torch.zeros(shape, dtype=torch.float32)
    if tuple(tensor.shape) != tuple(shape):
        tensor = tensor.reshape(-1)[: int(torch.tensor(shape).prod().item())]
        if tensor.numel() < int(torch.tensor(shape).prod().item()):
            pad = torch.zeros((int(torch.tensor(shape).prod().item()) - tensor.numel(),), dtype=torch.float32)
            tensor = torch.cat([tensor.float(), pad], dim=0)
        tensor = tensor.view(*shape)
    return torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _coerce_pockets(payload: dict[str, Any] | None, *, length: int, ca_pos: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    if not payload:
        return []
    pockets = payload.get("pockets", [])
    if not isinstance(pockets, list):
        return []
    out: list[dict[str, Any]] = []
    for row_id, item in enumerate(pockets, start=1):
        if not isinstance(item, dict):
            continue
        idx = item.get("residue_index")
        if idx is None:
            idx = item.get("residue_indices")
        if idx is None:
            idx = item.get("residues")
        idx = torch.as_tensor(idx if idx is not None else [], dtype=torch.long).view(-1)
        idx = idx[(idx >= 0) & (idx < int(length))].unique(sorted=True)
        if idx.numel() == 0:
            continue
        rank = int(float(item.get("rank", row_id) or row_id))
        score = float(item.get("score", item.get("probability", item.get("prob", 0.0))) or 0.0)
        center_value = item.get("center", None)
        if center_value is None:
            center = ca_pos[idx].mean(dim=0) if idx.numel() else ca_pos.new_zeros((3,))
        else:
            center = _as_float_tensor(center_value, (3,))
            if not bool(torch.isfinite(center).all()) or float(center.abs().sum().item()) <= 1e-8:
                center = ca_pos[idx].mean(dim=0)
        out.append(
            {
                "rank": rank,
                "score": score,
                "center": center.float().view(3),
                "residue_index": idx.long(),
                "size": int(idx.numel()),
            }
        )
    out.sort(key=lambda pocket: (int(pocket["rank"]), -float(pocket["score"])))
    if int(top_k) > 0:
        out = out[: int(top_k)]
    return out


def _rsa_values(node_feat: torch.Tensor) -> torch.Tensor:
    length = int(node_feat.shape[0])
    if length <= 0 or node_feat.shape[-1] <= 1024:
        return torch.zeros((length,), dtype=torch.float32)
    return torch.nan_to_num(node_feat[:, 1024].float().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _boundary_distance(dssp: torch.Tensor, *, cap: int = 8) -> torch.Tensor:
    length = int(dssp.numel())
    if length <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    if length == 1:
        return torch.ones((1,), dtype=torch.float32)
    changes = torch.where(dssp[1:] != dssp[:-1])[0] + 1
    if changes.numel() == 0:
        return torch.ones((length,), dtype=torch.float32)
    pos = torch.arange(length, dtype=torch.float32).view(-1, 1)
    dist = torch.abs(pos - changes.float().view(1, -1)).min(dim=1).values
    return (dist / float(max(1, cap))).clamp(0.0, 1.0).float()


def _residue_pocket_features(
    *,
    length: int,
    ca_pos: torch.Tensor,
    p2rank_pockets: list[dict[str, Any]],
    fpocket_pockets: list[dict[str, Any]],
    dssp: torch.Tensor,
    rsa: torch.Tensor,
    rsa_bin: torch.Tensor,
    boundary_dist: torch.Tensor,
    feature_dim: int = POMOG_POCKET_RESIDUE_FEAT_DIM,
) -> torch.Tensor:
    feature_dim = int(feature_dim)
    feat = torch.zeros((length, max(POMOG_POCKET_RESIDUE_FEAT_DIM, feature_dim)), dtype=torch.float32)
    if length <= 0:
        return feat
    if feature_dim >= POMOG_POCKET_RESIDUE_FEAT_DIM_V2:
        p2_nearest = torch.full((length,), 30.0, dtype=torch.float32)
        fp_nearest = torch.full((length,), 30.0, dtype=torch.float32)
        p2_best_rank = torch.zeros((length,), dtype=torch.float32)
        fp_best_rank = torch.zeros((length,), dtype=torch.float32)
        p2_best_score = torch.zeros((length,), dtype=torch.float32)
        fp_best_score = torch.zeros((length,), dtype=torch.float32)
        p2_cover = torch.zeros((length,), dtype=torch.float32)
        fp_cover = torch.zeros((length,), dtype=torch.float32)

        if p2rank_pockets:
            centers = torch.stack([pocket["center"].float() for pocket in p2rank_pockets], dim=0)
            p2_nearest = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
            for pocket in p2rank_pockets:
                rank = max(1, int(pocket["rank"]))
                idx = pocket["residue_index"].long()
                if rank == 1:
                    feat[idx, 0] = 1.0
                if rank <= 3:
                    feat[idx, 1] = 1.0
                if rank <= 5:
                    feat[idx, 2] = 1.0
                    p2_cover[idx] = 1.0
                score = float(pocket["score"])
                update = (p2_best_rank[idx] <= 0.0) | ((1.0 / float(rank)) > p2_best_rank[idx])
                target = idx[update]
                p2_best_rank[target] = 1.0 / float(rank)
                p2_best_score[target] = score

        if fpocket_pockets:
            centers = torch.stack([pocket["center"].float() for pocket in fpocket_pockets], dim=0)
            fp_nearest = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
            for pocket in fpocket_pockets:
                rank = max(1, int(pocket["rank"]))
                idx = pocket["residue_index"].long()
                if rank == 1:
                    feat[idx, 3] = 1.0
                if rank <= 3:
                    feat[idx, 4] = 1.0
                if rank <= 5:
                    feat[idx, 5] = 1.0
                    fp_cover[idx] = 1.0
                score = float(pocket["score"])
                update = (fp_best_rank[idx] <= 0.0) | ((1.0 / float(rank)) > fp_best_rank[idx])
                target = idx[update]
                fp_best_rank[target] = 1.0 / float(rank)
                fp_best_score[target] = score

        feat[:, 6] = (p2_nearest / 30.0).clamp(0.0, 1.0)
        feat[:, 7] = (fp_nearest / 30.0).clamp(0.0, 1.0)
        feat[:, 8] = p2_best_score
        feat[:, 9] = fp_best_score
        feat[:, 10] = p2_best_rank
        feat[:, 11] = fp_best_rank
        feat[:, 12] = ((p2_cover > 0.5) & (fp_cover > 0.5)).float()
        feat[:, 13] = (p2_cover + fp_cover).clamp(0.0, 2.0) / 2.0
        feat[:, 14] = dssp.float() / 7.0
        feat[:, 15] = rsa.float().clamp(0.0, 1.0)
        feat[:, 16] = (rsa_bin == 0).float()
        feat[:, 17] = (rsa_bin == 1).float()
        feat[:, 18] = (rsa_bin == 2).float()
        feat[:, 19] = boundary_dist.float().clamp(0.0, 1.0)
        return torch.nan_to_num(feat[:, :feature_dim], nan=0.0, posinf=1.0, neginf=0.0)

    nearest_center = torch.full((length,), 30.0, dtype=torch.float32)
    best_rank = torch.zeros((length,), dtype=torch.float32)
    best_score = torch.zeros((length,), dtype=torch.float32)
    if p2rank_pockets:
        centers = torch.stack([pocket["center"].float() for pocket in p2rank_pockets], dim=0)
        center_dist = torch.cdist(ca_pos.float(), centers.float(), p=2.0)
        nearest_center = center_dist.min(dim=1).values.clamp(max=30.0)
        for pocket in p2rank_pockets:
            rank = max(1, int(pocket["rank"]))
            idx = pocket["residue_index"].long()
            if rank == 1:
                feat[idx, 0] = 1.0
            if rank <= 3:
                feat[idx, 1] = 1.0
            if rank <= 5:
                feat[idx, 2] = 1.0
            score = float(pocket["score"])
            update = (best_rank[idx] <= 0.0) | (float(rank) < (1.0 / best_rank[idx].clamp_min(1e-6)))
            target = idx[update]
            best_rank[target] = 1.0 / float(rank)
            best_score[target] = score
    for pocket in fpocket_pockets:
        rank = max(1, int(pocket["rank"]))
        idx = pocket["residue_index"].long()
        if rank <= 3:
            feat[idx, 6] = 1.0
        if rank <= 5:
            feat[idx, 7] = 1.0
    feat[:, 3] = best_rank
    feat[:, 4] = best_score
    feat[:, 5] = (nearest_center / 30.0).clamp(0.0, 1.0)
    feat[:, 8] = dssp.float() / 7.0
    feat[:, 9] = rsa.float().clamp(0.0, 1.0)
    feat[:, 10] = (rsa_bin == 0).float()
    feat[:, 11] = (rsa_bin == 1).float()
    feat[:, 12] = (rsa_bin == 2).float()
    feat[:, 13] = boundary_dist.float().clamp(0.0, 1.0)
    return torch.nan_to_num(feat[:, :feature_dim], nan=0.0, posinf=1.0, neginf=0.0)


def _dense_membership_from_pockets(
    pockets: Sequence[dict[str, Any]],
    *,
    length: int,
    top_k: int,
) -> torch.Tensor:
    width = max(0, int(top_k))
    membership = torch.zeros((int(length), width), dtype=torch.bool)
    for col, pocket in enumerate(list(pockets)[:width]):
        idx = pocket["residue_index"].long()
        idx = idx[(idx >= 0) & (idx < int(length))]
        if idx.numel() > 0:
            membership[idx, col] = True
    return membership


def _pocket_feat_from_residue_feature(residue_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(residue_feat.shape[0])
    feat = torch.zeros((length, 14), dtype=torch.float32)
    if residue_feat.numel() == 0:
        empty = torch.zeros((length,), dtype=torch.bool)
        return feat, empty, empty, empty
    src = torch.nan_to_num(residue_feat.float().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    if src.shape[-1] >= POMOG_POCKET_RESIDUE_FEAT_DIM_V2:
        feat = src[:, :14].clone()
        in_p2 = feat[:, 2] > 0.5
        in_fp = feat[:, 5] > 0.5
        if not bool((in_p2 | in_fp).any()) and float(feat[:, [8, 9, 10, 11, 12, 13]].abs().sum().item()) <= 1e-8:
            feat.zero_()
        return feat, (in_p2 | in_fp), in_p2, in_fp

    take = min(3, int(src.shape[-1]))
    if take > 0:
        feat[:, :take] = src[:, :take]
    if src.shape[-1] >= 8:
        feat[:, 4] = src[:, 6]
        feat[:, 5] = src[:, 7]
    if src.shape[-1] >= 6:
        feat[:, 6] = src[:, 5]
    if src.shape[-1] >= 5:
        feat[:, 8] = src[:, 4]
    if src.shape[-1] >= 4:
        feat[:, 10] = src[:, 3]
    p2_cover = feat[:, 2] > 0.5
    fp_cover = feat[:, 5] > 0.5
    feat[:, 12] = (p2_cover & fp_cover).float()
    feat[:, 13] = (p2_cover.float() + fp_cover.float()).clamp(0.0, 2.0) / 2.0
    if not bool((p2_cover | fp_cover).any()) and float(feat[:, [8, 10, 12, 13]].abs().sum().item()) <= 1e-8:
        feat.zero_()
    return feat, (p2_cover | fp_cover), p2_cover, fp_cover


def _contiguous_micro_motifs(keys: torch.Tensor, type_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    length = int(keys.numel())
    if length <= 0:
        return torch.zeros((0,), dtype=torch.long), [], torch.zeros((0,), dtype=torch.long)
    assignment = torch.zeros((length,), dtype=torch.long)
    motifs: list[torch.Tensor] = []
    types: list[int] = []
    start = 0
    motif_id = 0
    for idx in range(1, length + 1):
        if idx < length and int(keys[idx]) == int(keys[start]):
            continue
        residues = torch.arange(start, idx, dtype=torch.long)
        assignment[residues] = motif_id
        motifs.append(residues)
        types.append(int(type_values[start]))
        motif_id += 1
        start = idx
    return assignment.long(), motifs, torch.tensor(types, dtype=torch.long)


def _merge_short_segments(
    assignment: torch.Tensor,
    motifs: list[torch.Tensor],
    motif_type: torch.Tensor,
    *,
    min_length: int = 2,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    if len(motifs) <= 1:
        return assignment, motifs, motif_type
    groups: list[list[torch.Tensor]] = []
    group_types: list[int] = []
    for idx, residues in enumerate(motifs):
        if int(residues.numel()) < int(min_length) and groups:
            groups[-1].append(residues)
            continue
        groups.append([residues])
        group_types.append(int(motif_type[idx]))
    if len(groups) > 1 and sum(int(chunk.numel()) for chunk in groups[-1]) < int(min_length):
        tail = groups.pop()
        group_types.pop()
        groups[-1].extend(tail)
    new_assignment = assignment.clone()
    new_motifs: list[torch.Tensor] = []
    for new_id, chunks in enumerate(groups):
        residues = torch.cat(chunks, dim=0).unique(sorted=True)
        new_assignment[residues] = new_id
        new_motifs.append(residues)
    return new_assignment.long(), new_motifs, torch.tensor(group_types, dtype=torch.long)


def _member_index_from_motifs(motifs: Sequence[torch.Tensor]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for motif_id, residues in enumerate(motifs):
        residues = residues.long().view(-1)
        if residues.numel() == 0:
            continue
        rows.append(residues)
        cols.append(torch.full_like(residues, int(motif_id)))
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0).long()


def _micro_raw_features(
    motifs: Sequence[torch.Tensor],
    motif_type: torch.Tensor,
    *,
    length: int,
    dssp: torch.Tensor,
    rsa: torch.Tensor,
    rsa_bin: torch.Tensor,
    boundary_dist: torch.Tensor,
    ca_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(motifs)
    raw = torch.zeros((count, POMOG_POCKET_NODE_RAW_DIM), dtype=torch.float32)
    centroid = torch.zeros((count, 3), dtype=torch.float32)
    motif_length = torch.zeros((count, 1), dtype=torch.float32)
    for motif_id, residues in enumerate(motifs):
        idx = residues.long()
        if idx.numel() == 0:
            continue
        dssp_id = int(dssp[idx[0]].item()) if dssp.numel() else 7
        rsa_id = int(rsa_bin[idx[0]].item()) if rsa_bin.numel() else 1
        raw[motif_id, 0] = float(dssp_id) / 7.0
        raw[motif_id, 1] = float(rsa_id) / 2.0
        raw[motif_id, 2] = float(idx.numel()) / float(max(length, 1))
        raw[motif_id, 3] = rsa[idx].mean() if rsa.numel() else 0.0
        raw[motif_id, 4] = boundary_dist[idx].min() if boundary_dist.numel() else 1.0
        raw[motif_id, 5] = float(motif_type[motif_id].item()) / 23.0
        raw[motif_id, 6:14] = torch.nn.functional.one_hot(torch.tensor(dssp_id).clamp(0, 7), num_classes=8).float()
        raw[motif_id, 14] = 1.0
        raw[motif_id, 15] = 0.0
        centroid[motif_id] = ca_pos[idx].mean(dim=0)
        motif_length[motif_id, 0] = raw[motif_id, 2]
    return raw, centroid, motif_length


def _pocket_node_tensors(
    pockets: Sequence[dict[str, Any]],
    *,
    length: int,
    ca_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.zeros((len(pockets), POMOG_POCKET_NODE_RAW_DIM), dtype=torch.float32)
    centers = torch.zeros((len(pockets), 3), dtype=torch.float32)
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for pocket_id, pocket in enumerate(pockets):
        idx = pocket["residue_index"].long()
        rank = max(1, int(pocket["rank"]))
        score = float(pocket["score"])
        size = int(idx.numel())
        center = pocket["center"].float().view(3)
        features[pocket_id, 0] = 1.0 / float(rank)
        features[pocket_id, 1] = score
        features[pocket_id, 2] = float(size) / float(max(length, 1))
        features[pocket_id, 3:6] = center / 100.0
        features[pocket_id, 6] = 0.0
        features[pocket_id, 7] = 1.0
        features[pocket_id, 14] = 0.0
        features[pocket_id, 15] = 1.0
        centers[pocket_id] = center
        if idx.numel() > 0:
            rows.append(idx)
            cols.append(torch.full_like(idx, int(pocket_id)))
    member = (
        torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0).long()
        if rows
        else torch.zeros((2, 0), dtype=torch.long)
    )
    ranks = torch.tensor([max(1, int(pocket["rank"])) for pocket in pockets], dtype=torch.long)
    scores = torch.tensor([float(pocket["score"]) for pocket in pockets], dtype=torch.float32)
    return features, centers, member, torch.stack([ranks.float(), scores], dim=-1) if len(pockets) else torch.zeros((0, 2))


def _edge_attr(rel_type: int, distance: float, *, overlap: bool = False) -> torch.Tensor:
    rel_count = 6
    out = torch.zeros((POMOG_POCKET_GRAPH_EDGE_DIM,), dtype=torch.float32)
    out[0] = float(rel_type) / float(max(rel_count - 1, 1))
    out[1] = min(max(float(distance), 0.0) / 30.0, 2.0)
    if 0 <= rel_type < rel_count:
        out[2 + rel_type] = 1.0
    if overlap:
        out[-1] = 1.0
    return out


def _build_sparse_context_graph(
    *,
    motifs: Sequence[torch.Tensor],
    micro_assignment: torch.Tensor,
    micro_centroid: torch.Tensor,
    pocket_centers: torch.Tensor,
    pocket_member_index: torch.Tensor,
    ca_pos: torch.Tensor,
    micro_knn: int,
    micro_contact_radius: float,
    pocket_link_radius: float,
    pocket_to_pocket_knn: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    micro_count = len(motifs)
    pocket_count = int(pocket_centers.shape[0])
    rows: dict[tuple[int, int], torch.Tensor] = {}

    def add(src: int, dst: int, rel_type: int, distance: float, *, overlap: bool = False) -> None:
        if src == dst:
            return
        key = (int(src), int(dst))
        attr = _edge_attr(rel_type, distance, overlap=overlap)
        old = rows.get(key)
        rows[key] = attr if old is None else torch.maximum(old, attr)

    if micro_count > 1:
        centroid_dist = torch.cdist(micro_centroid.float(), micro_centroid.float(), p=2.0)
        for src in range(micro_count - 1):
            dist = float(centroid_dist[src, src + 1].item())
            add(src, src + 1, _EDGE_SEQ, dist)
            add(src + 1, src, _EDGE_SEQ, dist)
        k = min(max(0, int(micro_knn)), micro_count - 1)
        if k > 0:
            masked = centroid_dist.clone()
            masked.fill_diagonal_(float("inf"))
            nearest = torch.topk(masked, k=k, largest=False, dim=-1).indices
            for src in range(micro_count):
                for dst in nearest[src].tolist():
                    add(src, int(dst), _EDGE_CENTROID, float(centroid_dist[src, int(dst)].item()))
        if float(micro_contact_radius) > 0.0 and int(ca_pos.shape[0]) > 0:
            ca_dist = torch.cdist(ca_pos.float(), ca_pos.float(), p=2.0)
            src_res, dst_res = torch.where((ca_dist <= float(micro_contact_radius)) & (ca_dist > 0.0))
            if src_res.numel() > 0:
                pairs = torch.stack([micro_assignment[src_res], micro_assignment[dst_res]], dim=1)
                pairs = pairs[pairs[:, 0] != pairs[:, 1]].unique(dim=0)
                for src, dst in pairs.tolist():
                    dist = float(torch.linalg.norm(micro_centroid[int(dst)] - micro_centroid[int(src)]).item())
                    add(int(src), int(dst), _EDGE_CONTACT, dist)

    if micro_count > 0 and pocket_count > 0:
        pocket_offset = micro_count
        micro_pocket_dist = torch.cdist(micro_centroid.float(), pocket_centers.float(), p=2.0)
        near_micro, near_pocket = torch.where(micro_pocket_dist <= float(pocket_link_radius))
        for src, pocket_id in zip(near_micro.tolist(), near_pocket.tolist()):
            dst = pocket_offset + int(pocket_id)
            dist = float(micro_pocket_dist[int(src), int(pocket_id)].item())
            add(int(src), dst, _EDGE_POCKET_RADIUS, dist)
            add(dst, int(src), _EDGE_POCKET_RADIUS, dist)
        if pocket_member_index.numel() > 0:
            residues, pocket_ids = pocket_member_index.long()
            micro_ids = micro_assignment[residues].long()
            pairs = torch.stack([micro_ids, pocket_ids], dim=1).unique(dim=0)
            for micro_id, pocket_id in pairs.tolist():
                dst = pocket_offset + int(pocket_id)
                dist = float(micro_pocket_dist[int(micro_id), int(pocket_id)].item())
                add(int(micro_id), dst, _EDGE_OVERLAP, dist, overlap=True)
                add(dst, int(micro_id), _EDGE_OVERLAP, dist, overlap=True)

    if pocket_count > 1 and int(pocket_to_pocket_knn) > 0:
        pocket_offset = micro_count
        pocket_dist = torch.cdist(pocket_centers.float(), pocket_centers.float(), p=2.0)
        k = min(int(pocket_to_pocket_knn), pocket_count - 1)
        masked = pocket_dist.clone()
        masked.fill_diagonal_(float("inf"))
        nearest = torch.topk(masked, k=k, largest=False, dim=-1).indices
        for src in range(pocket_count):
            for dst in nearest[src].tolist():
                add(pocket_offset + src, pocket_offset + int(dst), _EDGE_POCKET_KNN, float(pocket_dist[src, int(dst)].item()))

    if not rows:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, POMOG_POCKET_GRAPH_EDGE_DIM), dtype=torch.float32)
    items = sorted(rows.items())
    edge_index = torch.tensor([item[0] for item in items], dtype=torch.long).t().contiguous()
    edge_attr = torch.stack([item[1] for item in items], dim=0).float()
    return edge_index, edge_attr


def _build_context_edges(
    *,
    length: int,
    micro_assignment: torch.Tensor,
    micro_count: int,
    pocket_centers: torch.Tensor,
    pocket_member_index: torch.Tensor,
    ca_pos: torch.Tensor,
    pocket_context_top_k: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    if length > 0 and micro_count > 0:
        residues = torch.arange(length, dtype=torch.long)
        rows.append(residues)
        cols.append(micro_assignment.long().clamp(min=0, max=max(micro_count - 1, 0)))
    pocket_count = int(pocket_centers.shape[0])
    pocket_offset = int(micro_count)
    if pocket_count > 0 and int(pocket_context_top_k) > 0:
        k = min(int(pocket_context_top_k), pocket_count)
        dist = torch.cdist(ca_pos.float(), pocket_centers.float(), p=2.0)
        nearest = torch.topk(dist, k=k, largest=False, dim=-1).indices
        residues = torch.arange(length, dtype=torch.long).repeat_interleave(k)
        rows.append(residues)
        cols.append((nearest.reshape(-1) + pocket_offset).long())
    if pocket_member_index.numel() > 0:
        residues, pocket_ids = pocket_member_index.long()
        rows.append(residues)
        cols.append(pocket_ids + pocket_offset)
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0).long()
    return torch.unique(edge_index.t(), dim=0).t().contiguous()


def _labels_from_member(labels: torch.Tensor, member_index: torch.Tensor, count: int) -> torch.Tensor:
    out = torch.zeros((count, 1), dtype=torch.float32)
    if count <= 0 or member_index.numel() == 0:
        return out
    labels = labels.float().view(-1).cpu()
    residues, groups = member_index.long().cpu()
    for group_id in range(count):
        idx = residues[groups == group_id]
        if idx.numel() > 0:
            out[group_id, 0] = labels[idx].max()
    return out


def _load_cache(path: Path, fingerprint: str, length: int) -> dict[str, torch.Tensor] | None:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return None
    required = (
        "residue_pocket_feature",
        "residue_dssp8_id",
        "residue_rsa_bin",
        "residue_boundary_distance",
        "micro_motif_assignment",
        "micro_motif_type",
        "micro_motif_length",
        "micro_motif_centroid",
        "micro_motif_raw_feature",
        "micro_motif_member_index",
        "pocket_node_feature",
        "pocket_center",
        "pocket_rank_score",
        "pocket_member_index",
        "context_node_member_index",
        "context_node_type",
        "context_node_raw_feature",
        "context_graph_edge_index",
        "context_graph_edge_attr",
        "context_edge_index",
    )
    if not isinstance(payload, dict):
        return None
    if payload.get("fingerprint") != fingerprint or int(payload.get("length", -1)) != int(length):
        return None
    if any(key not in payload for key in required):
        return None
    optional = (
        "pocket_feat",
        "in_any_pocket",
        "in_p2rank_topk",
        "in_fpocket_topk",
        "p2rank_membership",
        "fpocket_membership",
        "pocket_missing",
    )
    out = {key: payload[key] for key in required}
    out.update({key: payload[key] for key in optional if key in payload})
    return out


def _ensure_pocket_schema(graph: PoMoGPocketData) -> None:
    """Backfill optional pocket tensors so mixed old/new cache entries can batch."""
    num_nodes = int(graph.seq.shape[0]) if hasattr(graph, "seq") else 0
    pocket_count = int(graph.pocket_node_feature.shape[0]) if hasattr(graph, "pocket_node_feature") else 0
    if not hasattr(graph, "pocket_rank_score"):
        graph.pocket_rank_score = torch.zeros((pocket_count, 1), dtype=torch.float32)
    else:
        score = graph.pocket_rank_score.float()
        if score.ndim == 1:
            score = score.view(-1, 1)
        if int(score.shape[0]) != pocket_count:
            fixed = torch.zeros((pocket_count, 1), dtype=torch.float32)
            take = min(pocket_count, int(score.shape[0]))
            if take > 0:
                fixed[:take, : min(1, int(score.shape[1]))] = score[:take, :1]
            score = fixed
        graph.pocket_rank_score = score
    residue_feat = getattr(graph, "residue_pocket_feature", torch.zeros((num_nodes, POMOG_POCKET_RESIDUE_FEAT_DIM), dtype=torch.float32))
    pocket_feat, in_any, in_p2rank, in_fpocket = _pocket_feat_from_residue_feature(residue_feat)
    if not hasattr(graph, "pocket_feat"):
        graph.pocket_feat = pocket_feat.float()
    if not hasattr(graph, "in_any_pocket"):
        graph.in_any_pocket = in_any.bool()
    if not hasattr(graph, "in_p2rank_topk"):
        graph.in_p2rank_topk = in_p2rank.bool()
    if not hasattr(graph, "in_fpocket_topk"):
        graph.in_fpocket_topk = in_fpocket.bool()
    if not hasattr(graph, "p2rank_membership"):
        membership_width = max(5, pocket_count)
        graph.p2rank_membership = torch.zeros((int(graph.seq.shape[0]) if hasattr(graph, "seq") else 0, membership_width), dtype=torch.bool)
        if hasattr(graph, "pocket_member_index") and graph.pocket_member_index.numel() > 0:
            rows, cols = graph.pocket_member_index.long()
            valid = (rows >= 0) & (rows < graph.p2rank_membership.shape[0]) & (cols >= 0) & (cols < graph.p2rank_membership.shape[1])
            graph.p2rank_membership[rows[valid], cols[valid]] = True
    if not hasattr(graph, "fpocket_membership"):
        graph.fpocket_membership = torch.zeros_like(graph.p2rank_membership)
    if not hasattr(graph, "pocket_missing"):
        graph.pocket_missing = torch.tensor([not bool(graph.in_any_pocket.any())], dtype=torch.bool)


def _save_cache(path: Path, payload: dict[str, torch.Tensor | str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


class PoMoGPocketDataset(data.Dataset):
    """Active-site dataset wrapper for pocket-guided PoMoG branches."""

    def __init__(
        self,
        dataset: Dict[str, list],
        index: Sequence,
        root_dir: str,
        radius: float = 10,
        mode: str = "train",
        p2rank_cache_dir: str | None = None,
        fpocket_cache_dir: str | None = None,
        pocket_top_k: int = 5,
        pocket_link_radius: float = 10.0,
        pocket_context_top_k: int = 3,
        partition_method: str = "dssp8_rsa3_micro",
        rsa_low: float = 0.10,
        rsa_high: float = 0.30,
        micro_knn: int = 4,
        micro_contact_radius: float = 8.0,
        micro_merge_ratio_threshold: float = 0.80,
        merge_short_motifs: bool = True,
        min_motif_length: int = 2,
        build_micro_motifs: bool = True,
        build_pocket_nodes: bool = True,
        pocket_to_pocket_knn: int = 0,
        residue_pocket_feature_dim: int = POMOG_POCKET_RESIDUE_FEAT_DIM,
        pomog_pocket_cache_dir: str | None = None,
        pomog_pocket_cache_enabled: bool = True,
        pomog_pocket_cache_required: bool = False,
        pomog_pocket_cache_write: bool = True,
        structure_prior_cache_enabled: bool = False,
        structure_prior_cache_root: str | None = None,
        structure_prior_cache_required: bool = False,
        slow_sample_log_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.base = PoMoGDataset(dataset=dataset, index=index, root_dir=root_dir, radius=radius, mode=mode)
        self.p2rank_cache_dir = p2rank_cache_dir
        self.fpocket_cache_dir = fpocket_cache_dir
        self.pocket_top_k = int(pocket_top_k)
        self.pocket_link_radius = float(pocket_link_radius)
        self.pocket_context_top_k = int(pocket_context_top_k)
        self.partition_method = str(partition_method).lower()
        self.rsa_low = float(rsa_low)
        self.rsa_high = float(rsa_high)
        self.micro_knn = int(micro_knn)
        self.micro_contact_radius = float(micro_contact_radius)
        self.micro_merge_ratio_threshold = float(micro_merge_ratio_threshold)
        self.merge_short_motifs = bool(merge_short_motifs)
        self.min_motif_length = int(min_motif_length)
        self.build_micro_motifs = bool(build_micro_motifs)
        self.build_pocket_nodes = bool(build_pocket_nodes)
        self.pocket_to_pocket_knn = int(pocket_to_pocket_knn)
        self.residue_pocket_feature_dim = max(POMOG_POCKET_RESIDUE_FEAT_DIM, int(residue_pocket_feature_dim))
        self.cache_enabled = bool(pomog_pocket_cache_enabled)
        self.cache_required = bool(pomog_pocket_cache_required)
        self.cache_write = bool(pomog_pocket_cache_write)
        self.structure_prior_cache_enabled = bool(structure_prior_cache_enabled)
        self.structure_prior_cache_root = structure_prior_cache_root
        self.structure_prior_cache_required = bool(structure_prior_cache_required)
        self.cache_fingerprint = pomog_pocket_cache_fingerprint(
            radius=radius,
            p2rank_cache_dir=p2rank_cache_dir,
            fpocket_cache_dir=fpocket_cache_dir,
            pocket_top_k=pocket_top_k,
            pocket_link_radius=pocket_link_radius,
            pocket_context_top_k=pocket_context_top_k,
            partition_method=partition_method,
            rsa_low=rsa_low,
            rsa_high=rsa_high,
            micro_knn=micro_knn,
            micro_contact_radius=micro_contact_radius,
            micro_merge_ratio_threshold=micro_merge_ratio_threshold,
            merge_short_motifs=merge_short_motifs,
            build_micro_motifs=build_micro_motifs,
            build_pocket_nodes=build_pocket_nodes,
            pocket_to_pocket_knn=pocket_to_pocket_knn,
            residue_pocket_feature_dim=self.residue_pocket_feature_dim,
        )
        self.cache_dir = (
            resolve_pomog_pocket_cache_dir(root_dir, fingerprint=self.cache_fingerprint, explicit_dir=pomog_pocket_cache_dir)
            if self.cache_enabled
            else None
        )
        env_threshold = os.environ.get("POMOG_SLOW_SAMPLE_LOG_SECONDS", "").strip()
        self.slow_sample_log_threshold = (
            float(slow_sample_log_threshold)
            if slow_sample_log_threshold is not None
            else float(env_threshold) if env_threshold else 2.0
        )

    def __len__(self) -> int:
        return len(self.base)

    def _build_payload(self, graph: Data, name: str, length: int) -> dict[str, torch.Tensor | str | int]:
        ca_pos = torch.nan_to_num(graph.X[:, 1].float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        dssp = dssp8_index(graph.node_feat)
        rsa = _rsa_values(graph.node_feat)
        rsa_bin = rsa3_bin(graph.node_feat, low=self.rsa_low, high=self.rsa_high)
        boundary_dist = _boundary_distance(dssp)

        p2rank_payload = _load_pocket_payload(self.p2rank_cache_dir, "p2rank", name, length)
        fpocket_payload = _load_pocket_payload(self.fpocket_cache_dir, "fpocket", name, length)
        p2rank_pockets_all = _coerce_pockets(p2rank_payload, length=length, ca_pos=ca_pos, top_k=max(5, self.pocket_top_k))
        fpocket_pockets = _coerce_pockets(fpocket_payload, length=length, ca_pos=ca_pos, top_k=5)
        p2rank_pockets = p2rank_pockets_all[: max(0, int(self.pocket_top_k))] if self.build_pocket_nodes else []
        residue_feat = _residue_pocket_features(
            length=length,
            ca_pos=ca_pos,
            p2rank_pockets=p2rank_pockets_all,
            fpocket_pockets=fpocket_pockets,
            dssp=dssp,
            rsa=rsa,
            rsa_bin=rsa_bin,
            boundary_dist=boundary_dist,
            feature_dim=self.residue_pocket_feature_dim,
        )
        pocket_feat, in_any_pocket, in_p2rank_topk, in_fpocket_topk = _pocket_feat_from_residue_feature(residue_feat)
        p2rank_membership = _dense_membership_from_pockets(
            p2rank_pockets_all,
            length=length,
            top_k=max(0, int(self.pocket_top_k)),
        )
        fpocket_membership = _dense_membership_from_pockets(
            fpocket_pockets,
            length=length,
            top_k=max(0, int(self.pocket_top_k)),
        )
        pocket_missing = torch.tensor([not bool(p2rank_pockets_all or fpocket_pockets)], dtype=torch.bool)

        if self.build_micro_motifs:
            keys = dssp * 3 + rsa_bin if "rsa3" in self.partition_method else dssp
            type_values = keys.clone()
            assignment, motifs, motif_type = _contiguous_micro_motifs(keys, type_values)
            if self.merge_short_motifs or (length > 0 and len(motifs) > float(length) * self.micro_merge_ratio_threshold):
                assignment, motifs, motif_type = _merge_short_segments(
                    assignment,
                    motifs,
                    motif_type,
                    min_length=max(2, self.min_motif_length),
                )
            micro_raw, micro_centroid, micro_length = _micro_raw_features(
                motifs,
                motif_type,
                length=length,
                dssp=dssp,
                rsa=rsa,
                rsa_bin=rsa_bin,
                boundary_dist=boundary_dist,
                ca_pos=ca_pos,
            )
            micro_member = _member_index_from_motifs(motifs)
        else:
            assignment = torch.zeros((length,), dtype=torch.long)
            motifs = []
            motif_type = torch.zeros((0,), dtype=torch.long)
            micro_raw = torch.zeros((0, POMOG_POCKET_NODE_RAW_DIM), dtype=torch.float32)
            micro_centroid = torch.zeros((0, 3), dtype=torch.float32)
            micro_length = torch.zeros((0, 1), dtype=torch.float32)
            micro_member = torch.zeros((2, 0), dtype=torch.long)

        pocket_feature, pocket_centers, pocket_member, pocket_rank_score = _pocket_node_tensors(
            p2rank_pockets,
            length=length,
            ca_pos=ca_pos,
        )
        micro_count = int(micro_raw.shape[0])
        pocket_count = int(pocket_feature.shape[0])
        context_node_type = torch.cat(
            [
                torch.full((micro_count,), _MICRO_NODE, dtype=torch.long),
                torch.full((pocket_count,), _P2RANK_NODE, dtype=torch.long),
            ],
            dim=0,
        )
        context_node_raw = torch.cat([micro_raw, pocket_feature], dim=0) if micro_count + pocket_count > 0 else torch.zeros((0, POMOG_POCKET_NODE_RAW_DIM), dtype=torch.float32)
        if micro_member.numel() > 0 and pocket_member.numel() > 0:
            pocket_context_member = pocket_member.clone()
            pocket_context_member[1] += micro_count
            context_member = torch.cat([micro_member, pocket_context_member], dim=1)
        elif micro_member.numel() > 0:
            context_member = micro_member.clone()
        elif pocket_member.numel() > 0:
            context_member = pocket_member.clone()
            context_member[1] += micro_count
        else:
            context_member = torch.zeros((2, 0), dtype=torch.long)
        context_graph_edge_index, context_graph_edge_attr = _build_sparse_context_graph(
            motifs=motifs,
            micro_assignment=assignment,
            micro_centroid=micro_centroid,
            pocket_centers=pocket_centers,
            pocket_member_index=pocket_member,
            ca_pos=ca_pos,
            micro_knn=self.micro_knn,
            micro_contact_radius=self.micro_contact_radius,
            pocket_link_radius=self.pocket_link_radius,
            pocket_to_pocket_knn=self.pocket_to_pocket_knn,
        )
        context_edge_index = _build_context_edges(
            length=length,
            micro_assignment=assignment,
            micro_count=micro_count,
            pocket_centers=pocket_centers,
            pocket_member_index=pocket_member,
            ca_pos=ca_pos,
            pocket_context_top_k=self.pocket_context_top_k,
        )

        return {
            "cache_type": "pomog_pocket_guided",
            "cache_version": POMOG_POCKET_CACHE_VERSION,
            "fingerprint": self.cache_fingerprint,
            "name": name,
            "length": length,
            "residue_pocket_feature": residue_feat.float(),
            "pocket_feat": pocket_feat.float(),
            "in_any_pocket": in_any_pocket.bool(),
            "in_p2rank_topk": in_p2rank_topk.bool(),
            "in_fpocket_topk": in_fpocket_topk.bool(),
            "p2rank_membership": p2rank_membership.bool(),
            "fpocket_membership": fpocket_membership.bool(),
            "pocket_missing": pocket_missing.bool(),
            "residue_dssp8_id": dssp.long(),
            "residue_rsa_bin": rsa_bin.long(),
            "residue_boundary_distance": boundary_dist.float(),
            "micro_motif_assignment": assignment.long(),
            "micro_motif_type": motif_type.long(),
            "micro_motif_length": micro_length.float(),
            "micro_motif_centroid": micro_centroid.float(),
            "micro_motif_raw_feature": micro_raw.float(),
            "micro_motif_member_index": micro_member.long(),
            "pocket_node_feature": pocket_feature.float(),
            "pocket_center": pocket_centers.float(),
            "pocket_rank_score": pocket_rank_score.float(),
            "pocket_member_index": pocket_member.long(),
            "context_node_member_index": context_member.long(),
            "context_node_type": context_node_type.long(),
            "context_node_raw_feature": context_node_raw.float(),
            "context_graph_edge_index": context_graph_edge_index.long(),
            "context_graph_edge_attr": context_graph_edge_attr.float(),
            "context_edge_index": context_edge_index.long(),
        }

    def __getitem__(self, idx: int) -> PoMoGPocketData:
        t0 = time.perf_counter()
        cache_status = "disabled"
        base_graph = self.base[idx]
        graph = PoMoGPocketData(**base_graph.to_dict())
        name = str(graph.name)
        length = int(graph.seq.shape[0])
        cache_path = cache_file(self.cache_dir, name) if self.cache_dir is not None else None
        cached = _load_cache(cache_path, self.cache_fingerprint, length) if cache_path and cache_path.exists() else None
        cache_status = "hit" if cached is not None else "miss"

        if cached is None:
            if self.cache_required and cache_path is not None:
                raise FileNotFoundError(f"[pomog-pocket-cache] missing cache file: {cache_path}")
            cached = self._build_payload(graph, name, length)
            if cache_path is not None and self.cache_write:
                _save_cache(cache_path, cached)

        for key, value in cached.items():
            if isinstance(value, torch.Tensor):
                setattr(graph, key, value)
        _ensure_pocket_schema(graph)

        graph.micro_motif_label = _labels_from_member(
            graph.activate_site,
            graph.micro_motif_member_index,
            int(graph.micro_motif_type.shape[0]),
        )
        graph.pocket_label = _labels_from_member(
            graph.activate_site,
            graph.pocket_member_index,
            int(graph.pocket_node_feature.shape[0]),
        )
        graph.context_node_label = torch.cat([graph.micro_motif_label, graph.pocket_label], dim=0)
        graph.micro_motif_batch = torch.zeros((graph.micro_motif_type.shape[0],), dtype=torch.long)
        graph.pocket_batch = torch.zeros((graph.pocket_node_feature.shape[0],), dtype=torch.long)
        graph.context_node_batch = torch.zeros((graph.context_node_type.shape[0],), dtype=torch.long)
        graph.pomog_pocket_counts = torch.tensor(
            [
                float(graph.micro_motif_type.shape[0]),
                float(graph.pocket_node_feature.shape[0]),
                float(graph.context_graph_edge_index.shape[1]),
                float(graph.pocket_member_index.shape[1]),
            ],
            dtype=torch.float32,
        )
        if self.structure_prior_cache_enabled:
            prior_payload = load_structure_prior_for_protein(
                self.structure_prior_cache_root,
                name,
                expected_length=length,
                required=self.structure_prior_cache_required,
            )
            if prior_payload is None:
                prior_payload = empty_structure_prior_payload(name, length)
            graph.structure_prior_feat = prior_payload["residue_prior_feat"].float()
            graph.structure_prior_mask = prior_payload["residue_prior_mask"].bool()
            graph.structure_prior_residue_to_pocket_index = prior_payload["residue_to_pocket_index"].long()
            graph.structure_prior_residue_to_pocket_attr = prior_payload["residue_to_pocket_attr"].float()
            graph.structure_prior_detector_missing = detector_missing_tensor(prior_payload).view(1, -1)
            table = prior_payload.get("pocket_table", {})
            source_id = table.get("source_id", torch.zeros((0,), dtype=torch.long)).float().view(-1, 1)
            kind_id = table.get("kind_id", torch.zeros((0,), dtype=torch.long)).float().view(-1, 1)
            rank = table.get("rank", torch.zeros((0,), dtype=torch.long)).float().view(-1, 1)
            score = table.get("score", torch.zeros((0,), dtype=torch.float32)).float().view(-1, 1)
            size = table.get("size", torch.zeros((0,), dtype=torch.float32)).float().view(-1, 1)
            center = table.get("center", torch.zeros((0, 3), dtype=torch.float32)).float().view(-1, 3)
            if rank.numel() > 0:
                graph.structure_prior_pocket_feature = torch.cat(
                    [source_id, kind_id, 1.0 / rank.clamp_min(1.0), score, size / float(max(length, 1)), center / 100.0],
                    dim=-1,
                )
            else:
                graph.structure_prior_pocket_feature = torch.zeros((0, 8), dtype=torch.float32)
            graph.structure_prior_counts = torch.tensor(
                [
                    float(graph.structure_prior_pocket_feature.shape[0]),
                    float(graph.structure_prior_residue_to_pocket_index.shape[1]),
                    float(graph.structure_prior_mask.float().mean().item()) if graph.structure_prior_mask.numel() else 0.0,
                    float(STRUCTURE_PRIOR_FEATURE_DIM),
                    float(len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)),
                ],
                dtype=torch.float32,
            )

        elapsed = time.perf_counter() - t0
        if self.slow_sample_log_threshold > 0 and elapsed >= self.slow_sample_log_threshold:
            print(
                "[SLOW_SAMPLE] dataset=pomog_pocket "
                f"idx={idx} name={name} length={length} seconds={elapsed:.3f} "
                f"cache={cache_status} micro={int(graph.micro_motif_type.shape[0])} "
                f"pockets={int(graph.pocket_node_feature.shape[0])} "
                f"context_edges={int(graph.context_edge_index.shape[1])} "
                f"cache_path={cache_path}",
                flush=True,
            )
        return graph
