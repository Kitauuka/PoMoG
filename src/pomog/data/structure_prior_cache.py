from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = 2
SCHEMA_NAME = "pomog_external_structure_prior_cache"

STRUCTURE_PRIOR_FEATURE_NAMES: list[str] = [
    "p2rank_in_top1",
    "p2rank_in_top3",
    "p2rank_in_top5",
    "fpocket_in_top1",
    "fpocket_in_top3",
    "fpocket_in_top5",
    "p2rank_score_best",
    "fpocket_score_best",
    "p2rank_rank_recip",
    "fpocket_rank_recip",
    "dist_to_nearest_p2rank",
    "dist_to_nearest_fpocket",
    "p2rank_fpocket_agree",
    "detector_count",
    "castp_surface_pocket",
    "castp_interior_cavity",
    "castp_channel",
    "dist_to_castp_feature",
    "castp_volume_norm",
    "castp_area_norm",
    "caver_tunnel_lining",
    "dist_to_caver_centerline",
    "caver_bottleneck_radius",
    "pykvfinder_cavity",
    "dist_to_pykvfinder_cavity",
    "pykvfinder_volume_norm",
    "ghecom_pocket",
    "dist_to_ghecom_pocket",
    "ghecom_depth_norm",
    "mole_tunnel_lining",
    "dist_to_mole_centerline",
    "mole_bottleneck_radius",
    "rsa",
    "local_contact_density",
    "boundary_distance_norm",
    "concavity_proxy",
]

FEATURE_INDEX = {name: idx for idx, name in enumerate(STRUCTURE_PRIOR_FEATURE_NAMES)}
STRUCTURE_PRIOR_FEATURE_DIM = len(STRUCTURE_PRIOR_FEATURE_NAMES)
STRUCTURE_PRIOR_EDGE_ATTR_NAMES = [
    "source_id_norm",
    "rank_recip",
    "score",
    "center_distance",
    "region_size_norm",
]
DETECTOR_NAMES = ("p2rank", "fpocket", "castp", "caver", "pykvfinder", "ghecom", "mole")
DETECTOR_FEATURE_GROUPS: dict[str, list[int]] = {
    "p2rank": [
        FEATURE_INDEX["p2rank_in_top1"],
        FEATURE_INDEX["p2rank_in_top3"],
        FEATURE_INDEX["p2rank_in_top5"],
        FEATURE_INDEX["p2rank_score_best"],
        FEATURE_INDEX["p2rank_rank_recip"],
        FEATURE_INDEX["dist_to_nearest_p2rank"],
    ],
    "fpocket": [
        FEATURE_INDEX["fpocket_in_top1"],
        FEATURE_INDEX["fpocket_in_top3"],
        FEATURE_INDEX["fpocket_in_top5"],
        FEATURE_INDEX["fpocket_score_best"],
        FEATURE_INDEX["fpocket_rank_recip"],
        FEATURE_INDEX["dist_to_nearest_fpocket"],
    ],
    "castp": [
        FEATURE_INDEX["castp_surface_pocket"],
        FEATURE_INDEX["castp_interior_cavity"],
        FEATURE_INDEX["castp_channel"],
        FEATURE_INDEX["dist_to_castp_feature"],
        FEATURE_INDEX["castp_volume_norm"],
        FEATURE_INDEX["castp_area_norm"],
    ],
    "caver": [
        FEATURE_INDEX["caver_tunnel_lining"],
        FEATURE_INDEX["dist_to_caver_centerline"],
        FEATURE_INDEX["caver_bottleneck_radius"],
    ],
    "pykvfinder": [
        FEATURE_INDEX["pykvfinder_cavity"],
        FEATURE_INDEX["dist_to_pykvfinder_cavity"],
        FEATURE_INDEX["pykvfinder_volume_norm"],
    ],
    "ghecom": [
        FEATURE_INDEX["ghecom_pocket"],
        FEATURE_INDEX["dist_to_ghecom_pocket"],
        FEATURE_INDEX["ghecom_depth_norm"],
    ],
    "mole": [
        FEATURE_INDEX["mole_tunnel_lining"],
        FEATURE_INDEX["dist_to_mole_centerline"],
        FEATURE_INDEX["mole_bottleneck_radius"],
    ],
}
LOCAL_GEOMETRY_FEATURES = [
    FEATURE_INDEX["rsa"],
    FEATURE_INDEX["local_contact_density"],
    FEATURE_INDEX["boundary_distance_norm"],
    FEATURE_INDEX["concavity_proxy"],
]

SOURCE_TO_ID = {"p2rank": 0, "fpocket": 1, "castp": 2, "caver": 3, "pykvfinder": 4, "ghecom": 5, "mole": 6}
KIND_TO_ID = {
    "pocket": 0,
    "surface_pocket": 1,
    "interior_cavity": 2,
    "channel": 3,
    "tunnel": 4,
    "cavity": 5,
}


@dataclass(frozen=True)
class PriorRegion:
    source: str
    kind: str
    rank: int
    score: float
    residues: torch.Tensor
    center: torch.Tensor
    volume: float = 0.0
    area: float = 0.0
    bottleneck_radius: float = 0.0


def schema_payload() -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "feature_names": STRUCTURE_PRIOR_FEATURE_NAMES,
        "feature_dim": STRUCTURE_PRIOR_FEATURE_DIM,
        "edge_attr_names": STRUCTURE_PRIOR_EDGE_ATTR_NAMES,
        "detectors": list(DETECTOR_NAMES),
        "required_payload_keys": [
            "protein_id",
            "L",
            "residue_prior_feat",
            "residue_prior_mask",
            "pocket_table",
            "residue_to_pocket_index",
            "residue_to_pocket_attr",
            "detector_missing",
            "feature_names",
            "source_versions",
        ],
        "missing_policy": "Missing detector or cache files produce zero features with residue_prior_mask=False for detector-owned features.",
    }


def write_schema_json(cache_root: str | Path) -> Path:
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "schema.json"
    path.write_text(json.dumps(schema_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def candidate_pt_paths(root: str | Path | None, name: str, *, source: str | None = None) -> list[Path]:
    if not root:
        return []
    base = Path(root).expanduser().resolve()
    stems = [str(name), str(name).upper(), str(name).lower()]
    paths: list[Path] = []
    if source:
        paths.extend(base / str(source).lower() / f"{stem}.pt" for stem in stems)
    paths.extend(base / f"{stem}.pt" for stem in stems)
    return paths


def load_detector_payload(root: str | Path | None, source: str, protein_id: str, length: int | None = None) -> dict[str, Any] | None:
    for path in candidate_pt_paths(root, protein_id, source=source):
        if not path.exists():
            continue
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload_len = payload.get("L", payload.get("length", payload.get("seq_len")))
        if length is not None and payload_len is not None and int(payload_len) != int(length):
            continue
        return payload
    return None


def _as_tensor(value: Any, *, dtype: torch.dtype, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=dtype)
    except Exception:
        tensor = torch.zeros(shape or (0,), dtype=dtype)
    if shape is not None and tuple(tensor.shape) != shape:
        flat = tensor.reshape(-1)
        size = 1
        for dim in shape:
            size *= int(dim)
        out = torch.zeros((size,), dtype=dtype)
        take = min(size, int(flat.numel()))
        if take > 0:
            out[:take] = flat[:take].to(dtype)
        tensor = out.view(*shape)
    if tensor.is_floating_point():
        tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0).to(dtype)
    return tensor


def _residue_indices_from_item(item: Mapping[str, Any], length: int) -> torch.Tensor:
    for key in ("residue_index", "residue_indices", "residues", "residue_ids"):
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, str):
            values = []
            for token in value.replace(";", ",").replace(" ", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    raw = int(float(token.split(":")[-1]))
                except ValueError:
                    continue
                values.append(raw - 1 if raw >= 1 else raw)
            idx = torch.tensor(values, dtype=torch.long)
        else:
            idx = torch.as_tensor(value, dtype=torch.long).view(-1)
        idx = idx[(idx >= 0) & (idx < int(length))]
        return idx.unique(sorted=True)
    return torch.zeros((0,), dtype=torch.long)


def _center_from_item(item: Mapping[str, Any], residues: torch.Tensor, ca_pos: torch.Tensor) -> torch.Tensor:
    center_value = item.get("center", item.get("centroid"))
    if center_value is None:
        xyz_keys = ("x", "y", "z")
        if all(key in item for key in xyz_keys):
            center_value = [item[key] for key in xyz_keys]
        elif all(f"center_{key}" in item for key in xyz_keys):
            center_value = [item[f"center_{key}"] for key in xyz_keys]
    if center_value is not None:
        center = _as_tensor(center_value, dtype=torch.float32, shape=(3,))
        if torch.isfinite(center).all() and float(center.abs().sum().item()) > 1e-8:
            return center.float()
    if residues.numel() > 0 and ca_pos.numel() > 0:
        return ca_pos[residues.long()].mean(dim=0).float()
    return torch.zeros((3,), dtype=torch.float32)


def _region_lists(payload: Mapping[str, Any] | None, source: str) -> list[tuple[str, list[Any]]]:
    if not payload:
        return []
    source = source.lower()
    if source in {"p2rank", "fpocket"}:
        return [("pocket", list(payload.get("pockets", payload.get("regions", [])) or []))]
    if source == "castp":
        return [
            ("surface_pocket", list(payload.get("surface_pockets", payload.get("pockets", [])) or [])),
            ("interior_cavity", list(payload.get("interior_cavities", payload.get("cavities", [])) or [])),
            ("channel", list(payload.get("channels", [])) or []),
        ]
    if source == "caver":
        return [("tunnel", list(payload.get("tunnels", payload.get("channels", [])) or []))]
    if source == "pykvfinder":
        return [("cavity", list(payload.get("cavities", payload.get("regions", [])) or []))]
    if source == "ghecom":
        return [("pocket", list(payload.get("pockets", payload.get("regions", [])) or []))]
    if source == "mole":
        return [("tunnel", list(payload.get("tunnels", payload.get("channels", [])) or []))]
    return [("pocket", list(payload.get("regions", [])) or [])]


def coerce_regions(
    payload: Mapping[str, Any] | None,
    *,
    source: str,
    length: int,
    ca_pos: torch.Tensor,
    top_k: int = 5,
) -> list[PriorRegion]:
    regions: list[PriorRegion] = []
    for kind, rows in _region_lists(payload, source):
        for row_id, item in enumerate(rows, start=1):
            if not isinstance(item, Mapping):
                continue
            residues = _residue_indices_from_item(item, length)
            if residues.numel() == 0:
                continue
            rank = int(float(item.get("rank", item.get("id", row_id)) or row_id))
            score = float(item.get("score", item.get("probability", item.get("prob", 0.0))) or 0.0)
            center = _center_from_item(item, residues, ca_pos)
            regions.append(
                PriorRegion(
                    source=str(source).lower(),
                    kind=str(item.get("kind", kind)).lower(),
                    rank=max(1, rank),
                    score=score,
                    residues=residues.long(),
                    center=center.float(),
                    volume=float(item.get("volume", item.get("vol", 0.0)) or 0.0),
                    area=float(item.get("area", item.get("mouth_area", 0.0)) or 0.0),
                    bottleneck_radius=float(item.get("bottleneck_radius", item.get("radius", 0.0)) or 0.0),
                )
            )
    regions.sort(key=lambda row: (row.source, row.kind, row.rank, -row.score))
    if int(top_k) > 0:
        grouped: dict[tuple[str, str], list[PriorRegion]] = {}
        for region in regions:
            grouped.setdefault((region.source, region.kind), []).append(region)
        regions = [region for key in sorted(grouped) for region in grouped[key][: int(top_k)]]
    return regions


def dssp8_index_from_node_feat(node_feat: torch.Tensor) -> torch.Tensor:
    length = int(node_feat.shape[0]) if node_feat.ndim >= 2 else 0
    if length <= 0 or node_feat.shape[-1] < 1033:
        return torch.full((length,), 7, dtype=torch.long)
    ss = node_feat[:, 1025:1033].float().cpu()
    idx = torch.argmax(ss, dim=-1).long()
    idx[ss.sum(dim=-1) <= 0.0] = 7
    return idx


def rsa_from_node_feat(node_feat: torch.Tensor) -> torch.Tensor:
    length = int(node_feat.shape[0]) if node_feat.ndim >= 2 else 0
    if length <= 0 or node_feat.shape[-1] <= 1024:
        return torch.zeros((length,), dtype=torch.float32)
    return torch.nan_to_num(node_feat[:, 1024].float().cpu(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def boundary_distance_from_dssp(dssp8: torch.Tensor, *, cap: int = 8) -> torch.Tensor:
    length = int(dssp8.numel())
    if length <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    if length == 1:
        return torch.ones((1,), dtype=torch.float32)
    changes = torch.where(dssp8[1:] != dssp8[:-1])[0] + 1
    if changes.numel() == 0:
        return torch.ones((length,), dtype=torch.float32)
    pos = torch.arange(length, dtype=torch.float32).view(-1, 1)
    dist = torch.abs(pos - changes.float().view(1, -1)).min(dim=1).values
    return (dist / float(max(1, cap))).clamp(0.0, 1.0).float()


def contact_density(ca_pos: torch.Tensor, *, radius: float = 10.0) -> torch.Tensor:
    length = int(ca_pos.shape[0])
    if length <= 0:
        return torch.zeros((0,), dtype=torch.float32)
    dist = torch.cdist(ca_pos.float(), ca_pos.float(), p=2.0)
    degree = ((dist <= float(radius)) & (dist > 0.0)).sum(dim=1).float()
    denom = max(float(degree.max().item()), 1.0)
    return (degree / denom).clamp(0.0, 1.0)


def local_geometry_features(node_feat: torch.Tensor, ca_pos: torch.Tensor) -> dict[str, torch.Tensor]:
    rsa = rsa_from_node_feat(node_feat)
    density = contact_density(ca_pos, radius=10.0)
    boundary = boundary_distance_from_dssp(dssp8_index_from_node_feat(node_feat))
    concavity = ((1.0 - rsa.float().clamp(0.0, 1.0)) * density.float().clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return {
        "rsa": rsa.float(),
        "local_contact_density": density.float(),
        "boundary_distance_norm": boundary.float(),
        "concavity_proxy": concavity.float(),
    }


def _mark_detector_mask(mask: torch.Tensor, detector: str, present: bool) -> None:
    if present:
        mask[:, DETECTOR_FEATURE_GROUPS[detector]] = True


def _fill_pocket_like_features(
    feat: torch.Tensor,
    mask: torch.Tensor,
    regions: Sequence[PriorRegion],
    *,
    source: str,
    ca_pos: torch.Tensor,
) -> None:
    if not regions:
        return
    source = source.lower()
    if source == "p2rank":
        top_cols = ("p2rank_in_top1", "p2rank_in_top3", "p2rank_in_top5")
        score_col = "p2rank_score_best"
        rank_col = "p2rank_rank_recip"
        dist_col = "dist_to_nearest_p2rank"
    else:
        top_cols = ("fpocket_in_top1", "fpocket_in_top3", "fpocket_in_top5")
        score_col = "fpocket_score_best"
        rank_col = "fpocket_rank_recip"
        dist_col = "dist_to_nearest_fpocket"
    centers = torch.stack([region.center.float() for region in regions], dim=0)
    nearest = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
    feat[:, FEATURE_INDEX[dist_col]] = nearest
    mask[:, FEATURE_INDEX[dist_col]] = True
    for region in regions:
        idx = region.residues.long()
        rank = max(1, int(region.rank))
        if rank == 1:
            feat[idx, FEATURE_INDEX[top_cols[0]]] = 1.0
        if rank <= 3:
            feat[idx, FEATURE_INDEX[top_cols[1]]] = 1.0
        if rank <= 5:
            feat[idx, FEATURE_INDEX[top_cols[2]]] = 1.0
        recip = 1.0 / float(rank)
        update = recip > feat[idx, FEATURE_INDEX[rank_col]]
        target = idx[update]
        feat[target, FEATURE_INDEX[rank_col]] = recip
        feat[target, FEATURE_INDEX[score_col]] = float(region.score)


def _fill_castp_features(feat: torch.Tensor, mask: torch.Tensor, regions: Sequence[PriorRegion], ca_pos: torch.Tensor) -> None:
    if not regions:
        return
    centers = torch.stack([region.center.float() for region in regions], dim=0)
    feat[:, FEATURE_INDEX["dist_to_castp_feature"]] = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
    volumes = [max(0.0, float(region.volume)) for region in regions]
    areas = [max(0.0, float(region.area)) for region in regions]
    vol_denom = max(max(volumes, default=0.0), 1.0)
    area_denom = max(max(areas, default=0.0), 1.0)
    for region in regions:
        idx = region.residues.long()
        if region.kind == "channel":
            feat[idx, FEATURE_INDEX["castp_channel"]] = 1.0
        elif region.kind == "interior_cavity":
            feat[idx, FEATURE_INDEX["castp_interior_cavity"]] = 1.0
        else:
            feat[idx, FEATURE_INDEX["castp_surface_pocket"]] = 1.0
        feat[idx, FEATURE_INDEX["castp_volume_norm"]] = max(
            feat[idx, FEATURE_INDEX["castp_volume_norm"]].max().item() if idx.numel() else 0.0,
            min(max(0.0, float(region.volume)) / vol_denom, 1.0),
        )
        feat[idx, FEATURE_INDEX["castp_area_norm"]] = max(
            feat[idx, FEATURE_INDEX["castp_area_norm"]].max().item() if idx.numel() else 0.0,
            min(max(0.0, float(region.area)) / area_denom, 1.0),
        )


def _fill_caver_features(feat: torch.Tensor, mask: torch.Tensor, regions: Sequence[PriorRegion], ca_pos: torch.Tensor) -> None:
    if not regions:
        return
    centers = torch.stack([region.center.float() for region in regions], dim=0)
    feat[:, FEATURE_INDEX["dist_to_caver_centerline"]] = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
    radius_denom = max(max((float(region.bottleneck_radius) for region in regions), default=0.0), 1.0)
    for region in regions:
        idx = region.residues.long()
        feat[idx, FEATURE_INDEX["caver_tunnel_lining"]] = 1.0
        feat[idx, FEATURE_INDEX["caver_bottleneck_radius"]] = min(max(0.0, float(region.bottleneck_radius)) / radius_denom, 1.0)


def _fill_local_cavity_features(
    feat: torch.Tensor,
    mask: torch.Tensor,
    regions: Sequence[PriorRegion],
    ca_pos: torch.Tensor,
    *,
    source: str,
) -> None:
    if not regions:
        return
    centers = torch.stack([region.center.float() for region in regions], dim=0)
    nearest = torch.cdist(ca_pos.float(), centers.float(), p=2.0).min(dim=1).values.clamp(max=30.0)
    if source == "pykvfinder":
        feat[:, FEATURE_INDEX["dist_to_pykvfinder_cavity"]] = nearest
        volumes = [max(0.0, float(region.volume)) for region in regions]
        volume_denom = max(max(volumes, default=0.0), 1.0)
        for region in regions:
            idx = region.residues.long()
            feat[idx, FEATURE_INDEX["pykvfinder_cavity"]] = 1.0
            feat[idx, FEATURE_INDEX["pykvfinder_volume_norm"]] = max(
                feat[idx, FEATURE_INDEX["pykvfinder_volume_norm"]].max().item() if idx.numel() else 0.0,
                min(max(0.0, float(region.volume)) / volume_denom, 1.0),
            )
    elif source == "ghecom":
        feat[:, FEATURE_INDEX["dist_to_ghecom_pocket"]] = nearest
        depth_denom = max(max((float(region.score) for region in regions), default=0.0), 1.0)
        for region in regions:
            idx = region.residues.long()
            feat[idx, FEATURE_INDEX["ghecom_pocket"]] = 1.0
            feat[idx, FEATURE_INDEX["ghecom_depth_norm"]] = min(max(0.0, float(region.score)) / depth_denom, 1.0)
    elif source == "mole":
        feat[:, FEATURE_INDEX["dist_to_mole_centerline"]] = nearest
        radius_denom = max(max((float(region.bottleneck_radius) for region in regions), default=0.0), 1.0)
        for region in regions:
            idx = region.residues.long()
            feat[idx, FEATURE_INDEX["mole_tunnel_lining"]] = 1.0
            feat[idx, FEATURE_INDEX["mole_bottleneck_radius"]] = min(max(0.0, float(region.bottleneck_radius)) / radius_denom, 1.0)


def _pocket_table_and_edges(regions: Sequence[PriorRegion], length: int, ca_pos: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not regions:
        table = {
            "source_id": torch.zeros((0,), dtype=torch.long),
            "kind_id": torch.zeros((0,), dtype=torch.long),
            "rank": torch.zeros((0,), dtype=torch.long),
            "score": torch.zeros((0,), dtype=torch.float32),
            "size": torch.zeros((0,), dtype=torch.float32),
            "center": torch.zeros((0, 3), dtype=torch.float32),
        }
        return table, torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)), dtype=torch.float32)
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    attrs: list[torch.Tensor] = []
    for region_id, region in enumerate(regions):
        residues = region.residues.long()
        if residues.numel() == 0:
            continue
        center_dist = torch.linalg.norm(ca_pos[residues].float() - region.center.view(1, 3).float(), dim=-1).clamp(max=30.0)
        rows.append(residues)
        cols.append(torch.full_like(residues, int(region_id)))
        source_norm = float(SOURCE_TO_ID.get(region.source, 0)) / float(max(len(SOURCE_TO_ID) - 1, 1))
        attr = torch.stack(
            [
                torch.full((int(residues.numel()),), source_norm, dtype=torch.float32),
                torch.full((int(residues.numel()),), 1.0 / float(max(1, region.rank)), dtype=torch.float32),
                torch.full((int(residues.numel()),), float(region.score), dtype=torch.float32),
                center_dist.float(),
                torch.full((int(residues.numel()),), float(residues.numel()) / float(max(length, 1)), dtype=torch.float32),
            ],
            dim=-1,
        )
        attrs.append(attr)
    table = {
        "source_id": torch.tensor([SOURCE_TO_ID.get(region.source, 0) for region in regions], dtype=torch.long),
        "kind_id": torch.tensor([KIND_TO_ID.get(region.kind, 0) for region in regions], dtype=torch.long),
        "rank": torch.tensor([max(1, int(region.rank)) for region in regions], dtype=torch.long),
        "score": torch.tensor([float(region.score) for region in regions], dtype=torch.float32),
        "size": torch.tensor([float(region.residues.numel()) for region in regions], dtype=torch.float32),
        "center": torch.stack([region.center.float().view(3) for region in regions], dim=0),
    }
    edge_index = torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0).long() if rows else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.cat(attrs, dim=0).float() if attrs else torch.zeros((0, len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)), dtype=torch.float32)
    return table, edge_index, edge_attr


def empty_structure_prior_payload(
    protein_id: str,
    length: int,
    *,
    include_local_geometry: bool = False,
    node_feat: torch.Tensor | None = None,
    ca_pos: torch.Tensor | None = None,
) -> dict[str, Any]:
    feat = torch.zeros((int(length), STRUCTURE_PRIOR_FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((int(length), STRUCTURE_PRIOR_FEATURE_DIM), dtype=torch.bool)
    if include_local_geometry and node_feat is not None and ca_pos is not None:
        local = local_geometry_features(node_feat, ca_pos)
        for name, values in local.items():
            feat[:, FEATURE_INDEX[name]] = values.float().view(-1)[: int(length)]
            mask[:, FEATURE_INDEX[name]] = True
    table, edge_index, edge_attr = _pocket_table_and_edges([], int(length), torch.zeros((int(length), 3), dtype=torch.float32))
    return {
        "protein_id": str(protein_id),
        "L": int(length),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "residue_prior_feat": feat,
        "residue_prior_mask": mask,
        "pocket_table": table,
        "residue_to_pocket_index": edge_index,
        "residue_to_pocket_attr": edge_attr,
        "detector_missing": {name: True for name in DETECTOR_NAMES},
        "feature_names": list(STRUCTURE_PRIOR_FEATURE_NAMES),
        "source_versions": {},
    }


def build_structure_prior_payload(
    *,
    protein_id: str,
    length: int,
    ca_pos: torch.Tensor,
    node_feat: torch.Tensor,
    p2rank_payload: Mapping[str, Any] | None = None,
    fpocket_payload: Mapping[str, Any] | None = None,
    castp_payload: Mapping[str, Any] | None = None,
    caver_payload: Mapping[str, Any] | None = None,
    pykvfinder_payload: Mapping[str, Any] | None = None,
    ghecom_payload: Mapping[str, Any] | None = None,
    mole_payload: Mapping[str, Any] | None = None,
    top_k: int = 5,
    source_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    length = int(length)
    ca_pos = _as_tensor(ca_pos, dtype=torch.float32, shape=(length, 3))
    feat = torch.zeros((length, STRUCTURE_PRIOR_FEATURE_DIM), dtype=torch.float32)
    mask = torch.zeros((length, STRUCTURE_PRIOR_FEATURE_DIM), dtype=torch.bool)
    detector_payloads = {
        "p2rank": p2rank_payload,
        "fpocket": fpocket_payload,
        "castp": castp_payload,
        "caver": caver_payload,
        "pykvfinder": pykvfinder_payload,
        "ghecom": ghecom_payload,
        "mole": mole_payload,
    }
    detector_missing = {name: detector_payloads[name] is None for name in DETECTOR_NAMES}

    regions_by_source = {
        source: coerce_regions(payload, source=source, length=length, ca_pos=ca_pos, top_k=top_k)
        for source, payload in detector_payloads.items()
    }
    for source, regions in regions_by_source.items():
        _mark_detector_mask(mask, source, present=not detector_missing[source])
        if source in {"p2rank", "fpocket"}:
            _fill_pocket_like_features(feat, mask, regions, source=source, ca_pos=ca_pos)
        elif source == "castp":
            _fill_castp_features(feat, mask, regions, ca_pos)
        elif source == "caver":
            _fill_caver_features(feat, mask, regions, ca_pos)
        elif source in {"pykvfinder", "ghecom", "mole"}:
            _fill_local_cavity_features(feat, mask, regions, ca_pos, source=source)

    p2_cover = feat[:, FEATURE_INDEX["p2rank_in_top5"]] > 0.5
    fp_cover = feat[:, FEATURE_INDEX["fpocket_in_top5"]] > 0.5
    castp_cover = (
        (feat[:, FEATURE_INDEX["castp_surface_pocket"]] > 0.5)
        | (feat[:, FEATURE_INDEX["castp_interior_cavity"]] > 0.5)
        | (feat[:, FEATURE_INDEX["castp_channel"]] > 0.5)
    )
    caver_cover = feat[:, FEATURE_INDEX["caver_tunnel_lining"]] > 0.5
    pykv_cover = feat[:, FEATURE_INDEX["pykvfinder_cavity"]] > 0.5
    ghecom_cover = feat[:, FEATURE_INDEX["ghecom_pocket"]] > 0.5
    mole_cover = feat[:, FEATURE_INDEX["mole_tunnel_lining"]] > 0.5
    feat[:, FEATURE_INDEX["p2rank_fpocket_agree"]] = (p2_cover & fp_cover).float()
    feat[:, FEATURE_INDEX["detector_count"]] = (
        p2_cover.float()
        + fp_cover.float()
        + castp_cover.float()
        + caver_cover.float()
        + pykv_cover.float()
        + ghecom_cover.float()
        + mole_cover.float()
    )
    mask[:, FEATURE_INDEX["p2rank_fpocket_agree"]] = (not detector_missing["p2rank"]) or (not detector_missing["fpocket"])
    mask[:, FEATURE_INDEX["detector_count"]] = True

    local = local_geometry_features(node_feat, ca_pos)
    for name, values in local.items():
        feat[:, FEATURE_INDEX[name]] = values.float().view(-1)[:length]
        mask[:, FEATURE_INDEX[name]] = True

    all_regions = [region for source in DETECTOR_NAMES for region in regions_by_source[source]]
    table, edge_index, edge_attr = _pocket_table_and_edges(all_regions, length, ca_pos)
    return sanitize_structure_prior_payload(
        {
            "protein_id": str(protein_id),
            "L": length,
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "residue_prior_feat": feat,
            "residue_prior_mask": mask,
            "pocket_table": table,
            "residue_to_pocket_index": edge_index,
            "residue_to_pocket_attr": edge_attr,
            "detector_missing": detector_missing,
            "feature_names": list(STRUCTURE_PRIOR_FEATURE_NAMES),
            "source_versions": dict(source_versions or {}),
        }
    )


def sanitize_structure_prior_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("structure prior payload must be a mapping")
    protein_id = str(payload.get("protein_id", payload.get("name", "")))
    length = int(payload.get("L", payload.get("length", 0)))
    if length < 0:
        raise ValueError(f"invalid structure prior length for {protein_id}: {length}")
    feat = _as_tensor(payload.get("residue_prior_feat"), dtype=torch.float32, shape=(length, STRUCTURE_PRIOR_FEATURE_DIM))
    mask = _as_tensor(payload.get("residue_prior_mask"), dtype=torch.bool, shape=(length, STRUCTURE_PRIOR_FEATURE_DIM)).bool()
    edge_index = _as_tensor(payload.get("residue_to_pocket_index"), dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_index = edge_index.view(2, -1).long()
    edge_attr = _as_tensor(payload.get("residue_to_pocket_attr"), dtype=torch.float32)
    if edge_attr.numel() == 0:
        edge_attr = torch.zeros((edge_index.shape[1], len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)), dtype=torch.float32)
    edge_attr = edge_attr.view(-1, len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)).float()
    if edge_attr.shape[0] != edge_index.shape[1]:
        fixed = torch.zeros((edge_index.shape[1], len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES)), dtype=torch.float32)
        take = min(int(edge_attr.shape[0]), int(fixed.shape[0]))
        fixed[:take] = edge_attr[:take]
        edge_attr = fixed
    detector_missing_raw = payload.get("detector_missing", {})
    detector_missing = {
        name: bool(detector_missing_raw.get(name, True)) if isinstance(detector_missing_raw, Mapping) else True
        for name in DETECTOR_NAMES
    }
    table = payload.get("pocket_table")
    if not isinstance(table, Mapping):
        table, _, _ = _pocket_table_and_edges([], length, torch.zeros((length, 3), dtype=torch.float32))
    out_table = {
        "source_id": _as_tensor(table.get("source_id"), dtype=torch.long).view(-1),
        "kind_id": _as_tensor(table.get("kind_id"), dtype=torch.long).view(-1),
        "rank": _as_tensor(table.get("rank"), dtype=torch.long).view(-1),
        "score": _as_tensor(table.get("score"), dtype=torch.float32).view(-1),
        "size": _as_tensor(table.get("size"), dtype=torch.float32).view(-1),
        "center": _as_tensor(table.get("center"), dtype=torch.float32).view(-1, 3),
    }
    return {
        "protein_id": protein_id,
        "L": length,
        "schema_name": str(payload.get("schema_name", SCHEMA_NAME)),
        "schema_version": int(payload.get("schema_version", SCHEMA_VERSION)),
        "residue_prior_feat": feat,
        "residue_prior_mask": mask,
        "pocket_table": out_table,
        "residue_to_pocket_index": edge_index,
        "residue_to_pocket_attr": edge_attr,
        "detector_missing": detector_missing,
        "feature_names": list(STRUCTURE_PRIOR_FEATURE_NAMES),
        "source_versions": dict(payload.get("source_versions", {}) or {}),
    }


def load_structure_prior(path: str | Path, *, expected_length: int | None = None, required: bool = True) -> dict[str, Any] | None:
    path = Path(path).expanduser()
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing structure prior cache: {path}")
        return None
    payload = sanitize_structure_prior_payload(torch.load(path, map_location="cpu"))
    if expected_length is not None and int(payload["L"]) != int(expected_length):
        raise ValueError(f"structure prior length mismatch for {path}: {payload['L']} != {expected_length}")
    return payload


def load_structure_prior_for_protein(
    cache_root: str | Path | None,
    protein_id: str,
    *,
    expected_length: int | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    if not cache_root:
        if required:
            raise FileNotFoundError("structure prior cache root was not provided")
        return None
    for path in candidate_pt_paths(cache_root, protein_id):
        if path.exists():
            return load_structure_prior(path, expected_length=expected_length, required=True)
    if required:
        raise FileNotFoundError(f"missing structure prior cache for {protein_id} under {cache_root}")
    return None


def save_structure_prior(path: str | Path, payload: Mapping[str, Any]) -> None:
    clean = sanitize_structure_prior_payload(payload)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(clean, tmp)
    os.replace(tmp, path)


def detector_missing_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    missing = payload.get("detector_missing", {})
    return torch.tensor([bool(missing.get(name, True)) for name in DETECTOR_NAMES], dtype=torch.bool)


def batch_structure_prior_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    feat_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    missing_rows: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_attrs: list[torch.Tensor] = []
    residue_offset = 0
    pocket_offset = 0
    for payload in payloads:
        clean = sanitize_structure_prior_payload(payload)
        feat_rows.append(clean["residue_prior_feat"])
        mask_rows.append(clean["residue_prior_mask"])
        missing_rows.append(detector_missing_tensor(clean))
        edge_index = clean["residue_to_pocket_index"].clone()
        if edge_index.numel() > 0:
            edge_index[0] += int(residue_offset)
            edge_index[1] += int(pocket_offset)
            edge_indices.append(edge_index)
            edge_attrs.append(clean["residue_to_pocket_attr"])
        residue_offset += int(clean["L"])
        pocket_offset += int(clean["pocket_table"]["rank"].numel())
    return {
        "structure_prior_feat": torch.cat(feat_rows, dim=0) if feat_rows else torch.zeros((0, STRUCTURE_PRIOR_FEATURE_DIM)),
        "structure_prior_mask": torch.cat(mask_rows, dim=0) if mask_rows else torch.zeros((0, STRUCTURE_PRIOR_FEATURE_DIM), dtype=torch.bool),
        "structure_prior_detector_missing": torch.stack(missing_rows, dim=0) if missing_rows else torch.zeros((0, len(DETECTOR_NAMES)), dtype=torch.bool),
        "structure_prior_residue_to_pocket_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        "structure_prior_residue_to_pocket_attr": torch.cat(edge_attrs, dim=0) if edge_attrs else torch.zeros((0, len(STRUCTURE_PRIOR_EDGE_ATTR_NAMES))),
    }
