from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch


MOTIF_DSSP8_TYPE_COUNT = 8
MOTIF_RSA_BIN_COUNT = 3
MOTIF_TYPE_DSSP8_BASE = 0
MOTIF_TYPE_DSSP8_RSA3_BASE = MOTIF_DSSP8_TYPE_COUNT
MOTIF_TYPE_BOUNDARY = MOTIF_TYPE_DSSP8_RSA3_BASE + MOTIF_DSSP8_TYPE_COUNT * MOTIF_RSA_BIN_COUNT
MOTIF_TYPE_COMPONENT = MOTIF_TYPE_BOUNDARY + 1
MOTIF_TYPE_COUNT = MOTIF_TYPE_COMPONENT + 1

MOTIF_REL_SEQUENCE = 0
MOTIF_REL_SPATIAL = 1
MOTIF_REL_TOPK = 2
MOTIF_REL_CONTACT = 3
MOTIF_REL_COMPONENT = 4
MOTIF_REL_TYPE_COUNT = 5

MOTIF_EDGE_ATTR_DIM = 14
MOTIF_LOCAL_EDGE_ATTR_DIM = 4


@dataclass
class MotifPartitionResult:
    residue_to_motif: torch.Tensor
    motifs: list[torch.Tensor]
    motif_type: torch.Tensor
    motif_member_index: torch.Tensor
    primary_motif_count: int
    residue_to_micro_motif: torch.Tensor | None = None
    micro_to_component: torch.Tensor | None = None
    component_links: torch.Tensor | None = None


@dataclass
class MotifGraphResult:
    motif_length: torch.Tensor
    motif_centroid: torch.Tensor
    motif_edge_index: torch.Tensor
    motif_edge_attr: torch.Tensor
    motif_rel_edge_index: torch.Tensor
    motif_rel_edge_type: torch.Tensor
    motif_rel_edge_attr: torch.Tensor


def cache_digest(payload: Dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def external_pocket_fingerprint_payload(
    *,
    external_pocket_enabled: bool = False,
    external_pocket_cache_dir: str | None = None,
    external_pocket_source: str = "fpocket",
    external_pocket_mode: str = "append",
    external_pocket_top_k: int = 3,
    external_pocket_min_residues: int = 1,
    external_pocket_link_radius: float = 10.0,
) -> dict[str, object]:
    return {
        "external_pocket_enabled": bool(external_pocket_enabled),
        "external_pocket_cache_dir": str(external_pocket_cache_dir or ""),
        "external_pocket_source": str(external_pocket_source).lower(),
        "external_pocket_mode": str(external_pocket_mode).lower(),
        "external_pocket_top_k": int(external_pocket_top_k),
        "external_pocket_min_residues": int(external_pocket_min_residues),
        "external_pocket_link_radius": round(float(external_pocket_link_radius), 6),
        "external_pocket_schema": "v5_external_pocket_motif_v1",
    }


def _external_pocket_cache_path(cache_dir: str | None, source: str, name: str) -> Path | None:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser().resolve()
    source = str(source).lower()
    candidates = [
        root / source / f"{name}.pt",
        root / source / f"{name.upper()}.pt",
        root / source / f"{name.lower()}.pt",
        root / f"{name}.pt",
        root / f"{name.upper()}.pt",
        root / f"{name.lower()}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_external_pocket_motifs(
    *,
    name: str,
    length: int,
    cache_dir: str | None,
    source: str,
    top_k: int,
    min_residues: int,
) -> list[torch.Tensor]:
    path = _external_pocket_cache_path(cache_dir, source, name)
    if path is None:
        return []
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return []
    if not isinstance(payload, dict) or int(payload.get("length", length)) != int(length):
        return []
    pockets = payload.get("pockets", [])
    if not isinstance(pockets, list):
        return []
    out: list[torch.Tensor] = []
    for item in pockets:
        if not isinstance(item, dict):
            continue
        idx = item.get("residue_index")
        if idx is None:
            continue
        idx = torch.as_tensor(idx, dtype=torch.long).view(-1)
        idx = idx[(idx >= 0) & (idx < int(length))].unique(sorted=True)
        if int(idx.numel()) < int(min_residues):
            continue
        out.append(idx.cpu())
        if int(top_k) > 0 and len(out) >= int(top_k):
            break
    return out


def augment_partition_with_external_pockets(
    partition: MotifPartitionResult,
    *,
    name: str,
    length: int,
    ca_pos: torch.Tensor,
    external_pocket_enabled: bool = False,
    external_pocket_cache_dir: str | None = None,
    external_pocket_source: str = "fpocket",
    external_pocket_mode: str = "append",
    external_pocket_top_k: int = 3,
    external_pocket_min_residues: int = 1,
    external_pocket_link_radius: float = 10.0,
) -> MotifPartitionResult:
    if not external_pocket_enabled:
        return partition
    pocket_motifs = _load_external_pocket_motifs(
        name=name,
        length=length,
        cache_dir=external_pocket_cache_dir,
        source=external_pocket_source,
        top_k=external_pocket_top_k,
        min_residues=external_pocket_min_residues,
    )
    if not pocket_motifs:
        return partition

    mode = str(external_pocket_mode).lower()
    base_motifs = [item.long().cpu() for item in partition.motifs]
    base_types = partition.motif_type.long().cpu()
    pocket_types = torch.full((len(pocket_motifs),), MOTIF_TYPE_COMPONENT, dtype=torch.long)
    if mode in {"pocket_primary", "primary", "pocket_subgraph"}:
        motifs = [item.clone() for item in pocket_motifs] + base_motifs
        motif_type = torch.cat([pocket_types, base_types], dim=0)
        residue_to_motif = partition.residue_to_motif.long().cpu() + len(pocket_motifs)
        assigned = torch.zeros((int(length),), dtype=torch.bool)
        for pocket_id, residues in enumerate(pocket_motifs):
            residues = residues[(residues >= 0) & (residues < int(length))]
            residues = residues[~assigned[residues]]
            if residues.numel() == 0:
                continue
            residue_to_motif[residues] = int(pocket_id)
            assigned[residues] = True
        primary_count = len(pocket_motifs) + int(partition.primary_motif_count)
    else:
        motifs = base_motifs + [item.clone() for item in pocket_motifs]
        motif_type = torch.cat([base_types, pocket_types], dim=0)
        residue_to_motif = partition.residue_to_motif.long().cpu()
        primary_count = int(partition.primary_motif_count)

    member_index = _member_index_from_motifs(motifs)
    links: set[tuple[int, int]] = set()
    if partition.component_links is not None and partition.component_links.numel() > 0:
        offset = len(pocket_motifs) if mode in {"pocket_primary", "primary", "pocket_subgraph"} else 0
        for src, dst in partition.component_links.long().cpu().t().tolist():
            links.add((int(src) + offset, int(dst) + offset))

    ca_pos = torch.nan_to_num(ca_pos.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    ca_dist = torch.cdist(ca_pos, ca_pos, p=2.0) if int(length) > 0 else torch.zeros((0, 0))
    pocket_offset = 0 if mode in {"pocket_primary", "primary", "pocket_subgraph"} else len(base_motifs)
    base_offset = len(pocket_motifs) if mode in {"pocket_primary", "primary", "pocket_subgraph"} else 0
    for pocket_i, pocket_res in enumerate(pocket_motifs):
        pocket_id = pocket_offset + pocket_i
        pset = set(int(item) for item in pocket_res.tolist())
        for base_i, base_res in enumerate(base_motifs[: int(partition.primary_motif_count)]):
            base_id = base_offset + base_i
            bset = set(int(item) for item in base_res.tolist())
            overlap = bool(pset.intersection(bset))
            close = False
            if pocket_res.numel() > 0 and base_res.numel() > 0 and ca_dist.numel() > 0:
                close = float(ca_dist[pocket_res.long()][:, base_res.long()].min().item()) <= float(external_pocket_link_radius)
            if overlap or close:
                links.add((base_id, pocket_id))
                links.add((pocket_id, base_id))
    component_links = (
        torch.tensor(sorted(links), dtype=torch.long).t().contiguous()
        if links
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return MotifPartitionResult(
        residue_to_motif=residue_to_motif.long(),
        motifs=motifs,
        motif_type=motif_type.long(),
        motif_member_index=member_index.long(),
        primary_motif_count=primary_count,
        residue_to_micro_motif=partition.residue_to_micro_motif,
        micro_to_component=partition.micro_to_component,
        component_links=component_links.long(),
    )


def dssp8_index(node_feat: torch.Tensor) -> torch.Tensor:
    length = int(node_feat.shape[0])
    if length <= 0:
        return torch.zeros((0,), dtype=torch.long)
    if node_feat.shape[-1] < 1033:
        return torch.full((length,), 7, dtype=torch.long)
    ss_onehot = torch.nan_to_num(node_feat[:, 1025:1033].float(), nan=0.0, posinf=0.0, neginf=0.0)
    ss_idx = torch.argmax(ss_onehot, dim=-1).long()
    missing = ss_onehot.sum(dim=-1) <= 0.0
    ss_idx[missing] = 7
    return ss_idx.cpu()


def rsa3_bin(node_feat: torch.Tensor, *, low: float = 0.10, high: float = 0.30) -> torch.Tensor:
    length = int(node_feat.shape[0])
    if length <= 0:
        return torch.zeros((0,), dtype=torch.long)
    if node_feat.shape[-1] <= 1024:
        return torch.ones((length,), dtype=torch.long)
    rsa = torch.nan_to_num(node_feat[:, 1024].float(), nan=0.0, posinf=1.0, neginf=0.0).cpu()
    bins = torch.zeros((length,), dtype=torch.long)
    bins[rsa >= float(low)] = 1
    bins[rsa >= float(high)] = 2
    return bins


def _contiguous_motifs(keys: torch.Tensor, type_values: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    length = int(keys.shape[0])
    if length <= 0:
        return torch.zeros((0,), dtype=torch.long), [], torch.zeros((0,), dtype=torch.long)

    assignment = torch.zeros((length,), dtype=torch.long)
    motifs: list[torch.Tensor] = []
    motif_types: list[int] = []
    cur_key = int(keys[0])
    cur_start = 0
    motif_id = 0
    for idx in range(1, length + 1):
        should_close = idx == length or int(keys[idx]) != cur_key
        if not should_close:
            continue
        assignment[cur_start:idx] = motif_id
        motifs.append(torch.arange(cur_start, idx, dtype=torch.long))
        motif_types.append(int(type_values[cur_start]))
        motif_id += 1
        if idx < length:
            cur_key = int(keys[idx])
            cur_start = idx
    return assignment, motifs, torch.tensor(motif_types, dtype=torch.long)


def _member_index_from_motifs(motifs: Sequence[torch.Tensor]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for motif_id, residues in enumerate(motifs):
        residues = residues.long().view(-1).cpu()
        if residues.numel() == 0:
            continue
        rows.append(residues)
        cols.append(torch.full_like(residues, int(motif_id)))
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.stack([torch.cat(rows, dim=0), torch.cat(cols, dim=0)], dim=0).long()


def _min_contact_distance(ca_dist: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> tuple[float, int]:
    if src.numel() == 0 or dst.numel() == 0:
        return float("inf"), 0
    sub = ca_dist[src.long()][:, dst.long()]
    min_dist = float(sub.min().item()) if sub.numel() else float("inf")
    contact_count = int((sub <= 10.0).sum().item()) if sub.numel() else 0
    return min_dist, contact_count


def _connected_components(primary_motifs: Sequence[torch.Tensor], ca_pos: torch.Tensor, radius: float) -> list[list[int]]:
    count = len(primary_motifs)
    if count <= 0:
        return []
    ca_dist = torch.cdist(ca_pos.float().cpu(), ca_pos.float().cpu(), p=2.0)
    neighbors: list[set[int]] = [set() for _ in range(count)]
    for src in range(count):
        for dst in range(src + 1, count):
            min_dist, _ = _min_contact_distance(ca_dist, primary_motifs[src], primary_motifs[dst])
            if min_dist <= float(radius):
                neighbors[src].add(dst)
                neighbors[dst].add(src)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(count):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in sorted(neighbors[cur]):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        components.append(sorted(component))
    return components


class MotifPartitioner:
    def __init__(
        self,
        *,
        partition_method: str = "dssp8_exact",
        rsa_low: float = 0.10,
        rsa_high: float = 0.30,
        boundary_window: int = 2,
        component_contact_radius: float = 10.0,
        merge_short_motifs: bool = False,
        min_motif_length: int = 2,
    ) -> None:
        self.partition_method = str(partition_method).lower()
        self.rsa_low = float(rsa_low)
        self.rsa_high = float(rsa_high)
        self.boundary_window = int(boundary_window)
        self.component_contact_radius = float(component_contact_radius)
        self.merge_short_motifs = bool(merge_short_motifs)
        self.min_motif_length = int(min_motif_length)

    def __call__(self, node_feat: torch.Tensor, ca_pos: torch.Tensor) -> MotifPartitionResult:
        dssp = dssp8_index(node_feat)
        rsa = rsa3_bin(node_feat, low=self.rsa_low, high=self.rsa_high)
        length = int(dssp.shape[0])
        use_rsa = "rsa3" in self.partition_method or "asa3" in self.partition_method
        use_boundary = "boundary" in self.partition_method
        use_component = "component" in self.partition_method or "pocket" in self.partition_method

        if use_rsa:
            keys = dssp * MOTIF_RSA_BIN_COUNT + rsa
            type_values = MOTIF_TYPE_DSSP8_RSA3_BASE + keys
        else:
            keys = dssp
            type_values = MOTIF_TYPE_DSSP8_BASE + dssp

        assignment, motifs, motif_type = _contiguous_motifs(keys, type_values)
        if self.merge_short_motifs:
            assignment, motifs, motif_type = self._merge_short_segments(assignment, motifs, motif_type, ca_pos)

        primary_motif_count = len(motifs)
        all_motifs = [item.clone() for item in motifs]
        all_types = [int(item) for item in motif_type.tolist()]
        links: list[tuple[int, int]] = []

        if use_boundary and length > 1 and primary_motif_count > 0:
            for idx in range(1, length):
                if int(dssp[idx]) == int(dssp[idx - 1]):
                    continue
                start = max(0, idx - max(1, self.boundary_window))
                end = min(length, idx + max(1, self.boundary_window))
                residues = torch.arange(start, end, dtype=torch.long)
                if residues.numel() == 0:
                    continue
                motif_id = len(all_motifs)
                all_motifs.append(residues)
                all_types.append(MOTIF_TYPE_BOUNDARY)
                left = int(assignment[idx - 1])
                right = int(assignment[idx])
                for primary_id in {left, right}:
                    links.extend([(primary_id, motif_id), (motif_id, primary_id)])

        micro_to_component: torch.Tensor | None = None
        if use_component and primary_motif_count > 0:
            components = _connected_components(motifs, ca_pos, self.component_contact_radius)
            micro_to_component = torch.full((primary_motif_count,), -1, dtype=torch.long)
            for component in components:
                residues = torch.cat([motifs[item].long() for item in component], dim=0).unique(sorted=True)
                motif_id = len(all_motifs)
                all_motifs.append(residues)
                all_types.append(MOTIF_TYPE_COMPONENT)
                for micro_id in component:
                    micro_to_component[int(micro_id)] = motif_id
                    links.extend([(int(micro_id), motif_id), (motif_id, int(micro_id))])

        member_index = _member_index_from_motifs(all_motifs)
        component_links = (
            torch.tensor(sorted(set(links)), dtype=torch.long).t().contiguous()
            if links
            else torch.zeros((2, 0), dtype=torch.long)
        )
        return MotifPartitionResult(
            residue_to_motif=assignment.long(),
            motifs=all_motifs,
            motif_type=torch.tensor(all_types, dtype=torch.long),
            motif_member_index=member_index.long(),
            primary_motif_count=primary_motif_count,
            residue_to_micro_motif=assignment.long(),
            micro_to_component=micro_to_component,
            component_links=component_links.long(),
        )

    def _merge_short_segments(
        self,
        assignment: torch.Tensor,
        motifs: list[torch.Tensor],
        motif_type: torch.Tensor,
        ca_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if len(motifs) <= 1:
            return assignment, motifs, motif_type
        target = list(range(len(motifs)))
        ca_dist = torch.cdist(ca_pos.float().cpu(), ca_pos.float().cpu(), p=2.0)
        for motif_id, residues in enumerate(motifs):
            if int(residues.numel()) > self.min_motif_length:
                continue
            candidates = []
            for other_id, other_residues in enumerate(motifs):
                if other_id == motif_id:
                    continue
                seq_gap = min(abs(int(residues[0]) - int(other_residues[-1])), abs(int(residues[-1]) - int(other_residues[0])))
                min_dist, _ = _min_contact_distance(ca_dist, residues, other_residues)
                candidates.append((seq_gap, min_dist, other_id))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                target[motif_id] = int(candidates[0][2])

        remap: dict[int, int] = {}
        merged: list[list[torch.Tensor]] = []
        merged_types: list[int] = []
        for old_id in range(len(motifs)):
            root = target[old_id]
            if root == old_id or int(motifs[root].numel()) <= self.min_motif_length:
                root = old_id
            if root not in remap:
                remap[root] = len(merged)
                merged.append([])
                merged_types.append(int(motif_type[root]))
            remap[old_id] = remap[root]
            merged[remap[root]].append(motifs[old_id])

        new_assignment = assignment.clone()
        new_motifs: list[torch.Tensor] = []
        for new_id, chunks in enumerate(merged):
            residues = torch.cat(chunks, dim=0).unique(sorted=True)
            new_motifs.append(residues)
            new_assignment[residues] = new_id
        return new_assignment.long(), new_motifs, torch.tensor(merged_types, dtype=torch.long)


class MotifGraphBuilder:
    def __init__(
        self,
        *,
        edge_mode: str = "contact_aware",
        spatial_radius: float = 12.0,
        contact_radius: float = 10.0,
        top_k: int = 4,
    ) -> None:
        self.edge_mode = str(edge_mode).lower()
        self.spatial_radius = float(spatial_radius)
        self.contact_radius = float(contact_radius)
        self.top_k = int(top_k)

    def build(self, partition: MotifPartitionResult, ca_pos: torch.Tensor) -> MotifGraphResult:
        motifs = partition.motifs
        motif_type = partition.motif_type.long()
        num_motifs = len(motifs)
        if num_motifs <= 0:
            return MotifGraphResult(
                motif_length=torch.zeros((0, 1), dtype=torch.float32),
                motif_centroid=torch.zeros((0, 3), dtype=torch.float32),
                motif_edge_index=torch.zeros((2, 0), dtype=torch.long),
                motif_edge_attr=torch.zeros((0, MOTIF_EDGE_ATTR_DIM), dtype=torch.float32),
                motif_rel_edge_index=torch.zeros((2, 0), dtype=torch.long),
                motif_rel_edge_type=torch.zeros((0,), dtype=torch.long),
                motif_rel_edge_attr=torch.zeros((0, MOTIF_EDGE_ATTR_DIM), dtype=torch.float32),
            )

        ca_pos = torch.nan_to_num(ca_pos.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        length = max(int(ca_pos.shape[0]), 1)
        centroids = []
        lengths = []
        for residues in motifs:
            residues = residues.long()
            lengths.append([float(residues.numel()) / float(length)])
            centroids.append(ca_pos[residues].mean(dim=0, keepdim=True) if residues.numel() else ca_pos.new_zeros((1, 3)))
        motif_centroid = torch.cat(centroids, dim=0).float()
        motif_length = torch.tensor(lengths, dtype=torch.float32)

        if num_motifs <= 1:
            empty_edge = torch.zeros((2, 0), dtype=torch.long)
            empty_attr = torch.zeros((0, MOTIF_EDGE_ATTR_DIM), dtype=torch.float32)
            return MotifGraphResult(motif_length, motif_centroid, empty_edge, empty_attr, empty_edge, torch.zeros((0,), dtype=torch.long), empty_attr)

        mode = self.edge_mode
        use_seq = mode in {"full", "hybrid", "contact_aware", "seq_centroid", "seq_adjacent", "micro_component"}
        use_spatial = mode in {"full", "hybrid", "contact_aware", "seq_centroid", "centroid_radius", "micro_component"}
        use_topk = mode in {"full", "hybrid", "contact_aware", "seq_centroid", "centroid_knn", "micro_component"}
        use_contact = mode in {"full", "hybrid", "contact_aware", "min_residue_contact", "contact_count", "micro_component"}

        centroid_dist = torch.cdist(motif_centroid, motif_centroid, p=2.0)
        ca_dist = torch.cdist(ca_pos, ca_pos, p=2.0)
        flags: dict[tuple[int, int], set[str]] = {}

        def add(src: int, dst: int, flag: str) -> None:
            if src == dst:
                return
            flags.setdefault((int(src), int(dst)), set()).add(flag)

        if use_seq:
            for src in range(max(0, partition.primary_motif_count - 1)):
                add(src, src + 1, "seq")
                add(src + 1, src, "seq")

        if use_spatial:
            mask = (centroid_dist <= self.spatial_radius) & ~torch.eye(num_motifs, dtype=torch.bool)
            src_idx, dst_idx = torch.where(mask)
            for src, dst in zip(src_idx.tolist(), dst_idx.tolist()):
                add(src, dst, "spatial")

        if use_topk and self.top_k > 0:
            k = min(max(1, self.top_k), num_motifs - 1)
            masked = centroid_dist.clone()
            masked.fill_diagonal_(float("inf"))
            nearest = torch.topk(masked, k=k, largest=False, dim=-1).indices
            for src in range(num_motifs):
                for dst in nearest[src].tolist():
                    add(src, int(dst), "topk")

        pair_stats: dict[tuple[int, int], tuple[float, int, int]] = {}
        for src in range(num_motifs):
            src_set = set(int(item) for item in motifs[src].tolist())
            for dst in range(num_motifs):
                if src == dst:
                    continue
                dst_set = set(int(item) for item in motifs[dst].tolist())
                overlap = len(src_set.intersection(dst_set))
                min_dist, contact_count = _min_contact_distance(ca_dist, motifs[src], motifs[dst])
                pair_stats[(src, dst)] = (min_dist, contact_count, overlap)
                if use_contact and (min_dist <= self.contact_radius or contact_count > 0):
                    add(src, dst, "contact")
                if overlap > 0:
                    add(src, dst, "overlap")

        if partition.component_links is not None and partition.component_links.numel() > 0:
            for src, dst in partition.component_links.long().t().tolist():
                add(src, dst, "component")

        rows: list[tuple[int, int]] = []
        rel_types: list[int] = []
        attrs: list[torch.Tensor] = []
        for src, dst in sorted(flags):
            flag_set = flags[(src, dst)]
            min_dist, contact_count, overlap = pair_stats.get((src, dst), (float(centroid_dist[src, dst]), 0, 0))
            rel_type = self._relation_type(flag_set)
            rows.append((src, dst))
            rel_types.append(rel_type)
            attrs.append(
                self._edge_attr(
                    src,
                    dst,
                    motif_type,
                    centroid_dist=float(centroid_dist[src, dst]),
                    min_dist=float(min_dist),
                    contact_count=int(contact_count),
                    overlap=int(overlap),
                    flags=flag_set,
                    rel_type=rel_type,
                )
            )

        if not rows:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, MOTIF_EDGE_ATTR_DIM), dtype=torch.float32)
            rel_type_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(attrs, dim=0).float()
            rel_type_tensor = torch.tensor(rel_types, dtype=torch.long)
        return MotifGraphResult(motif_length, motif_centroid, edge_index, edge_attr, edge_index, rel_type_tensor, edge_attr)

    def _relation_type(self, flags: set[str]) -> int:
        if "component" in flags:
            return MOTIF_REL_COMPONENT
        if "contact" in flags or "overlap" in flags:
            return MOTIF_REL_CONTACT
        if "topk" in flags:
            return MOTIF_REL_TOPK
        if "spatial" in flags:
            return MOTIF_REL_SPATIAL
        return MOTIF_REL_SEQUENCE

    def _edge_attr(
        self,
        src: int,
        dst: int,
        motif_type: torch.Tensor,
        *,
        centroid_dist: float,
        min_dist: float,
        contact_count: int,
        overlap: int,
        flags: set[str],
        rel_type: int,
    ) -> torch.Tensor:
        src_type = int(motif_type[src].clamp(min=0, max=MOTIF_TYPE_COUNT - 1))
        dst_type = int(motif_type[dst].clamp(min=0, max=MOTIF_TYPE_COUNT - 1))
        values = [
            min(float(centroid_dist) / 20.0, 5.0),
            min(float(min_dist) / 20.0, 5.0) if min_dist != float("inf") else 5.0,
            min(abs(int(dst) - int(src)) / 32.0, 4.0),
            min(float(contact_count) / 16.0, 4.0),
            1.0 if "seq" in flags else 0.0,
            1.0 if "spatial" in flags else 0.0,
            1.0 if "topk" in flags else 0.0,
            1.0 if "contact" in flags else 0.0,
            1.0 if "component" in flags else 0.0,
            min(float(overlap) / 8.0, 4.0),
            float(src_type) / max(float(MOTIF_TYPE_COUNT - 1), 1.0),
            float(dst_type) / max(float(MOTIF_TYPE_COUNT - 1), 1.0),
            1.0 if src_type == dst_type else 0.0,
            float(rel_type) / max(float(MOTIF_REL_TYPE_COUNT - 1), 1.0),
        ]
        return torch.tensor(values, dtype=torch.float32)


def motif_labels(labels: torch.Tensor, motif_member_index: torch.Tensor, num_motifs: int) -> torch.Tensor:
    out = torch.zeros((num_motifs, 1), dtype=torch.float32)
    if num_motifs <= 0 or labels.numel() == 0 or motif_member_index.numel() == 0:
        return out
    labels = labels.float().view(-1).cpu()
    residues, motifs = motif_member_index.long().cpu()
    for motif_id in range(num_motifs):
        idx = residues[motifs == motif_id]
        if idx.numel() > 0:
            out[motif_id, 0] = labels[idx].max()
    return out


def motif_residue_relpos(ca_pos: torch.Tensor, motif_assignment: torch.Tensor, motif_centroid: torch.Tensor) -> torch.Tensor:
    length = int(ca_pos.shape[0])
    relpos = torch.zeros((length, 5), dtype=torch.float32)
    num_motifs = int(motif_centroid.shape[0])
    if length <= 0 or num_motifs <= 0:
        return relpos
    ca_pos = torch.nan_to_num(ca_pos.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    motif_assignment = motif_assignment.long().cpu().clamp(min=0, max=max(num_motifs - 1, 0))
    for motif_id in range(num_motifs):
        idx = torch.where(motif_assignment == motif_id)[0]
        if idx.numel() == 0:
            continue
        order = torch.arange(idx.numel(), dtype=torch.float32)
        denom = max(float(idx.numel() - 1), 1.0)
        vec = ca_pos[idx].float() - motif_centroid[motif_id].float()
        dist = torch.linalg.norm(vec, dim=-1, keepdim=True)
        unit = vec / dist.clamp(min=1e-8)
        relpos[idx, 0] = (order / denom).clamp(min=0.0, max=1.0)
        relpos[idx, 1:2] = (dist / 20.0).clamp(min=0.0, max=2.0)
        relpos[idx, 2:5] = torch.nan_to_num(unit, nan=0.0, posinf=0.0, neginf=0.0)
    return relpos


def local_edge_attr(src: int, dst: int, ca_pos: torch.Tensor, *, sequence_neighbor: bool, self_loop: bool) -> torch.Tensor:
    dist = torch.linalg.norm(ca_pos[int(dst)] - ca_pos[int(src)])
    seq_sep = abs(int(dst) - int(src))
    return torch.tensor(
        [
            float(dist) / 20.0,
            min(float(seq_sep) / 32.0, 2.0),
            1.0 if sequence_neighbor else 0.0,
            1.0 if self_loop else 0.0,
        ],
        dtype=torch.float32,
    )


def build_motif_local_edges(
    ca_pos: torch.Tensor,
    motif_member_index: torch.Tensor,
    *,
    num_motifs: int,
    local_radius: float = 10.0,
    use_sequence_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = int(ca_pos.shape[0])
    if length <= 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, MOTIF_LOCAL_EDGE_ATTR_DIM), dtype=torch.float32)
    ca_pos = torch.nan_to_num(ca_pos.float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    ca_dist = torch.cdist(ca_pos, ca_pos, p=2.0)
    motif_to_residues: list[list[int]] = [[] for _ in range(max(num_motifs, 0))]
    if motif_member_index.numel() > 0:
        for residue_id, motif_id in motif_member_index.long().t().tolist():
            if 0 <= motif_id < num_motifs and 0 <= residue_id < length:
                motif_to_residues[motif_id].append(residue_id)

    flags: dict[tuple[int, int], tuple[bool, bool]] = {}
    for idx in range(length):
        flags[(idx, idx)] = (False, True)

    for residues in motif_to_residues:
        if not residues:
            continue
        idx = torch.tensor(sorted(set(residues)), dtype=torch.long)
        sub = ca_dist[idx][:, idx]
        src_local, dst_local = torch.where(sub <= float(local_radius))
        for src_l, dst_l in zip(src_local.tolist(), dst_local.tolist()):
            src = int(idx[src_l])
            dst = int(idx[dst_l])
            seq_flag, self_flag = flags.get((src, dst), (False, False))
            flags[(src, dst)] = (seq_flag, self_flag or src == dst)
        if use_sequence_edges:
            idx_set = set(idx.tolist())
            for src in idx.tolist():
                dst = int(src) + 1
                if dst not in idx_set:
                    continue
                for a, b in ((src, dst), (dst, src)):
                    _, self_flag = flags.get((a, b), (False, False))
                    flags[(a, b)] = (True, self_flag)

    rows: list[tuple[int, int]] = []
    attrs: list[torch.Tensor] = []
    for src, dst in sorted(flags):
        seq_flag, self_flag = flags[(src, dst)]
        rows.append((src, dst))
        attrs.append(local_edge_attr(src, dst, ca_pos, sequence_neighbor=seq_flag, self_loop=self_flag))
    return torch.tensor(rows, dtype=torch.long).t().contiguous(), torch.stack(attrs, dim=0).float()


def sanitize_cache_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))


def cache_file(cache_dir: Path, name: str) -> Path:
    stem = sanitize_cache_name(name)
    suffix = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{stem}.{suffix}.pt"


def tensor_payload(keys: Iterable[str], payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: payload[key] for key in keys if key in payload}
