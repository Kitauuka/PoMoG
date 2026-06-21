from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_mean

from pomog.data.structure_prior_cache import STRUCTURE_PRIOR_FEATURE_DIM


class LocalEnvironmentEncoder(nn.Module):
    """Residue-centric local environment encoder over existing sparse residue edges."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        prior_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        aa_vocab: int = 21,
        aa_dim: int = 16,
        geom_dim: int = 8,
        dropout: float = 0.1,
        use_prior: bool = True,
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.prior_dim = int(prior_dim)
        self.use_prior = bool(use_prior)
        self.aa_embedding = nn.Embedding(int(aa_vocab), int(aa_dim))
        pair_dim = self.hidden_dim + int(aa_dim) + int(geom_dim)
        if self.use_prior:
            pair_dim += self.prior_dim
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self._init(gate_bias_init)

    def _init(self, gate_bias_init: float) -> None:
        nn.init.normal_(self.aa_embedding.weight, mean=0.0, std=0.02)
        for module in (self.pair_mlp, self.pool_projection, self.gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        final_pool = self.pool_projection[-1]
        if isinstance(final_pool, nn.Linear):
            nn.init.zeros_(final_pool.weight)
            if final_pool.bias is not None:
                nn.init.zeros_(final_pool.bias)
        final_gate = self.gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

    def _geometry(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        ca = torch.nan_to_num(X[:, 1].float(), nan=0.0, posinf=0.0, neginf=0.0)
        src, dst = edge_index.long()
        rel = ca[src] - ca[dst]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6, max=30.0)
        unit = rel / dist
        rbf_centers = torch.tensor([4.0, 6.0, 8.0, 10.0], device=X.device, dtype=X.dtype).view(1, -1)
        rbf = torch.exp(-((dist - rbf_centers) ** 2) / 8.0)
        return torch.cat([unit.float(), (dist / 30.0).float(), rbf.float()], dim=-1)

    def _prior_for_edges(self, prior_feat: torch.Tensor | None, src: torch.Tensor, node_count: int, device: torch.device) -> torch.Tensor:
        if not self.use_prior:
            return torch.zeros((int(src.numel()), 0), device=device)
        if prior_feat is None or prior_feat.numel() == 0:
            return torch.zeros((int(src.numel()), self.prior_dim), device=device)
        feat = torch.nan_to_num(prior_feat.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        fixed = torch.zeros((node_count, self.prior_dim), device=device, dtype=feat.dtype)
        take_n = min(node_count, int(feat.shape[0]))
        take_f = min(self.prior_dim, int(feat.shape[-1]))
        fixed[:take_n, :take_f] = feat[:take_n, :take_f]
        return fixed[src]

    def forward(
        self,
        h: torch.Tensor,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        seq: torch.Tensor,
        prior_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        if node_count <= 0 or edge_index is None or edge_index.numel() == 0:
            local = torch.zeros_like(h)
            return h, local
        edge_index = edge_index.long().to(h.device)
        src = edge_index[0].clamp(min=0, max=max(node_count - 1, 0))
        dst = edge_index[1].clamp(min=0, max=max(node_count - 1, 0))
        seq = seq.long().to(h.device).clamp(min=0, max=self.aa_embedding.num_embeddings - 1)
        pair_parts = [
            h[src],
            self.aa_embedding(seq[src]),
            self._geometry(X.to(h.device), torch.stack([src, dst], dim=0)),
            self._prior_for_edges(prior_feat, src, node_count, h.device),
        ]
        pair_state = self.pair_mlp(torch.cat([part for part in pair_parts if part.numel() > 0], dim=-1))
        mean_state = scatter_mean(pair_state, dst, dim=0, dim_size=node_count)
        max_state, _ = scatter_max(pair_state, dst, dim=0, dim_size=node_count)
        max_state = torch.where(torch.isfinite(max_state), max_state, torch.zeros_like(max_state))
        local = self.pool_projection(torch.cat([mean_state, max_state], dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([h, local], dim=-1)))
        return h + gate * local, local


class MultiScaleLocalEnvironmentEncoder(nn.Module):
    """Multi-radius LEE variant over the sparse residue graph."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        prior_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        aa_vocab: int = 21,
        aa_dim: int = 16,
        radii: Sequence[float] | None = None,
        max_neighbors: Sequence[int] | None = None,
        dropout: float = 0.1,
        use_prior: bool = True,
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.radii = [float(value) for value in (radii or (6.0, 10.0, 14.0))]
        raw_neighbors = list(max_neighbors or (16, 32, 48))
        if len(raw_neighbors) < len(self.radii):
            raw_neighbors.extend([raw_neighbors[-1] if raw_neighbors else 32] * (len(self.radii) - len(raw_neighbors)))
        self.max_neighbors = [max(1, int(value)) for value in raw_neighbors[: len(self.radii)]]
        self.encoders = nn.ModuleList(
            [
                LocalEnvironmentEncoder(
                    self.hidden_dim,
                    prior_dim=prior_dim,
                    aa_vocab=aa_vocab,
                    aa_dim=aa_dim,
                    dropout=dropout,
                    use_prior=use_prior,
                    gate_bias_init=gate_bias_init,
                )
                for _ in self.radii
            ]
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * len(self.radii)),
            nn.Linear(self.hidden_dim * len(self.radii), self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self._init(gate_bias_init)

    def _init(self, gate_bias_init: float) -> None:
        for module in (self.fuse, self.gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        final_fuse = self.fuse[-1]
        if isinstance(final_fuse, nn.Linear):
            nn.init.zeros_(final_fuse.weight)
            if final_fuse.bias is not None:
                nn.init.zeros_(final_fuse.bias)
        final_gate = self.gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

    @staticmethod
    def _filtered_edges(
        X: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        radius: float,
        max_neighbors: int,
        node_count: int,
    ) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0 or node_count <= 0:
            return torch.zeros((2, 0), dtype=torch.long, device=X.device)
        edge_index = edge_index.long().to(X.device)
        src = edge_index[0].clamp(min=0, max=max(node_count - 1, 0))
        dst = edge_index[1].clamp(min=0, max=max(node_count - 1, 0))
        ca = torch.nan_to_num(X[:, 1].float(), nan=0.0, posinf=0.0, neginf=0.0)
        dist = torch.linalg.norm(ca[src] - ca[dst], dim=-1)
        keep = torch.where(dist <= float(radius))[0]
        if keep.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=X.device)
        kept_src = src[keep]
        kept_dst = dst[keep]
        kept_dist = dist[keep]
        selected: list[torch.Tensor] = []
        for node_id in kept_dst.unique(sorted=True).tolist():
            row = torch.where(kept_dst == int(node_id))[0]
            if row.numel() > int(max_neighbors):
                order = torch.argsort(kept_dist[row])[: int(max_neighbors)]
                row = row[order]
            selected.append(row)
        if not selected:
            return torch.zeros((2, 0), dtype=torch.long, device=X.device)
        idx = torch.cat(selected, dim=0)
        return torch.stack([kept_src[idx], kept_dst[idx]], dim=0).long()

    def forward(
        self,
        h: torch.Tensor,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        seq: torch.Tensor,
        prior_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        if node_count <= 0 or edge_index is None or edge_index.numel() == 0:
            local = torch.zeros_like(h)
            return h, local
        locals_by_scale: list[torch.Tensor] = []
        X_device = X.to(h.device)
        for radius, max_neighbors, encoder in zip(self.radii, self.max_neighbors, self.encoders):
            scale_edges = self._filtered_edges(
                X_device,
                edge_index.to(h.device),
                radius=radius,
                max_neighbors=max_neighbors,
                node_count=node_count,
            )
            _, local = encoder(h, X_device, scale_edges, seq, prior_feat)
            locals_by_scale.append(local)
        fused = self.fuse(torch.cat(locals_by_scale, dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat([h, fused], dim=-1)))
        return h + gate * fused, fused
