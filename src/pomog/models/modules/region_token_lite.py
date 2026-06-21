from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class RegionTokenLiteAdapter(nn.Module):
    """Lightweight detector-region token adapter for residue states."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        pocket_feature_dim: int = 8,
        edge_attr_dim: int = 5,
        dropout: float = 0.1,
        alpha_init: float = 1e-3,
        gate_bias_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pocket_feature_dim = int(pocket_feature_dim)
        self.edge_attr_dim = int(edge_attr_dim)
        self.token_feature = nn.Sequential(
            nn.LayerNorm(self.pocket_feature_dim),
            nn.Linear(self.pocket_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(self.edge_attr_dim),
            nn.Linear(self.edge_attr_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.residue_gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self._init(gate_bias_init)

    def _init(self, gate_bias_init: float) -> None:
        for module in (self.token_feature, self.token_fuse, self.edge_gate, self.residue_gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        final_token = self.token_fuse[-1]
        if isinstance(final_token, nn.Linear):
            nn.init.zeros_(final_token.weight)
            if final_token.bias is not None:
                nn.init.zeros_(final_token.bias)
        final_gate = self.residue_gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

    def _sanitize_pocket_feature(
        self,
        pocket_feature: torch.Tensor | None,
        pocket_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        out = torch.zeros((pocket_count, self.pocket_feature_dim), device=device)
        if pocket_feature is None or pocket_feature.numel() == 0 or pocket_count <= 0:
            return out
        feat = torch.nan_to_num(pocket_feature.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(pocket_count, int(feat.shape[0]))
        take_f = min(self.pocket_feature_dim, int(feat.shape[-1]))
        out[:take_n, :take_f] = feat[:take_n, :take_f]
        return out

    def _sanitize_edge_attr(self, edge_attr: torch.Tensor | None, edge_count: int, device: torch.device) -> torch.Tensor:
        out = torch.zeros((edge_count, self.edge_attr_dim), device=device)
        if edge_attr is None or edge_attr.numel() == 0 or edge_count <= 0:
            return out
        attr = torch.nan_to_num(edge_attr.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(edge_count, int(attr.shape[0]))
        take_f = min(self.edge_attr_dim, int(attr.shape[-1]))
        out[:take_n, :take_f] = attr[:take_n, :take_f]
        return out

    def forward(
        self,
        h: torch.Tensor,
        pocket_feature: torch.Tensor | None,
        residue_to_pocket_index: torch.Tensor | None,
        residue_to_pocket_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_count = int(h.shape[0])
        pocket_count = int(pocket_feature.shape[0]) if pocket_feature is not None and pocket_feature.ndim >= 2 else 0
        if node_count <= 0 or pocket_count <= 0 or residue_to_pocket_index is None or residue_to_pocket_index.numel() == 0:
            zeros = torch.zeros_like(h)
            return h, zeros, zeros
        member = residue_to_pocket_index.long().to(h.device)
        residue = member[0].clamp(min=0, max=max(node_count - 1, 0))
        pocket = member[1].clamp(min=0, max=max(pocket_count - 1, 0))
        pocket_feat = self._sanitize_pocket_feature(pocket_feature, pocket_count, h.device)
        token_seed = self.token_feature(pocket_feat)
        pooled = scatter_mean(h[residue], pocket, dim=0, dim_size=pocket_count)
        token = self.token_fuse(torch.cat([token_seed, pooled], dim=-1))
        edge_attr = self._sanitize_edge_attr(residue_to_pocket_attr, int(residue.numel()), h.device)
        edge_weight = torch.sigmoid(self.edge_gate(edge_attr)).view(-1, 1)
        residue_msg = scatter_mean(token[pocket] * edge_weight, residue, dim=0, dim_size=node_count)
        gate = torch.sigmoid(self.residue_gate(torch.cat([h, residue_msg], dim=-1)))
        updated = h + self.alpha * gate * residue_msg
        return updated, token, gate
