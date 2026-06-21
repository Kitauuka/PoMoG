from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from pomog.data.structure_prior_cache import DETECTOR_FEATURE_GROUPS, STRUCTURE_PRIOR_FEATURE_DIM


class StructurePriorGatedAdapter(nn.Module):
    """Zero-initialized gated residual adapter for residue-level structure priors."""

    def __init__(
        self,
        hidden_dim: int,
        prior_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        *,
        dropout: float = 0.1,
        alpha_init: float = 1e-3,
        gate_bias_init: float = -2.0,
        detector_dropout: float = 0.0,
        enabled_feature_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.prior_dim = int(prior_dim)
        self.detector_dropout = float(detector_dropout)
        self.enabled_feature_indices = [int(idx) for idx in enabled_feature_indices] if enabled_feature_indices else None
        self.prior_norm = nn.LayerNorm(self.prior_dim)
        self.prior_mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.hidden_dim),
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self._init(gate_bias_init)

    def _init(self, gate_bias_init: float) -> None:
        for module in (self.prior_mlp, self.gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        final_prior = self.prior_mlp[-1]
        if isinstance(final_prior, nn.Linear):
            nn.init.zeros_(final_prior.weight)
            if final_prior.bias is not None:
                nn.init.zeros_(final_prior.bias)
        final_gate = self.gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

    def _sanitize(self, prior_feat: torch.Tensor | None, prior_mask: torch.Tensor | None, node_count: int, device: torch.device) -> torch.Tensor:
        if prior_feat is None or prior_feat.numel() == 0:
            return torch.zeros((node_count, self.prior_dim), device=device)
        feat = torch.nan_to_num(prior_feat.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.zeros((node_count, self.prior_dim), device=device, dtype=feat.dtype)
        take_n = min(node_count, int(feat.shape[0]))
        take_f = min(self.prior_dim, int(feat.shape[-1]))
        out[:take_n, :take_f] = feat[:take_n, :take_f]
        if prior_mask is not None and prior_mask.numel() > 0:
            mask = torch.zeros_like(out, dtype=torch.bool)
            raw = prior_mask.bool().to(device)
            mask[:take_n, :take_f] = raw[:take_n, :take_f]
            out = out * mask.float()
        if self.enabled_feature_indices is not None:
            keep = torch.zeros((self.prior_dim,), device=device, dtype=torch.bool)
            keep[self.enabled_feature_indices] = True
            out = out * keep.view(1, -1).float()
        if self.training and self.detector_dropout > 0.0:
            out = self._apply_detector_dropout(out)
        return out

    def _apply_detector_dropout(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.numel() == 0:
            return feat
        keep = torch.ones((self.prior_dim,), device=feat.device, dtype=feat.dtype)
        for indices in DETECTOR_FEATURE_GROUPS.values():
            if torch.rand((), device=feat.device) < self.detector_dropout:
                keep[torch.tensor(indices, device=feat.device, dtype=torch.long)] = 0.0
        return feat * keep.view(1, -1)

    def forward(
        self,
        h: torch.Tensor,
        prior_feat: torch.Tensor | None,
        prior_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior = self._sanitize(prior_feat, prior_mask, int(h.shape[0]), h.device)
        prior_state = self.prior_mlp(self.prior_norm(prior))
        gate = torch.sigmoid(self.gate(torch.cat([h, prior_state], dim=-1)))
        updated = h + self.alpha * gate * prior_state
        return updated, prior_state, gate

