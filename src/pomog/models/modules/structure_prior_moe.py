from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from pomog.data.structure_prior_cache import DETECTOR_NAMES, FEATURE_INDEX, STRUCTURE_PRIOR_FEATURE_DIM


class StructurePriorMoEHead(nn.Module):
    """Residue-level MoE over base, detector-specific, and local-environment experts."""

    def __init__(
        self,
        input_dim: int,
        *,
        prior_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        local_dim: int = 0,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        gate_entropy_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.prior_dim = int(prior_dim)
        self.local_dim = int(local_dim)
        self.temperature = max(0.05, float(temperature))
        self.gate_entropy_weight = float(gate_entropy_weight)
        hidden = int(hidden_dim or max(32, self.input_dim))
        self.base_head = self._head(self.input_dim, hidden, float(dropout))
        self.p2rank_head = self._head(self.input_dim + 6, hidden, float(dropout))
        self.fpocket_head = self._head(self.input_dim + 6, hidden, float(dropout))
        self.castp_head = self._head(self.input_dim + 6, hidden, float(dropout))
        self.caver_head = self._head(self.input_dim + 3, hidden, float(dropout))
        self.lee_head = self._head(self.input_dim + self.local_dim, hidden, float(dropout))
        gate_dim = self.input_dim + self.prior_dim + self.local_dim + 5
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 6),
        )
        self._last_entropy_loss: torch.Tensor | None = None
        self._init()

    @staticmethod
    def _head(dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _init(self) -> None:
        for module in (self.base_head, self.p2rank_head, self.fpocket_head, self.castp_head, self.caver_head, self.lee_head, self.gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def _prior(self, prior_feat: torch.Tensor | None, node_count: int, device: torch.device) -> torch.Tensor:
        out = torch.zeros((node_count, self.prior_dim), device=device)
        if prior_feat is None or prior_feat.numel() == 0:
            return out
        feat = torch.nan_to_num(prior_feat.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(node_count, int(feat.shape[0]))
        take_f = min(self.prior_dim, int(feat.shape[-1]))
        out[:take_n, :take_f] = feat[:take_n, :take_f]
        return out

    def _local(self, local_env: torch.Tensor | None, node_count: int, device: torch.device) -> torch.Tensor:
        if self.local_dim <= 0:
            return torch.zeros((node_count, 0), device=device)
        out = torch.zeros((node_count, self.local_dim), device=device)
        if local_env is None or local_env.numel() == 0:
            return out
        local = torch.nan_to_num(local_env.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(node_count, int(local.shape[0]))
        take_f = min(self.local_dim, int(local.shape[-1]))
        out[:take_n, :take_f] = local[:take_n, :take_f]
        return out

    @staticmethod
    def _cols(prior: torch.Tensor, names: list[str]) -> torch.Tensor:
        cols = [FEATURE_INDEX[name] for name in names if FEATURE_INDEX[name] < int(prior.shape[-1])]
        if not cols:
            return prior.new_zeros((int(prior.shape[0]), 0))
        return prior[:, cols]

    def forward(
        self,
        encoded: torch.Tensor,
        prior_feat: torch.Tensor | None,
        local_env: torch.Tensor | None = None,
        detector_missing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_count = int(encoded.shape[0])
        prior = self._prior(prior_feat, node_count, encoded.device)
        local = self._local(local_env, node_count, encoded.device)
        coverage_stats = torch.stack(
            [
                (prior[:, FEATURE_INDEX["p2rank_in_top5"]] > 0.5).float(),
                (prior[:, FEATURE_INDEX["fpocket_in_top5"]] > 0.5).float(),
                (prior[:, FEATURE_INDEX["detector_count"]] / float(max(len(DETECTOR_NAMES), 1))).clamp(0.0, 1.0),
                prior[:, FEATURE_INDEX["rsa"]].clamp(0.0, 1.0),
                prior[:, FEATURE_INDEX["local_contact_density"]].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        gate_input = torch.cat([encoded, prior, local, coverage_stats], dim=-1)
        gate_logits = self.gate(gate_input) / self.temperature
        if detector_missing is not None and detector_missing.numel() > 0:
            missing = detector_missing.bool().to(encoded.device)
            if missing.ndim == 2 and missing.shape[0] != node_count:
                missing = missing[:1].expand(node_count, -1)
            if missing.ndim == 1:
                missing = missing.view(1, -1).expand(node_count, -1)
            if missing.shape[-1] >= 4:
                # Expert order: base, p2rank, fpocket, castp, caver, lee.
                expert_missing = torch.stack(
                    [
                        torch.zeros((node_count,), device=encoded.device, dtype=torch.bool),
                        missing[:, 0],
                        missing[:, 1],
                        missing[:, 2],
                        missing[:, 3],
                        torch.zeros((node_count,), device=encoded.device, dtype=torch.bool),
                    ],
                    dim=-1,
                )
                gate_logits = gate_logits.masked_fill(expert_missing, -1e4)
        weights = F.softmax(gate_logits, dim=-1)
        experts = torch.stack(
            [
                self.base_head(encoded),
                self.p2rank_head(torch.cat([encoded, self._cols(prior, [
                    "p2rank_in_top1",
                    "p2rank_in_top3",
                    "p2rank_in_top5",
                    "p2rank_score_best",
                    "p2rank_rank_recip",
                    "dist_to_nearest_p2rank",
                ])], dim=-1)),
                self.fpocket_head(torch.cat([encoded, self._cols(prior, [
                    "fpocket_in_top1",
                    "fpocket_in_top3",
                    "fpocket_in_top5",
                    "fpocket_score_best",
                    "fpocket_rank_recip",
                    "dist_to_nearest_fpocket",
                ])], dim=-1)),
                self.castp_head(torch.cat([encoded, self._cols(prior, [
                    "castp_surface_pocket",
                    "castp_interior_cavity",
                    "castp_channel",
                    "dist_to_castp_feature",
                    "castp_volume_norm",
                    "castp_area_norm",
                ])], dim=-1)),
                self.caver_head(torch.cat([encoded, self._cols(prior, [
                    "caver_tunnel_lining",
                    "dist_to_caver_centerline",
                    "caver_bottleneck_radius",
                ])], dim=-1)),
                self.lee_head(torch.cat([encoded, local], dim=-1)),
            ],
            dim=1,
        )
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        self._last_entropy_loss = -self.gate_entropy_weight * entropy if self.gate_entropy_weight else None
        return (weights.unsqueeze(-1) * experts).sum(dim=1), weights
