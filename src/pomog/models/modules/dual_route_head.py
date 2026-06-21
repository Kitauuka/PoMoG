from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pomog.data.structure_prior_cache import STRUCTURE_PRIOR_FEATURE_DIM


@dataclass
class DualRouteOutput:
    final_logit: torch.Tensor
    main_logit: torch.Tensor
    pocket_logit: torch.Tensor
    rescue_logit: torch.Tensor
    route_gate: torch.Tensor


class DualRoutePocketRescueHead(nn.Module):
    """Pocket route plus out-of-pocket rescue route for residue logits."""

    def __init__(
        self,
        input_dim: int,
        *,
        prior_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        local_dim: int = 0,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        gate_bias_init: float = 0.0,
        combination: str = "mixture",
        main_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.prior_dim = int(prior_dim)
        self.local_dim = int(local_dim)
        self.combination = str(combination or "mixture").lower()
        self.main_scale = float(main_scale)
        hidden = int(hidden_dim or max(32, self.input_dim))
        total_dim = self.input_dim + self.prior_dim + self.local_dim
        self.main_head = self._head(total_dim, hidden, float(dropout))
        self.pocket_head = self._head(total_dim, hidden, float(dropout))
        self.rescue_head = self._head(total_dim, hidden, float(dropout))
        self.route_gate = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, 1),
        )
        self._init(gate_bias_init)

    @staticmethod
    def _head(total_dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _init(self, gate_bias_init: float) -> None:
        for module in (self.main_head, self.pocket_head, self.rescue_head, self.route_gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        final_gate = self.route_gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_bias_init))

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
        feat = torch.nan_to_num(local_env.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(node_count, int(feat.shape[0]))
        take_f = min(self.local_dim, int(feat.shape[-1]))
        out[:take_n, :take_f] = feat[:take_n, :take_f]
        return out

    def forward(
        self,
        encoded: torch.Tensor,
        prior_feat: torch.Tensor | None,
        local_env: torch.Tensor | None = None,
    ) -> DualRouteOutput:
        node_count = int(encoded.shape[0])
        basis = torch.cat(
            [
                encoded,
                self._prior(prior_feat, node_count, encoded.device),
                self._local(local_env, node_count, encoded.device),
            ],
            dim=-1,
        )
        main = self.main_head(basis)
        pocket = self.pocket_head(basis)
        rescue = self.rescue_head(basis)
        gate = torch.sigmoid(self.route_gate(basis))
        if self.combination in {"additive_delta", "additive", "main_delta"}:
            final = main + gate * pocket + (1.0 - gate) * rescue
        else:
            final = gate * pocket + (1.0 - gate) * rescue + self.main_scale * main
        return DualRouteOutput(
            final_logit=final,
            main_logit=main,
            pocket_logit=pocket,
            rescue_logit=rescue,
            route_gate=gate,
        )
