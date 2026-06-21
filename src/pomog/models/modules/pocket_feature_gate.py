from __future__ import annotations

import torch
import torch.nn as nn


class PocketFeatureGate(nn.Module):
    """Late gated residual adapter for noisy residue-level pocket priors."""

    def __init__(
        self,
        pocket_feat_dim: int,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        gate_bias_init: float = -2.0,
        alpha_init: float = 0.0,
        channel_wise: bool = True,
    ) -> None:
        super().__init__()
        self.pocket_feat_dim = int(pocket_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.channel_wise = bool(channel_wise)
        adapter_dim = max(1, int(adapter_dim))
        gate_out_dim = self.hidden_dim if self.channel_wise else 1

        self.pocket_encoder = nn.Sequential(
            nn.LayerNorm(self.pocket_feat_dim),
            nn.Linear(self.pocket_feat_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(adapter_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(adapter_dim, self.hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, adapter_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(adapter_dim, gate_out_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self._reset_parameters(float(gate_bias_init))

    def _reset_parameters(self, gate_bias_init: float) -> None:
        for module in (self.pocket_encoder, self.delta, self.gate):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        delta_final = self.delta[-1]
        if isinstance(delta_final, nn.Linear):
            nn.init.zeros_(delta_final.weight)
            if delta_final.bias is not None:
                nn.init.zeros_(delta_final.bias)
        gate_final = self.gate[-1]
        if isinstance(gate_final, nn.Linear) and gate_final.bias is not None:
            nn.init.constant_(gate_final.bias, gate_bias_init)

    def forward(self, h: torch.Tensor, pocket_feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.numel() == 0:
            empty_gate = h.new_zeros((int(h.shape[0]), self.hidden_dim if self.channel_wise else 1))
            return h, {"gate": empty_gate, "alpha": self.alpha.detach()}

        feat = torch.nan_to_num(pocket_feat.float().to(h.device), nan=0.0, posinf=1.0, neginf=0.0)
        if feat.shape[0] != h.shape[0] or feat.shape[-1] != self.pocket_feat_dim:
            fixed = h.new_zeros((int(h.shape[0]), self.pocket_feat_dim))
            take_n = min(int(h.shape[0]), int(feat.shape[0]))
            take_f = min(self.pocket_feat_dim, int(feat.shape[-1]))
            if take_n > 0 and take_f > 0:
                fixed[:take_n, :take_f] = feat[:take_n, :take_f]
            feat = fixed

        encoded = self.pocket_encoder(feat)
        gate = torch.sigmoid(self.gate(torch.cat([h.float(), encoded], dim=-1)))
        delta = self.delta(encoded)
        if self.channel_wise:
            out = h + self.alpha * gate * delta
        else:
            out = h + self.alpha * gate.view(-1, 1) * delta
        return out, {"gate": gate.detach(), "alpha": self.alpha.detach()}
