from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_max, scatter_mean

from pomog.data.graph import get_geo_feat
from pomog.data.pocket_dataset import (
    POMOG_POCKET_GRAPH_EDGE_DIM,
    POMOG_POCKET_NODE_RAW_DIM,
    POMOG_POCKET_RESIDUE_FEAT_DIM,
)
from pomog.data.structure_prior_cache import STRUCTURE_PRIOR_FEATURE_DIM
from pomog.models.layers import Graph_encoder
from pomog.models.modules.dual_route_head import DualRoutePocketRescueHead
from pomog.models.modules.local_environment_encoder import LocalEnvironmentEncoder, MultiScaleLocalEnvironmentEncoder
from pomog.models.modules.pocket_feature_gate import PocketFeatureGate
from pomog.models.modules.region_token_lite import RegionTokenLiteAdapter
from pomog.models.modules.structure_prior_adapter import StructurePriorGatedAdapter
from pomog.models.modules.structure_prior_moe import StructurePriorMoEHead
from pomog.models.residue_encoder import PoMoG as _BasePoMoG
from pomog.models.residue_encoder import _build_vector_geometry

_BASELINE_GEO_NODE_DIM = 184


class PoMoG(_BasePoMoG):
    """PoMoG pocket-guided branch on top of the PoMoG/m01 residue encoder."""

    def __init__(
        self,
        *args,
        projection_dim: int = 128,
        residue_pocket_feature_dim: int = POMOG_POCKET_RESIDUE_FEAT_DIM,
        context_node_raw_dim: int = POMOG_POCKET_NODE_RAW_DIM,
        context_graph_edge_dim: int = POMOG_POCKET_GRAPH_EDGE_DIM,
        pomog_context_layers: int = 1,
        pomog_context_heads: int = 4,
        pomog_use_context_graph: bool = True,
        pomog_use_residue_features: bool = True,
        pomog_context_residual_scale: float = 0.25,
        pomog_gate_init: float = -4.0,
        pomog_aux_hidden_mult: float = 1.0,
        pomog_encoder: str = "encoder",
        pomog_writeback_enabled: bool = True,
        pomog_detach_aux_input: bool = False,
        pomog_logit_prior_enabled: bool = False,
        pomog_logit_prior_gate_init: float = -4.0,
        pomog_logit_prior_scale: float = 1.0,
        pomog_classifier_fusion: str = "legacy",
        pomog_classifier_use_residue_features: bool = True,
        pomog_pocket_bias_enabled: bool = False,
        pomog_pocket_bias_scale_init: float = 0.0,
        pomog_pocket_bias_dropout: float | None = None,
        pomog_reliability_moe_enabled: bool = False,
        pomog_moe_gate_temperature: float = 1.0,
        use_pocket_softfeature_gate: bool = False,
        pocket_feature_dim: int | None = None,
        pocket_gate_adapter_dim: int = 64,
        pocket_gate_dropout: float = 0.1,
        pocket_feature_dropout: float = 0.0,
        pocket_gate_bias_init: float = -2.0,
        pocket_gate_alpha_init: float = 0.0,
        pocket_gate_channel_wise: bool = True,
        structure_prior_feature_dim: int = STRUCTURE_PRIOR_FEATURE_DIM,
        pomog_structure_prior_gate_enabled: bool = False,
        pomog_structure_prior_gate_alpha_init: float = 1e-3,
        pomog_structure_prior_gate_bias_init: float = -2.0,
        pomog_structure_prior_detector_dropout: float = 0.0,
        pomog_structure_prior_shuffle_enabled: bool = False,
        pomog_structure_prior_shuffle_shift: int = 17,
        pomog_lee_enabled: bool = False,
        pomog_lee_use_prior: bool = True,
        pomog_lee_gate_bias_init: float = -3.0,
        pomog_lee_multiscale_enabled: bool = False,
        pomog_lee_multiscale_radii: Sequence[float] | None = None,
        pomog_lee_multiscale_max_neighbors: Sequence[int] | None = None,
        pomog_region_token_lite_enabled: bool = False,
        pomog_region_token_lite_alpha_init: float = 1e-3,
        pomog_region_token_lite_gate_bias_init: float = -3.0,
        pomog_dual_route_enabled: bool = False,
        pomog_dual_route_gate_bias_init: float = 0.0,
        pomog_dual_route_combination: str = "mixture",
        pomog_dual_route_main_scale: float = 0.1,
        pomog_structure_prior_moe_v2_enabled: bool = False,
        pomog_structure_prior_moe_temperature: float = 1.0,
        pomog_structure_prior_moe_gate_entropy_weight: float = 0.0,
        pomog_neg_suppress_enabled: bool = False,
        pomog_neg_suppress_gate_bias_init: float = -2.0,
        pomog_dual_delta_enabled: bool = False,
        pomog_dual_delta_neg_gate_bias_init: float = -2.0,
        pomog_dual_delta_rescue_gate_bias_init: float = -3.0,
        pomog_prior_logit_adapter_enabled: bool = False,
        pomog_prior_logit_adapter_alpha_init: float = 0.0,
        pomog_prior_logit_adapter_dropout: float = 0.1,
        pomog_prior_logit_adapter_feature_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pomog_encoder = str(pomog_encoder or "encoder").lower()
        if self.pomog_encoder in {"pomog_baseline", "baseline", "graphec_as"}:
            node_input_dim = int(kwargs.get("node_input_dim", 1024 + 9))
            edge_input_dim = int(kwargs.get("edge_input_dim", 450))
            hidden_dim_override = int(kwargs.get("hidden_dim", 128))
            num_layers = int(kwargs.get("num_layers", 4))
            dropout_override = float(kwargs.get("dropout", 0.2))
            self.encoder = Graph_encoder(
                node_in_dim=node_input_dim + _BASELINE_GEO_NODE_DIM,
                edge_in_dim=edge_input_dim,
                hidden_dim=hidden_dim_override,
                seq_in=False,
                num_layers=num_layers,
                drop_rate=dropout_override,
            )
            for param in self.encoder.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        hidden_dim = int(self.fc1.out_features)
        head_dim = int(self.fc1.in_features)
        dropout = float(kwargs.get("dropout", 0.2))
        heads = int(pomog_context_heads)
        if hidden_dim % heads != 0:
            heads = 1

        self.residue_pocket_feature_dim = int(residue_pocket_feature_dim)
        self.context_node_raw_dim = int(context_node_raw_dim)
        self.context_graph_edge_dim = int(context_graph_edge_dim)
        self.use_context_graph = bool(pomog_use_context_graph)
        self.use_residue_features = bool(pomog_use_residue_features)
        self.context_residual_scale = float(pomog_context_residual_scale)
        self.writeback_enabled = bool(pomog_writeback_enabled)
        self.detach_aux_input = bool(pomog_detach_aux_input)
        self.logit_prior_enabled = bool(pomog_logit_prior_enabled)
        self.logit_prior_scale = float(pomog_logit_prior_scale)
        self.classifier_fusion = str(pomog_classifier_fusion or "legacy").lower()
        if self.classifier_fusion in {"concat", "context_concat"}:
            self.classifier_fusion = "concat_expand"
        self.classifier_use_residue_features = bool(pomog_classifier_use_residue_features)
        self.pocket_bias_enabled = bool(pomog_pocket_bias_enabled)
        self.reliability_moe_enabled = bool(pomog_reliability_moe_enabled)
        self.moe_gate_temperature = max(0.05, float(pomog_moe_gate_temperature))
        self.use_pocket_softfeature_gate = bool(use_pocket_softfeature_gate)
        self.pocket_gate_feature_dim = int(pocket_feature_dim or self.residue_pocket_feature_dim)
        self.pocket_feature_dropout = max(0.0, min(1.0, float(pocket_feature_dropout)))
        self.structure_prior_feature_dim = int(structure_prior_feature_dim)
        self.pomog_structure_prior_gate_enabled = bool(pomog_structure_prior_gate_enabled)
        self.pomog_structure_prior_shuffle_enabled = bool(pomog_structure_prior_shuffle_enabled)
        self.pomog_structure_prior_shuffle_shift = int(pomog_structure_prior_shuffle_shift)
        self.pomog_lee_enabled = bool(pomog_lee_enabled)
        self.pomog_lee_multiscale_enabled = bool(pomog_lee_multiscale_enabled)
        self.pomog_region_token_lite_enabled = bool(pomog_region_token_lite_enabled)
        self.pomog_dual_route_enabled = bool(pomog_dual_route_enabled)
        self.pomog_structure_prior_moe_v2_enabled = bool(pomog_structure_prior_moe_v2_enabled)
        self.pomog_neg_suppress_enabled = bool(pomog_neg_suppress_enabled)
        self.pomog_dual_delta_enabled = bool(pomog_dual_delta_enabled)
        self.pomog_prior_logit_adapter_enabled = bool(pomog_prior_logit_adapter_enabled)
        self.pomog_prior_logit_adapter_feature_dropout = float(pomog_prior_logit_adapter_feature_dropout)

        pool_dim = hidden_dim * 3
        aux_hidden = max(16, int(hidden_dim * float(pomog_aux_hidden_mult)))
        self.context_pool_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, aux_hidden),
            nn.GELU(),
            nn.Linear(aux_hidden, 1),
        )
        self.context_pool_projection = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.context_raw_projection = nn.Sequential(
            nn.LayerNorm(self.context_node_raw_dim),
            nn.Linear(self.context_node_raw_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_edge_projection = nn.Sequential(
            nn.LayerNorm(self.context_graph_edge_dim),
            nn.Linear(self.context_graph_edge_dim, hidden_dim),
            nn.GELU(),
        )
        self.context_layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    beta=True,
                )
                for _ in range(max(0, int(pomog_context_layers)))
            ]
        )
        self.context_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.context_layers])
        self.context_dropout = nn.Dropout(dropout)

        self.context_q = nn.Linear(hidden_dim, hidden_dim)
        self.context_k = nn.Linear(hidden_dim, hidden_dim)
        self.context_v = nn.Linear(hidden_dim, hidden_dim)
        pair_bias_dropout = dropout if pomog_pocket_bias_dropout is None else float(pomog_pocket_bias_dropout)
        self.pocket_attention_bias: nn.Module | None = None
        if self.pocket_bias_enabled:
            pair_dim = self.residue_pocket_feature_dim + hidden_dim + 1
            self.pocket_attention_bias = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(pair_bias_dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.pocket_attention_bias_scale = nn.Parameter(torch.tensor(float(pomog_pocket_bias_scale_init)))
        self.residue_feature_projection = nn.Sequential(
            nn.LayerNorm(self.residue_pocket_feature_dim),
            nn.Linear(self.residue_pocket_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        gate_dim = hidden_dim * 4 + self.residue_pocket_feature_dim
        self.writeback_delta = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.writeback_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.writeback_norm = nn.LayerNorm(hidden_dim)
        if self.classifier_fusion == "concat_expand":
            classifier_dim = head_dim + hidden_dim
        else:
            classifier_dim = head_dim + hidden_dim * 3
        if self.classifier_use_residue_features:
            classifier_dim += self.residue_pocket_feature_dim
        self.fc1 = nn.Linear(classifier_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.moe_base_head: nn.Module | None = None
        self.moe_p2rank_head: nn.Module | None = None
        self.moe_fpocket_head: nn.Module | None = None
        self.moe_consensus_head: nn.Module | None = None
        self.moe_gate: nn.Module | None = None
        if self.reliability_moe_enabled:
            expert_hidden = max(32, hidden_dim)
            pocket_basis_dim = head_dim + self.residue_pocket_feature_dim
            self.moe_base_head = nn.Sequential(
                nn.LayerNorm(head_dim),
                nn.Linear(head_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            self.moe_p2rank_head = nn.Sequential(
                nn.LayerNorm(pocket_basis_dim),
                nn.Linear(pocket_basis_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            self.moe_fpocket_head = nn.Sequential(
                nn.LayerNorm(pocket_basis_dim),
                nn.Linear(pocket_basis_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            self.moe_consensus_head = nn.Sequential(
                nn.LayerNorm(pocket_basis_dim),
                nn.Linear(pocket_basis_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1),
            )
            self.moe_gate = nn.Sequential(
                nn.LayerNorm(pocket_basis_dim),
                nn.Linear(pocket_basis_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 4),
            )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, int(projection_dim)),
        )
        self.pocket_aux_head = nn.Linear(hidden_dim, 1)
        self.micro_motif_aux_head = nn.Linear(hidden_dim, 1)
        self.logit_prior_head: nn.Module | None = None
        if self.logit_prior_enabled:
            self.logit_prior_head = nn.Sequential(
                nn.LayerNorm(self.residue_pocket_feature_dim),
                nn.Linear(self.residue_pocket_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.logit_prior_gate_logit = nn.Parameter(torch.tensor(float(pomog_logit_prior_gate_init)))

        self.pocket_feature_gate: PocketFeatureGate | None = None
        if self.use_pocket_softfeature_gate:
            self.pocket_feature_gate = PocketFeatureGate(
                pocket_feat_dim=self.pocket_gate_feature_dim,
                hidden_dim=hidden_dim,
                adapter_dim=int(pocket_gate_adapter_dim),
                dropout=float(pocket_gate_dropout),
                gate_bias_init=float(pocket_gate_bias_init),
                alpha_init=float(pocket_gate_alpha_init),
                channel_wise=bool(pocket_gate_channel_wise),
            )

        self.structure_prior_adapter: StructurePriorGatedAdapter | None = None
        if self.pomog_structure_prior_gate_enabled:
            self.structure_prior_adapter = StructurePriorGatedAdapter(
                hidden_dim,
                self.structure_prior_feature_dim,
                dropout=dropout,
                alpha_init=float(pomog_structure_prior_gate_alpha_init),
                gate_bias_init=float(pomog_structure_prior_gate_bias_init),
                detector_dropout=float(pomog_structure_prior_detector_dropout),
            )
        self.local_environment_encoder: LocalEnvironmentEncoder | MultiScaleLocalEnvironmentEncoder | None = None
        if self.pomog_lee_enabled:
            if self.pomog_lee_multiscale_enabled:
                self.local_environment_encoder = MultiScaleLocalEnvironmentEncoder(
                    hidden_dim,
                    prior_dim=self.structure_prior_feature_dim,
                    radii=pomog_lee_multiscale_radii,
                    max_neighbors=pomog_lee_multiscale_max_neighbors,
                    dropout=dropout,
                    use_prior=bool(pomog_lee_use_prior),
                    gate_bias_init=float(pomog_lee_gate_bias_init),
                )
            else:
                self.local_environment_encoder = LocalEnvironmentEncoder(
                    hidden_dim,
                    prior_dim=self.structure_prior_feature_dim,
                    dropout=dropout,
                    use_prior=bool(pomog_lee_use_prior),
                    gate_bias_init=float(pomog_lee_gate_bias_init),
                )
        self.region_token_lite_adapter: RegionTokenLiteAdapter | None = None
        if self.pomog_region_token_lite_enabled:
            self.region_token_lite_adapter = RegionTokenLiteAdapter(
                hidden_dim,
                dropout=dropout,
                alpha_init=float(pomog_region_token_lite_alpha_init),
                gate_bias_init=float(pomog_region_token_lite_gate_bias_init),
            )
        local_dim = hidden_dim if self.pomog_lee_enabled else 0
        self.dual_route_head: DualRoutePocketRescueHead | None = None
        if self.pomog_dual_route_enabled:
            self.dual_route_head = DualRoutePocketRescueHead(
                head_dim,
                prior_dim=self.structure_prior_feature_dim,
                local_dim=local_dim,
                hidden_dim=max(32, hidden_dim),
                dropout=dropout,
                gate_bias_init=float(pomog_dual_route_gate_bias_init),
                combination=str(pomog_dual_route_combination),
                main_scale=float(pomog_dual_route_main_scale),
            )
        self.structure_prior_moe_v2_head: StructurePriorMoEHead | None = None
        if self.pomog_structure_prior_moe_v2_enabled:
            self.structure_prior_moe_v2_head = StructurePriorMoEHead(
                head_dim,
                prior_dim=self.structure_prior_feature_dim,
                local_dim=local_dim,
                hidden_dim=max(32, hidden_dim),
                dropout=dropout,
                temperature=float(pomog_structure_prior_moe_temperature),
                gate_entropy_weight=float(pomog_structure_prior_moe_gate_entropy_weight),
            )

        logit_adapter_hidden = max(32, hidden_dim)
        logit_basis_dim = classifier_dim
        prior_logit_basis_dim = self.structure_prior_feature_dim
        self.neg_suppress_gate: nn.Module | None = None
        self.neg_suppress_delta: nn.Module | None = None
        self.neg_suppress_alpha: nn.Parameter | None = None
        if self.pomog_neg_suppress_enabled or self.pomog_dual_delta_enabled:
            self.neg_suppress_gate = self._logit_head(
                logit_basis_dim,
                logit_adapter_hidden,
                dropout,
                final_bias=float(pomog_dual_delta_neg_gate_bias_init if self.pomog_dual_delta_enabled else pomog_neg_suppress_gate_bias_init),
            )
            self.neg_suppress_delta = self._logit_head(logit_basis_dim, logit_adapter_hidden, dropout)
            self.neg_suppress_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.rescue_boost_gate: nn.Module | None = None
        self.rescue_boost_delta: nn.Module | None = None
        self.rescue_boost_alpha: nn.Parameter | None = None
        if self.pomog_dual_delta_enabled:
            self.rescue_boost_gate = self._logit_head(
                logit_basis_dim,
                logit_adapter_hidden,
                dropout,
                final_bias=float(pomog_dual_delta_rescue_gate_bias_init),
            )
            self.rescue_boost_delta = self._logit_head(logit_basis_dim, logit_adapter_hidden, dropout)
            self.rescue_boost_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.prior_logit_adapter: nn.Module | None = None
        self.prior_logit_adapter_alpha: nn.Parameter | None = None
        if self.pomog_prior_logit_adapter_enabled:
            self.prior_logit_adapter = self._logit_head(
                prior_logit_basis_dim,
                logit_adapter_hidden,
                float(pomog_prior_logit_adapter_dropout),
            )
            self.prior_logit_adapter_alpha = nn.Parameter(torch.tensor(float(pomog_prior_logit_adapter_alpha_init), dtype=torch.float32))

        self._last_residue_gate: torch.Tensor | None = None
        self._last_residue_target: torch.Tensor | None = None
        self._last_pocket_aux_logits: torch.Tensor | None = None
        self._last_pocket_aux_target: torch.Tensor | None = None
        self._last_micro_motif_aux_logits: torch.Tensor | None = None
        self._last_micro_motif_aux_target: torch.Tensor | None = None
        self._last_context_node_state: torch.Tensor | None = None
        self._last_moe_weights: torch.Tensor | None = None
        self._last_structure_prior_gate: torch.Tensor | None = None
        self._last_pocket_feature_gate: torch.Tensor | None = None
        self._last_pocket_feature_alpha: torch.Tensor | None = None
        self._last_local_env: torch.Tensor | None = None
        self._last_region_token_gate: torch.Tensor | None = None
        self._last_region_token_count: int = 0
        self._last_dual_route_gate: torch.Tensor | None = None
        self._last_pocket_route_logits: torch.Tensor | None = None
        self._last_rescue_route_logits: torch.Tensor | None = None
        self._last_structure_prior_moe_v2_weights: torch.Tensor | None = None
        self._last_base_logits: torch.Tensor | None = None
        self._last_final_delta_logits: torch.Tensor | None = None
        self._last_neg_suppress_gate: torch.Tensor | None = None
        self._last_neg_suppress_delta: torch.Tensor | None = None
        self._last_neg_suppress_amount: torch.Tensor | None = None
        self._last_rescue_boost_gate: torch.Tensor | None = None
        self._last_rescue_boost_delta: torch.Tensor | None = None
        self._last_rescue_boost_amount: torch.Tensor | None = None
        self._last_prior_logit_delta: torch.Tensor | None = None
        self._init_pomog_pocket(float(pomog_gate_init))

    @staticmethod
    def _logit_head(
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        final_bias: float = 0.0,
        zero_final: bool = False,
    ) -> nn.Sequential:
        head = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        for child in head.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
        final = head[-1]
        if isinstance(final, nn.Linear):
            if zero_final:
                nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.constant_(final.bias, float(final_bias))
        return head

    def _init_pomog_pocket(self, gate_init: float) -> None:
        modules = (
            self.context_pool_score,
            self.context_pool_projection,
            self.context_raw_projection,
            self.context_edge_projection,
            self.context_q,
            self.context_k,
            self.context_v,
            self.pocket_attention_bias,
            self.residue_feature_projection,
            self.writeback_delta,
            self.writeback_gate,
            self.fc1,
            self.fc2,
            self.moe_base_head,
            self.moe_p2rank_head,
            self.moe_fpocket_head,
            self.moe_consensus_head,
            self.moe_gate,
            self.projection_head,
            self.pocket_aux_head,
            self.micro_motif_aux_head,
            self.logit_prior_head,
        )
        for module in modules:
            if module is None:
                continue
            for child in module.modules():
                if isinstance(child, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(child.weight)
                    if getattr(child, "bias", None) is not None:
                        nn.init.zeros_(child.bias)
        final_delta = self.writeback_delta[-1]
        if isinstance(final_delta, nn.Linear):
            nn.init.zeros_(final_delta.weight)
            if final_delta.bias is not None:
                nn.init.zeros_(final_delta.bias)
        final_gate = self.writeback_gate[-1]
        if isinstance(final_gate, nn.Linear) and final_gate.bias is not None:
            nn.init.constant_(final_gate.bias, float(gate_init))
        if self.pocket_attention_bias is not None:
            final_bias = self.pocket_attention_bias[-1]
            if isinstance(final_bias, nn.Linear):
                nn.init.zeros_(final_bias.weight)
                if final_bias.bias is not None:
                    nn.init.zeros_(final_bias.bias)

    def _sanitize_structure_prior(
        self,
        prior_feat: torch.Tensor | None,
        node_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        out = torch.zeros((node_count, self.structure_prior_feature_dim), device=device)
        if prior_feat is None or prior_feat.numel() == 0:
            return out
        feat = torch.nan_to_num(prior_feat.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        take_n = min(node_count, int(feat.shape[0]))
        take_f = min(self.structure_prior_feature_dim, int(feat.shape[-1]))
        out[:take_n, :take_f] = feat[:take_n, :take_f]
        if self.training and self.pomog_prior_logit_adapter_feature_dropout > 0.0:
            keep = torch.rand((self.structure_prior_feature_dim,), device=device) >= self.pomog_prior_logit_adapter_feature_dropout
            out = out * keep.float().view(1, -1)
        return out

    def _sanitize_residue_feature(self, feat: torch.Tensor, node_count: int, device: torch.device) -> torch.Tensor:
        if feat is None or feat.numel() == 0:
            return torch.zeros((node_count, self.residue_pocket_feature_dim), device=device)
        feat = torch.nan_to_num(feat.float().to(device), nan=0.0, posinf=1.0, neginf=0.0)
        if feat.shape[0] != node_count:
            out = torch.zeros((node_count, self.residue_pocket_feature_dim), device=device, dtype=feat.dtype)
            take = min(node_count, int(feat.shape[0]))
            out[:take, : min(self.residue_pocket_feature_dim, feat.shape[-1])] = feat[:take, : self.residue_pocket_feature_dim]
            return out
        if feat.shape[-1] == self.residue_pocket_feature_dim:
            return feat
        out = torch.zeros((node_count, self.residue_pocket_feature_dim), device=device, dtype=feat.dtype)
        out[:, : min(self.residue_pocket_feature_dim, feat.shape[-1])] = feat[:, : self.residue_pocket_feature_dim]
        return out

    def _encode_residues(
        self,
        X: torch.Tensor,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        seq: torch.Tensor,
        batch_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_id is None:
            batch_id = torch.zeros(h_V.size(0), device=h_V.device, dtype=torch.long)
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)
        if self.pomog_encoder in {"pomog_baseline", "baseline", "graphec_as"}:
            edge_index = edge_index.long()
            h_V_geo, h_E = get_geo_feat(X.float(), edge_index)
            h_V = torch.cat([h_V.float(), h_V_geo.float()], dim=-1)
            h_V = self.encoder(h_V, edge_index, h_E.float(), seq, batch_id.long())
            return h_V, edge_index
        edge_index = self._prepare_edge_index(X.float(), edge_index.long(), batch_id.long())
        h_V_geo, h_E = get_geo_feat(X.float(), edge_index)
        node_vectors, edge_vectors, edge_vector_scalar = _build_vector_geometry(
            X.float(),
            edge_index,
            use_sidechain_scalar=self.use_sidechain_scalar,
            use_distance_drift=self.use_distance_drift,
            edge_vector_mode=self.edge_vector_mode,
        )
        h_V = torch.cat([h_V.float(), h_V_geo.float()], dim=-1)
        h_V = self.encoder(
            h_V,
            edge_index,
            h_E.float(),
            seq,
            batch_id.long(),
            node_vectors,
            edge_vectors,
            edge_vector_scalar,
        )
        return h_V, edge_index

    def _pool_context_nodes(
        self,
        h_residue: torch.Tensor,
        member_index: torch.Tensor,
        node_count: int,
        raw_feature: torch.Tensor,
    ) -> torch.Tensor:
        device = h_residue.device
        if node_count <= 0:
            return h_residue.new_zeros((0, h_residue.size(-1)))
        raw_feature = torch.nan_to_num(raw_feature.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        if raw_feature.shape[0] != node_count:
            fixed = h_residue.new_zeros((node_count, self.context_node_raw_dim))
            take = min(node_count, int(raw_feature.shape[0]))
            fixed[:take, : min(self.context_node_raw_dim, raw_feature.shape[-1])] = raw_feature[:take, : self.context_node_raw_dim]
            raw_feature = fixed
        if member_index is None or member_index.numel() == 0:
            pooled = h_residue.new_zeros((node_count, h_residue.size(-1) * 3))
        else:
            member = member_index.long().to(device)
            residues = member[0].clamp(min=0, max=max(int(h_residue.size(0)) - 1, 0))
            nodes = member[1].clamp(min=0, max=max(node_count - 1, 0))
            source = h_residue[residues]
            mean_state = scatter_mean(source, nodes, dim=0, dim_size=node_count)
            max_state, _ = scatter_max(source, nodes, dim=0, dim_size=node_count)
            max_state = torch.where(torch.isfinite(max_state), max_state, torch.zeros_like(max_state))
            scores = self.context_pool_score(source).view(-1)
            alpha = softmax(scores, nodes, num_nodes=node_count)
            attn_state = scatter_add(alpha.unsqueeze(-1) * source, nodes, dim=0, dim_size=node_count)
            pooled = torch.cat([mean_state, max_state, attn_state], dim=-1)
        return self.context_pool_projection(pooled) + self.context_raw_projection(raw_feature)

    def _apply_context_graph(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_context_graph or node_state.numel() == 0 or edge_index is None or edge_index.numel() == 0:
            return node_state
        edge_index = edge_index.long().to(node_state.device)
        edge_attr = torch.nan_to_num(edge_attr.float().to(node_state.device), nan=0.0, posinf=0.0, neginf=0.0)
        edge_hidden = self.context_edge_projection(edge_attr)
        for conv, norm in zip(self.context_layers, self.context_norms):
            update = conv(node_state, edge_index, edge_hidden)
            node_state = norm(node_state + self.context_dropout(update))
        return node_state

    def _context_to_residue(
        self,
        h_residue: torch.Tensor,
        node_state: torch.Tensor,
        context_edge_index: torch.Tensor,
        residue_feature: torch.Tensor | None = None,
        context_node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_state.numel() == 0 or context_edge_index is None or context_edge_index.numel() == 0:
            return torch.zeros_like(h_residue)
        edge_index = context_edge_index.long().to(h_residue.device)
        residues = edge_index[0].clamp(min=0, max=max(int(h_residue.size(0)) - 1, 0))
        nodes = edge_index[1].clamp(min=0, max=max(int(node_state.size(0)) - 1, 0))
        q = self.context_q(h_residue[residues])
        k = self.context_k(node_state[nodes])
        v = self.context_v(node_state[nodes])
        scores = (q * k).sum(dim=-1) / max(float(q.size(-1)) ** 0.5, 1.0)
        if self.pocket_attention_bias is not None and residue_feature is not None:
            residue_part = residue_feature[residues].float()
            node_type = (
                context_node_type[nodes].float().view(-1, 1)
                if context_node_type is not None and context_node_type.numel() > 0
                else torch.zeros((int(nodes.numel()), 1), device=h_residue.device, dtype=h_residue.dtype)
            )
            pair = torch.cat([residue_part, node_state[nodes].float(), node_type], dim=-1)
            bias = self.pocket_attention_bias(pair).view(-1)
            scores = scores + self.pocket_attention_bias_scale * bias
        alpha = softmax(scores, residues, num_nodes=h_residue.size(0))
        return scatter_add(alpha.unsqueeze(-1) * v, residues, dim=0, dim_size=h_residue.size(0))

    def _apply_reliability_moe(self, encoded: torch.Tensor, residue_feature: torch.Tensor) -> torch.Tensor:
        if (
            self.moe_base_head is None
            or self.moe_p2rank_head is None
            or self.moe_fpocket_head is None
            or self.moe_consensus_head is None
            or self.moe_gate is None
        ):
            raise RuntimeError("reliability MoE requested but MoE heads were not initialized")
        pocket_basis = torch.cat([encoded, residue_feature], dim=-1)
        gate_logits = self.moe_gate(pocket_basis) / self.moe_gate_temperature
        weights = F.softmax(gate_logits, dim=-1)
        expert_logits = torch.stack(
            [
                self.moe_base_head(encoded),
                self.moe_p2rank_head(pocket_basis),
                self.moe_fpocket_head(pocket_basis),
                self.moe_consensus_head(pocket_basis),
            ],
            dim=1,
        )
        self._last_moe_weights = weights.detach()
        return (weights.unsqueeze(-1) * expert_logits).sum(dim=1)

    def _writeback(
        self,
        h_residue: torch.Tensor,
        context: torch.Tensor,
        residue_feature: torch.Tensor,
    ) -> torch.Tensor:
        if not self.writeback_enabled:
            self._last_residue_gate = torch.zeros_like(h_residue)
            return h_residue
        basis = torch.cat(
            [
                h_residue,
                context,
                h_residue * context,
                torch.abs(h_residue - context),
                residue_feature,
            ],
            dim=-1,
        )
        delta = self.writeback_delta(basis)
        gate = torch.sigmoid(self.writeback_gate(basis))
        self._last_residue_gate = gate.detach()
        return self.writeback_norm(h_residue + self.context_residual_scale * gate * delta)

    def _maybe_shuffle_structure_prior(
        self,
        prior_feat: torch.Tensor | None,
        prior_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.pomog_structure_prior_shuffle_enabled or prior_feat is None or prior_feat.numel() == 0:
            return prior_feat, prior_mask
        node_count = int(prior_feat.shape[0])
        if node_count <= 1:
            return prior_feat, prior_mask
        shift = int(self.pomog_structure_prior_shuffle_shift) % node_count
        if shift == 0:
            shift = max(1, node_count // 3)
        shuffled_feat = torch.roll(prior_feat, shifts=shift, dims=0)
        shuffled_mask = torch.roll(prior_mask, shifts=shift, dims=0) if prior_mask is not None and prior_mask.numel() > 0 else prior_mask
        return shuffled_feat, shuffled_mask

    def _store_aux(
        self,
        node_state: torch.Tensor,
        node_type: torch.Tensor,
        context_label: torch.Tensor | None,
    ) -> None:
        self._last_pocket_aux_logits = None
        self._last_pocket_aux_target = None
        self._last_micro_motif_aux_logits = None
        self._last_micro_motif_aux_target = None
        self._last_context_node_state = node_state.detach() if node_state.numel() else None
        if node_state.numel() == 0 or node_type is None or node_type.numel() == 0:
            return
        node_type = node_type.long().to(node_state.device).view(-1)
        if context_label is not None and context_label.numel() > 0:
            labels = context_label.float().to(node_state.device).view(-1, 1)
        else:
            labels = None
        aux_state = node_state.detach() if self.detach_aux_input else node_state
        micro_mask = node_type == 0
        pocket_mask = node_type == 1
        if bool(micro_mask.any()):
            self._last_micro_motif_aux_logits = self.micro_motif_aux_head(aux_state[micro_mask])
            if labels is not None:
                self._last_micro_motif_aux_target = labels[micro_mask]
        if bool(pocket_mask.any()):
            self._last_pocket_aux_logits = self.pocket_aux_head(aux_state[pocket_mask])
            if labels is not None:
                self._last_pocket_aux_target = labels[pocket_mask]

    def forward(
        self,
        X: torch.Tensor,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        seq: torch.Tensor,
        batch_id: torch.Tensor,
        residue_pocket_feature: torch.Tensor,
        context_node_type: torch.Tensor,
        context_node_raw_feature: torch.Tensor,
        context_node_member_index: torch.Tensor,
        context_graph_edge_index: torch.Tensor,
        context_graph_edge_attr: torch.Tensor,
        context_edge_index: torch.Tensor,
        context_node_label: torch.Tensor | None = None,
        structure_prior_feat: torch.Tensor | None = None,
        structure_prior_mask: torch.Tensor | None = None,
        structure_prior_detector_missing: torch.Tensor | None = None,
        return_projection: bool = False,
        structure_prior_pocket_feature: torch.Tensor | None = None,
        structure_prior_residue_to_pocket_index: torch.Tensor | None = None,
        structure_prior_residue_to_pocket_attr: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_residue, edge_index = self._encode_residues(X, h_V, edge_index, seq, batch_id)
        residue_feature = self._sanitize_residue_feature(
            residue_pocket_feature,
            int(h_residue.size(0)),
            h_residue.device,
        )
        context_node_type = context_node_type.long().to(h_residue.device).view(-1)
        node_count = int(context_node_type.numel())
        node_state = self._pool_context_nodes(
            h_residue,
            context_node_member_index,
            node_count,
            context_node_raw_feature,
        )
        node_state = self._apply_context_graph(node_state, context_graph_edge_index, context_graph_edge_attr)
        self._store_aux(node_state, context_node_type, context_node_label)

        context = self._context_to_residue(
            h_residue,
            node_state,
            context_edge_index,
            residue_feature=residue_feature,
            context_node_type=context_node_type,
        )
        if self.use_residue_features:
            context = context + self.residue_feature_projection(residue_feature)
        h_new = self._writeback(h_residue, context, residue_feature)
        structure_prior_input, structure_prior_mask_input = self._maybe_shuffle_structure_prior(
            structure_prior_feat,
            structure_prior_mask,
        )
        self._last_structure_prior_gate = None
        self._last_local_env = None
        self._last_pocket_feature_gate = None
        self._last_pocket_feature_alpha = None
        if self.pocket_feature_gate is not None:
            gate_feat = residue_feature[:, : min(self.pocket_gate_feature_dim, int(residue_feature.shape[-1]))]
            if self.training and self.pocket_feature_dropout > 0.0 and gate_feat.numel() > 0:
                keep = torch.rand((int(gate_feat.shape[-1]),), device=gate_feat.device) >= self.pocket_feature_dropout
                gate_feat = gate_feat * keep.float().view(1, -1)
            h_new, pocket_gate_stats = self.pocket_feature_gate(h_new, gate_feat)
            self._last_pocket_feature_gate = pocket_gate_stats["gate"]
            self._last_pocket_feature_alpha = pocket_gate_stats["alpha"]
        if self.structure_prior_adapter is not None:
            h_new, _, prior_gate = self.structure_prior_adapter(h_new, structure_prior_input, structure_prior_mask_input)
            self._last_structure_prior_gate = prior_gate.detach()
        local_env = None
        if self.local_environment_encoder is not None:
            h_new, local_env = self.local_environment_encoder(h_new, X, edge_index, seq, structure_prior_input)
            self._last_local_env = local_env.detach()
        self._last_region_token_gate = None
        self._last_region_token_count = 0
        if self.region_token_lite_adapter is not None:
            h_new, region_tokens, region_gate = self.region_token_lite_adapter(
                h_new,
                structure_prior_pocket_feature,
                structure_prior_residue_to_pocket_index,
                structure_prior_residue_to_pocket_attr,
            )
            self._last_region_token_gate = region_gate.detach()
            self._last_region_token_count = int(region_tokens.shape[0])
        self._last_residue_target = None

        if self.classifier_fusion == "concat_expand":
            encoded = self._build_head_features(h_residue, edge_index)
            classifier_parts = [encoded, context]
            projection_state = h_residue
        else:
            encoded = self._build_head_features(h_new, edge_index)
            classifier_parts = [encoded, h_residue, h_new - h_residue, context]
            projection_state = h_new
        if self.classifier_use_residue_features:
            classifier_parts.append(residue_feature)
        classifier_input = torch.cat(classifier_parts, dim=-1)
        self._last_moe_weights = None
        self._last_dual_route_gate = None
        self._last_pocket_route_logits = None
        self._last_rescue_route_logits = None
        self._last_structure_prior_moe_v2_weights = None
        self._last_base_logits = None
        self._last_final_delta_logits = None
        self._last_neg_suppress_gate = None
        self._last_neg_suppress_delta = None
        self._last_neg_suppress_amount = None
        self._last_rescue_boost_gate = None
        self._last_rescue_boost_delta = None
        self._last_rescue_boost_amount = None
        self._last_prior_logit_delta = None
        out = self.fc2(F.elu(self.fc1(classifier_input)))
        if self.reliability_moe_enabled:
            out = self._apply_reliability_moe(encoded, residue_feature)
        if self.dual_route_head is not None:
            route = self.dual_route_head(encoded, structure_prior_input, local_env)
            out = route.final_logit
            self._last_dual_route_gate = route.route_gate.detach()
            self._last_pocket_route_logits = route.pocket_logit
            self._last_rescue_route_logits = route.rescue_logit
        if self.structure_prior_moe_v2_head is not None:
            out, weights = self.structure_prior_moe_v2_head(
                encoded,
                structure_prior_input,
                local_env,
                structure_prior_detector_missing,
            )
            self._last_structure_prior_moe_v2_weights = weights.detach()
        if self.logit_prior_head is not None:
            prior = self.logit_prior_head(residue_feature)
            out = out + self.logit_prior_scale * torch.sigmoid(self.logit_prior_gate_logit) * prior
        self._last_base_logits = out.detach()
        base_out = out
        if self.neg_suppress_gate is not None and self.neg_suppress_delta is not None and self.neg_suppress_alpha is not None:
            neg_gate = torch.sigmoid(self.neg_suppress_gate(classifier_input))
            neg_delta = self.neg_suppress_delta(classifier_input)
            neg_amount = self.neg_suppress_alpha * neg_gate * F.softplus(neg_delta)
            out = out - neg_amount
            self._last_neg_suppress_gate = neg_gate.detach()
            self._last_neg_suppress_delta = neg_delta.detach()
            self._last_neg_suppress_amount = neg_amount
        if self.rescue_boost_gate is not None and self.rescue_boost_delta is not None and self.rescue_boost_alpha is not None:
            rescue_gate = torch.sigmoid(self.rescue_boost_gate(classifier_input))
            rescue_delta = self.rescue_boost_delta(classifier_input)
            rescue_amount = self.rescue_boost_alpha * rescue_gate * F.softplus(rescue_delta)
            out = out + rescue_amount
            self._last_rescue_boost_gate = rescue_gate.detach()
            self._last_rescue_boost_delta = rescue_delta.detach()
            self._last_rescue_boost_amount = rescue_amount
        if self.prior_logit_adapter is not None and self.prior_logit_adapter_alpha is not None:
            prior_basis = self._sanitize_structure_prior(structure_prior_input, int(out.shape[0]), out.device)
            prior_delta = self.prior_logit_adapter(prior_basis)
            prior_amount = self.prior_logit_adapter_alpha * prior_delta
            out = out + prior_amount
            self._last_prior_logit_delta = prior_amount
        self._last_final_delta_logits = (out - base_out).detach()
        if not return_projection:
            return out
        projection = self.projection_head(projection_state)
        return out, projection, projection_state
