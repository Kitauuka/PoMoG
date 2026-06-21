from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv, radius_graph
from torch_scatter import scatter_mean

from pomog.data.graph import get_geo_feat
from pomog.models.layers import Context
from pomog.models.vector_layers import GVP, GVPConvLayer, GVPLayerNorm

_GEO_NODE_DIM = 184
_CA_COORD_INDEX = 1
_NODE_VECTOR_CHANNELS = 3
_EDGE_VECTOR_CHANNELS = 1
_MULTIVECTOR_ATOM_PAIRS = (
    (1, 1),  # CA_src -> CA_dst
    (0, 3),  # N_src -> O_dst
    (3, 0),  # O_src -> N_dst
    (4, 1),  # sidechain_src -> CA_dst
    (1, 4),  # CA_src -> sidechain_dst
    (4, 4),  # sidechain_src -> sidechain_dst
    (2, 0),  # C_src -> N_dst
    (0, 2),  # N_src -> C_dst
)
_BASE_VECTOR_EDGE_FEATURE_DIM = 10
_SIDECHAIN_EDGE_FEATURE_DIM = 4
_DISTANCE_DRIFT_FEATURE_DIM = 1


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return torch.nan_to_num(x / denom, nan=0.0, posinf=0.0, neginf=0.0)


def _build_vector_geometry(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    use_sidechain_scalar: bool = False,
    use_distance_drift: bool = False,
    edge_vector_mode: str = "ca",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = X.float()
    ca = X[:, _CA_COORD_INDEX]
    node_vectors = torch.stack(
        [
            X[:, 0] - ca,
            X[:, 2] - ca,
            X[:, 3] - ca,
        ],
        dim=1,
    )
    node_vectors = torch.nan_to_num(node_vectors, nan=0.0, posinf=0.0, neginf=0.0)

    src, dst = edge_index.long()
    edge_raw = ca[dst] - ca[src]
    edge_dist = torch.linalg.norm(edge_raw, dim=-1, keepdim=True)
    ca_edge_vectors = _safe_normalize(edge_raw).unsqueeze(1)
    if edge_vector_mode == "atom_pair_8":
        edge_vectors = torch.stack(
            [
                _safe_normalize(X[dst, dst_atom] - X[src, src_atom])
                for src_atom, dst_atom in _MULTIVECTOR_ATOM_PAIRS
            ],
            dim=1,
        )
        edge_vectors = torch.nan_to_num(edge_vectors, nan=0.0, posinf=0.0, neginf=0.0)
    elif edge_vector_mode == "ca":
        edge_vectors = ca_edge_vectors
    else:
        raise ValueError(f"unsupported edge_vector_mode: {edge_vector_mode!r}")
    edge_dir = ca_edge_vectors.squeeze(1)

    src_vec = _safe_normalize(node_vectors[src])
    dst_vec = _safe_normalize(node_vectors[dst])
    dot_src = (src_vec * edge_dir.unsqueeze(1)).sum(dim=-1)
    dot_dst = (dst_vec * (-edge_dir).unsqueeze(1)).sum(dim=-1)
    local_alignment = (src_vec * dst_vec).sum(dim=-1)
    edge_dist_scaled = torch.clamp(edge_dist / 20.0, min=0.0, max=2.0)

    edge_scalar_parts = [dot_src, dot_dst, local_alignment, edge_dist_scaled]
    if use_sidechain_scalar:
        side_raw = X[:, 4] - ca
        side_norm = _safe_normalize(side_raw)
        side_available = (
            torch.isfinite(side_raw).all(dim=-1, keepdim=True)
            & (torch.linalg.norm(side_raw, dim=-1, keepdim=True) > 1e-4)
        ).to(dtype=X.dtype)
        side_src = side_norm[src]
        side_dst = side_norm[dst]
        side_dot_src = (side_src * edge_dir).sum(dim=-1, keepdim=True)
        side_dot_dst = (side_dst * (-edge_dir)).sum(dim=-1, keepdim=True)
        side_alignment = (side_src * side_dst).sum(dim=-1, keepdim=True)
        side_pair_available = side_available[src] * side_available[dst]
        edge_scalar_parts.append(
            torch.cat([side_dot_src, side_dot_dst, side_alignment, side_pair_available], dim=-1)
        )
    if use_distance_drift:
        virtual_shift = 0.1 * _safe_normalize(node_vectors.sum(dim=1))
        virtual_ca = ca + virtual_shift
        virtual_dist = torch.linalg.norm(virtual_ca[dst] - virtual_ca[src], dim=-1, keepdim=True)
        drift = torch.clamp((virtual_dist - edge_dist) / 5.0, min=-1.0, max=1.0)
        edge_scalar_parts.append(drift)

    edge_scalar = torch.cat(edge_scalar_parts, dim=-1)
    edge_scalar = torch.nan_to_num(edge_scalar, nan=0.0, posinf=0.0, neginf=0.0)
    return node_vectors, edge_vectors, edge_scalar


def _edge_vector_feature_dim(use_sidechain_scalar: bool, use_distance_drift: bool) -> int:
    dim = _BASE_VECTOR_EDGE_FEATURE_DIM
    if use_sidechain_scalar:
        dim += _SIDECHAIN_EDGE_FEATURE_DIM
    if use_distance_drift:
        dim += _DISTANCE_DRIFT_FEATURE_DIM
    return dim


def _unique_edges(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.long()
    return torch.unique(edge_index.long().t(), dim=0).t().contiguous()


def _build_sequence_edges(batch_id: torch.Tensor, span: int = 1) -> torch.Tensor:
    span = max(1, int(span))
    node_count = int(batch_id.numel())
    device = batch_id.device
    chunks = []
    for offset in range(1, span + 1):
        if node_count <= offset:
            break
        src = torch.arange(0, node_count - offset, device=device, dtype=torch.long)
        dst = src + offset
        mask = batch_id[src] == batch_id[dst]
        if bool(mask.any()):
            src = src[mask]
            dst = dst[mask]
            chunks.append(torch.stack([src, dst], dim=0))
            chunks.append(torch.stack([dst, src], dim=0))
    if not chunks:
        return torch.empty((2, 0), device=device, dtype=torch.long)
    return torch.cat(chunks, dim=1)


class _GatedEdgeAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        gate_init: float,
        input_dim: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, edge_vector_scalar: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit) * self.proj(edge_vector_scalar)


class _VectorNodeAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        gate_init: float,
        input_dim: int,
    ) -> None:
        super().__init__()
        message_dim = 3 * hidden_dim + int(input_dim)
        self.message = nn.Sequential(
            nn.LayerNorm(message_dim),
            nn.Linear(message_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        edge_vector_scalar: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index.long()
        message_input = torch.cat([h_V[src], h_E, h_V[dst], edge_vector_scalar], dim=-1)
        messages = self.message(message_input)
        delta = scatter_mean(messages, dst, dim=0, dim_size=h_V.size(0))
        return torch.sigmoid(self.gate_logit) * delta


class _GeometryFiLMAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, node_geom: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(node_geom).chunk(2, dim=-1)
        gate = torch.sigmoid(self.gate_logit)
        return dh * (1.0 + gate * torch.tanh(gamma)) + gate * beta


class _EdgeGroupGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, h_E: torch.Tensor, edge_vector_scalar: torch.Tensor) -> torch.Tensor:
        scale = torch.tanh(self.net(edge_vector_scalar))
        return h_E * (1.0 + torch.sigmoid(self.gate_logit) * scale)


class _CoordinateEdgeAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        message_dim = 3 * hidden_dim + int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(message_dim),
            nn.Linear(message_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        edge_vector_scalar: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index.long()
        inp = torch.cat([h_V[src], h_E, h_V[dst], edge_vector_scalar], dim=-1)
        return torch.sigmoid(self.gate_logit) * self.net(inp)


class _GeometryContextAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, edge_index: torch.Tensor, edge_vector_scalar: torch.Tensor, node_count: int) -> torch.Tensor:
        _, dst = edge_index.long()
        node_geom = scatter_mean(edge_vector_scalar, dst, dim=0, dim_size=node_count)
        return torch.sigmoid(self.gate_logit) * self.net(node_geom)


class _GVPLiteAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_vector_dim: int,
        dropout: float,
        gate_init: float,
        num_message_gvps: int,
        num_ff_gvps: int,
        vector_gate: bool,
        edge_vector_channels: int = _EDGE_VECTOR_CHANNELS,
    ) -> None:
        super().__init__()
        hidden_dims = (int(hidden_dim), int(hidden_vector_dim))
        node_dims = (int(hidden_dim), _NODE_VECTOR_CHANNELS)
        edge_dims = (int(hidden_dim), int(edge_vector_channels))
        self.node_embed = nn.Sequential(
            GVPLayerNorm(node_dims),
            GVP(node_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate),
        )
        self.edge_embed = nn.Sequential(
            GVPLayerNorm(edge_dims),
            GVP(edge_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate),
        )
        self.layer = GVPConvLayer(
            hidden_dims,
            hidden_dims,
            num_message_gvps=max(1, int(num_message_gvps)),
            num_ff_gvps=max(1, int(num_ff_gvps)),
            dropout=dropout,
            vector_gate=vector_gate,
        )
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        h_V: torch.Tensor,
        node_vectors: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self.node_embed((h_V, node_vectors))
        edge_state = self.edge_embed((h_E, edge_vectors))
        updated_scalar, _ = self.layer(node_state, edge_index.long(), edge_state)
        delta = self.project(updated_scalar - node_state[0])
        return torch.sigmoid(self.gate_logit) * delta


class _PoMoGEdgeMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.2,
        geom_dim: int = 0,
    ) -> None:
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.W11 = nn.Linear(3 * hidden_dim + self.geom_dim, hidden_dim, bias=True)
        self.W12 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.GELU()

    def forward(
        self,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        edge_geom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index.long()
        parts = [h_V[src], h_E, h_V[dst]]
        if self.geom_dim > 0:
            if edge_geom is None:
                edge_geom = h_E.new_zeros((h_E.size(0), self.geom_dim))
            parts.append(edge_geom.to(dtype=h_E.dtype))
        h_EV = torch.cat(parts, dim=-1)
        h_msg = self.W12(self.act(self.W11(h_EV)))
        return self.norm(h_E + self.dropout(h_msg))


class _PoMoGGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.2,
        num_heads: int = 4,
        hidden_vector_dim: int = 16,
        geometry_use_attention_adapter: bool = False,
        geometry_use_edge_adapter: bool = False,
        geometry_use_node_adapter: bool = False,
        geometry_use_gvp_lite: bool = False,
        geometry_use_edgemlp_geom: bool = False,
        geometry_use_confidence_dropout: bool = False,
        geometry_use_ffn_film: bool = False,
        geometry_use_edge_group_gate: bool = False,
        geometry_use_edge_memory: bool = False,
        geometry_use_coordinate_adapter: bool = False,
        geometry_use_distance_drift: bool = False,
        geometry_use_geo_context: bool = False,
        geometry_attention_gate_init: float = -3.0,
        geometry_edge_gate_init: float = -3.0,
        geometry_node_gate_init: float = -3.0,
        geometry_gvp_gate_init: float = -3.0,
        geometry_geom_gate_init: float = -3.0,
        geometry_confidence_dropout_base: float = 0.03,
        geometry_confidence_dropout_alpha: float = 0.12,
        geometry_confidence_dropout_max: float = 0.20,
        geometry_branch_dropout: float = 0.0,
        geometry_num_message_gvps: int = 1,
        geometry_num_ff_gvps: int = 1,
        geometry_vector_gate: bool = True,
        edge_vector_feature_dim: int = _BASE_VECTOR_EDGE_FEATURE_DIM,
        edge_vector_channels: int = _EDGE_VECTOR_CHANNELS,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.attention = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=hidden_dim,
            root_weight=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.use_confidence_dropout = bool(geometry_use_confidence_dropout)
        self.confidence_dropout_base = float(geometry_confidence_dropout_base)
        self.confidence_dropout_alpha = float(geometry_confidence_dropout_alpha)
        self.confidence_dropout_max = float(geometry_confidence_dropout_max)
        self.geometry_branch_dropout = float(geometry_branch_dropout)
        self.use_distance_drift = bool(geometry_use_distance_drift)
        self.edge_vector_feature_dim = int(edge_vector_feature_dim)
        self.edge_update = _PoMoGEdgeMLP(
            hidden_dim,
            dropout,
            geom_dim=self.edge_vector_feature_dim if bool(geometry_use_edgemlp_geom) else 0,
        )
        self.context = Context(hidden_dim)

        self.attention_adapter = (
            _GatedEdgeAdapter(hidden_dim, dropout, geometry_attention_gate_init, self.edge_vector_feature_dim)
            if geometry_use_attention_adapter
            else None
        )
        self.edge_adapter = (
            _GatedEdgeAdapter(hidden_dim, dropout, geometry_edge_gate_init, self.edge_vector_feature_dim)
            if geometry_use_edge_adapter
            else None
        )
        self.node_adapter = (
            _VectorNodeAdapter(hidden_dim, dropout, geometry_node_gate_init, self.edge_vector_feature_dim)
            if geometry_use_node_adapter
            else None
        )
        self.gvp_lite = (
            _GVPLiteAdapter(
                hidden_dim=hidden_dim,
                hidden_vector_dim=hidden_vector_dim,
                dropout=dropout,
                gate_init=geometry_gvp_gate_init,
                num_message_gvps=geometry_num_message_gvps,
                num_ff_gvps=geometry_num_ff_gvps,
                vector_gate=geometry_vector_gate,
                edge_vector_channels=edge_vector_channels,
            )
            if geometry_use_gvp_lite
            else None
        )
        self.ffn_film = (
            _GeometryFiLMAdapter(self.edge_vector_feature_dim, hidden_dim, dropout, geometry_geom_gate_init)
            if geometry_use_ffn_film
            else None
        )
        self.edge_group_gate = (
            _EdgeGroupGate(self.edge_vector_feature_dim, hidden_dim, dropout, geometry_geom_gate_init)
            if geometry_use_edge_group_gate
            else None
        )
        self.coordinate_adapter = (
            _CoordinateEdgeAdapter(self.edge_vector_feature_dim, hidden_dim, dropout, geometry_geom_gate_init)
            if geometry_use_coordinate_adapter or geometry_use_distance_drift
            else None
        )
        self.edge_memory = nn.GRUCell(hidden_dim, hidden_dim) if geometry_use_edge_memory else None
        self.edge_memory_gate_logit = (
            nn.Parameter(torch.tensor(float(geometry_geom_gate_init))) if geometry_use_edge_memory else None
        )
        self.geo_context_adapter = (
            _GeometryContextAdapter(self.edge_vector_feature_dim, hidden_dim, dropout, geometry_geom_gate_init)
            if geometry_use_geo_context
            else None
        )

    def _maybe_drop_geometry_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.training and self.geometry_branch_dropout > 0:
            return F.dropout(delta, p=self.geometry_branch_dropout, training=True)
        return delta

    def _apply_confidence_dropout(
        self,
        h_E: torch.Tensor,
        edge_vector_scalar: torch.Tensor,
    ) -> torch.Tensor:
        if not (self.training and self.use_confidence_dropout):
            return h_E
        dist_scaled = edge_vector_scalar[
            :, _BASE_VECTOR_EDGE_FEATURE_DIM - 1 : _BASE_VECTOR_EDGE_FEATURE_DIM
        ].clamp(min=0.0, max=2.0)
        confidence = torch.exp(-2.5 * dist_scaled)
        drop_prob = self.confidence_dropout_base + self.confidence_dropout_alpha * (1.0 - confidence)
        drop_prob = drop_prob.clamp(min=0.0, max=self.confidence_dropout_max)
        keep = (torch.rand_like(drop_prob) >= drop_prob).to(dtype=h_E.dtype)
        return h_E * keep

    def forward(
        self,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        batch_id: torch.Tensor,
        node_vectors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_vector_scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_edge_attr = h_E
        if self.edge_group_gate is not None:
            attn_edge_attr = self.edge_group_gate(attn_edge_attr, edge_vector_scalar)
        if self.attention_adapter is not None:
            attn_edge_attr = attn_edge_attr + self._maybe_drop_geometry_delta(
                self.attention_adapter(edge_vector_scalar)
            )
        attn_edge_attr = self._apply_confidence_dropout(attn_edge_attr, edge_vector_scalar)

        dh = self.attention(h_V, edge_index, attn_edge_attr)
        if self.node_adapter is not None:
            dh = dh + self._maybe_drop_geometry_delta(
                self.node_adapter(h_V, edge_index, h_E, edge_vector_scalar)
            )
        if self.gvp_lite is not None:
            dh = dh + self._maybe_drop_geometry_delta(
                self.gvp_lite(h_V, node_vectors, edge_index, h_E, edge_vectors)
            )

        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.ffn(h_V)
        if self.ffn_film is not None:
            _, dst = edge_index.long()
            node_geom = scatter_mean(edge_vector_scalar, dst, dim=0, dim_size=h_V.size(0))
            dh = self.ffn_film(node_geom, dh)
        h_V = self.norm[1](h_V + self.dropout(dh))

        edge_update_input = h_E
        if self.edge_adapter is not None:
            edge_update_input = edge_update_input + self._maybe_drop_geometry_delta(
                self.edge_adapter(edge_vector_scalar)
            )
        if self.coordinate_adapter is not None:
            edge_update_input = edge_update_input + self._maybe_drop_geometry_delta(
                self.coordinate_adapter(h_V, edge_index, h_E, edge_vector_scalar)
            )
        next_h_E = self.edge_update(
            h_V,
            edge_index,
            edge_update_input,
            edge_vector_scalar if self.edge_update.geom_dim > 0 else None,
        )
        if self.edge_memory is not None and self.edge_memory_gate_logit is not None:
            mem = self.edge_memory(next_h_E, h_E)
            next_h_E = next_h_E + torch.sigmoid(self.edge_memory_gate_logit) * (mem - next_h_E)
        h_E = next_h_E
        h_V = self.context(h_V, batch_id)
        if self.geo_context_adapter is not None:
            h_V = h_V + self._maybe_drop_geometry_delta(
                self.geo_context_adapter(edge_index, edge_vector_scalar, h_V.size(0))
            )
        return h_V, h_E


class _PoMoGGraphEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        seq_in: bool = False,
        num_layers: int = 4,
        drop_rate: float = 0.2,
        hidden_vector_dim: int = 16,
        geometry_use_attention_adapter: bool = False,
        geometry_use_edge_adapter: bool = False,
        geometry_use_node_adapter: bool = False,
        geometry_use_gvp_lite: bool = False,
        geometry_use_edgemlp_geom: bool = False,
        geometry_use_confidence_dropout: bool = False,
        geometry_use_ffn_film: bool = False,
        geometry_use_edge_group_gate: bool = False,
        geometry_use_edge_memory: bool = False,
        geometry_use_coordinate_adapter: bool = False,
        geometry_use_distance_drift: bool = False,
        geometry_use_geo_context: bool = False,
        geometry_attention_gate_init: float = -3.0,
        geometry_edge_gate_init: float = -3.0,
        geometry_node_gate_init: float = -3.0,
        geometry_gvp_gate_init: float = -3.0,
        geometry_geom_gate_init: float = -3.0,
        geometry_confidence_dropout_base: float = 0.03,
        geometry_confidence_dropout_alpha: float = 0.12,
        geometry_confidence_dropout_max: float = 0.20,
        geometry_branch_dropout: float = 0.0,
        geometry_num_message_gvps: int = 1,
        geometry_num_ff_gvps: int = 1,
        geometry_gvp_lite_start_layer: int = 0,
        geometry_vector_gate: bool = True,
        edge_vector_feature_dim: int = _BASE_VECTOR_EDGE_FEATURE_DIM,
        edge_vector_channels: int = _EDGE_VECTOR_CHANNELS,
    ) -> None:
        super().__init__()
        self.seq_in = seq_in
        if seq_in:
            self.W_s = nn.Embedding(21, 20)
            node_in_dim += 20

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_e = nn.Linear(hidden_dim, hidden_dim)
        gvp_start = max(0, int(geometry_gvp_lite_start_layer))
        self.layers = nn.ModuleList(
            _PoMoGGNNLayer(
                hidden_dim=hidden_dim,
                dropout=drop_rate,
                num_heads=4,
                hidden_vector_dim=hidden_vector_dim,
                geometry_use_attention_adapter=geometry_use_attention_adapter,
                geometry_use_edge_adapter=geometry_use_edge_adapter,
                geometry_use_node_adapter=geometry_use_node_adapter,
                geometry_use_gvp_lite=bool(geometry_use_gvp_lite) and layer_idx >= gvp_start,
                geometry_use_edgemlp_geom=geometry_use_edgemlp_geom,
                geometry_use_confidence_dropout=geometry_use_confidence_dropout,
                geometry_use_ffn_film=geometry_use_ffn_film,
                geometry_use_edge_group_gate=geometry_use_edge_group_gate,
                geometry_use_edge_memory=geometry_use_edge_memory,
                geometry_use_coordinate_adapter=geometry_use_coordinate_adapter,
                geometry_use_distance_drift=geometry_use_distance_drift,
                geometry_use_geo_context=geometry_use_geo_context,
                geometry_attention_gate_init=geometry_attention_gate_init,
                geometry_edge_gate_init=geometry_edge_gate_init,
                geometry_node_gate_init=geometry_node_gate_init,
                geometry_gvp_gate_init=geometry_gvp_gate_init,
                geometry_geom_gate_init=geometry_geom_gate_init,
                geometry_confidence_dropout_base=geometry_confidence_dropout_base,
                geometry_confidence_dropout_alpha=geometry_confidence_dropout_alpha,
                geometry_confidence_dropout_max=geometry_confidence_dropout_max,
                geometry_branch_dropout=geometry_branch_dropout,
                geometry_num_message_gvps=geometry_num_message_gvps,
                geometry_num_ff_gvps=geometry_num_ff_gvps,
                geometry_vector_gate=geometry_vector_gate,
                edge_vector_feature_dim=edge_vector_feature_dim,
                edge_vector_channels=edge_vector_channels,
            )
            for layer_idx in range(num_layers)
        )

    def forward(
        self,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        h_E: torch.Tensor,
        seq: torch.Tensor | None,
        batch_id: torch.Tensor,
        node_vectors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_vector_scalar: torch.Tensor,
    ) -> torch.Tensor:
        if self.seq_in and seq is not None:
            seq_emb = self.W_s(seq)
            h_V = torch.cat([h_V, seq_emb], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(
                h_V,
                edge_index,
                h_E,
                batch_id,
                node_vectors,
                edge_vectors,
                edge_vector_scalar,
            )
        return h_V


class PoMoG(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 1024 + 9,
        edge_input_dim: int = 450,
        hidden_dim: int = 128,
        hidden_vector_dim: int = 16,
        num_layers: int = 4,
        dropout: float = 0.2,
        augment_eps: float = 0.1,
        geometry_use_attention_adapter: bool = False,
        geometry_use_edge_adapter: bool = True,
        geometry_use_node_adapter: bool = False,
        geometry_use_gvp_lite: bool = False,
        geometry_use_edgemlp_geom: bool = False,
        geometry_use_confidence_dropout: bool = False,
        geometry_use_ffn_film: bool = False,
        geometry_use_sidechain_scalar: bool = False,
        geometry_use_edge_group_gate: bool = False,
        geometry_use_edge_memory: bool = False,
        geometry_use_coordinate_adapter: bool = False,
        geometry_use_distance_drift: bool = False,
        geometry_use_geo_context: bool = False,
        geometry_use_local_head: bool = False,
        geometry_use_motif_pool_head: bool = False,
        geometry_motif_pool_steps: int = 1,
        geometry_use_multiradius: bool = False,
        geometry_multiradius_values: tuple[float, ...] | list[float] = (6.0, 8.0, 12.0),
        geometry_use_sequence_edges: bool = False,
        geometry_sequence_edge_span: int = 1,
        geometry_attention_gate_init: float = -3.0,
        geometry_edge_gate_init: float = -3.0,
        geometry_node_gate_init: float = -3.0,
        geometry_gvp_gate_init: float = -3.0,
        geometry_geom_gate_init: float = -3.0,
        geometry_confidence_dropout_base: float = 0.03,
        geometry_confidence_dropout_alpha: float = 0.12,
        geometry_confidence_dropout_max: float = 0.20,
        geometry_branch_dropout: float = 0.0,
        geometry_num_message_gvps: int = 1,
        geometry_num_ff_gvps: int = 1,
        geometry_gvp_lite_start_layer: int = 0,
        geometry_vector_gate: bool = True,
        geometry_edge_vector_mode: str = "ca",
    ) -> None:
        super().__init__()
        self.augment_eps = float(augment_eps)
        self.use_sidechain_scalar = bool(geometry_use_sidechain_scalar)
        self.use_distance_drift = bool(geometry_use_distance_drift)
        self.use_multiradius = bool(geometry_use_multiradius)
        self.multiradius_values = tuple(float(v) for v in geometry_multiradius_values)
        self.use_sequence_edges = bool(geometry_use_sequence_edges)
        self.sequence_edge_span = int(geometry_sequence_edge_span)
        self.use_local_head = bool(geometry_use_local_head)
        self.use_motif_pool_head = bool(geometry_use_motif_pool_head)
        self.motif_pool_steps = max(1, int(geometry_motif_pool_steps))
        if geometry_edge_vector_mode not in {"ca", "atom_pair_8"}:
            raise ValueError(f"unsupported geometry_edge_vector_mode: {geometry_edge_vector_mode!r}")
        self.edge_vector_mode = str(geometry_edge_vector_mode)
        edge_vector_channels = (
            len(_MULTIVECTOR_ATOM_PAIRS) if self.edge_vector_mode == "atom_pair_8" else _EDGE_VECTOR_CHANNELS
        )
        edge_vector_feature_dim = _edge_vector_feature_dim(
            self.use_sidechain_scalar,
            self.use_distance_drift,
        )
        total_node_dim = int(node_input_dim) + _GEO_NODE_DIM
        self.encoder = _PoMoGGraphEncoder(
            node_in_dim=total_node_dim,
            edge_in_dim=int(edge_input_dim),
            hidden_dim=int(hidden_dim),
            seq_in=False,
            num_layers=int(num_layers),
            drop_rate=float(dropout),
            hidden_vector_dim=int(hidden_vector_dim),
            geometry_use_attention_adapter=bool(geometry_use_attention_adapter),
            geometry_use_edge_adapter=bool(geometry_use_edge_adapter),
            geometry_use_node_adapter=bool(geometry_use_node_adapter),
            geometry_use_gvp_lite=bool(geometry_use_gvp_lite),
            geometry_use_edgemlp_geom=bool(geometry_use_edgemlp_geom),
            geometry_use_confidence_dropout=bool(geometry_use_confidence_dropout),
            geometry_use_ffn_film=bool(geometry_use_ffn_film),
            geometry_use_edge_group_gate=bool(geometry_use_edge_group_gate),
            geometry_use_edge_memory=bool(geometry_use_edge_memory),
            geometry_use_coordinate_adapter=bool(geometry_use_coordinate_adapter),
            geometry_use_distance_drift=bool(geometry_use_distance_drift),
            geometry_use_geo_context=bool(geometry_use_geo_context),
            geometry_attention_gate_init=float(geometry_attention_gate_init),
            geometry_edge_gate_init=float(geometry_edge_gate_init),
            geometry_node_gate_init=float(geometry_node_gate_init),
            geometry_gvp_gate_init=float(geometry_gvp_gate_init),
            geometry_geom_gate_init=float(geometry_geom_gate_init),
            geometry_confidence_dropout_base=float(geometry_confidence_dropout_base),
            geometry_confidence_dropout_alpha=float(geometry_confidence_dropout_alpha),
            geometry_confidence_dropout_max=float(geometry_confidence_dropout_max),
            geometry_branch_dropout=float(geometry_branch_dropout),
            geometry_num_message_gvps=int(geometry_num_message_gvps),
            geometry_num_ff_gvps=int(geometry_num_ff_gvps),
            geometry_gvp_lite_start_layer=int(geometry_gvp_lite_start_layer),
            geometry_vector_gate=bool(geometry_vector_gate),
            edge_vector_feature_dim=edge_vector_feature_dim,
            edge_vector_channels=edge_vector_channels,
        )
        head_multiplier = 1
        if self.use_local_head:
            head_multiplier += 1
        if self.use_motif_pool_head:
            head_multiplier += 1
        self.fc1 = nn.Linear(hidden_dim * head_multiplier, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_edge_index(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        batch_id: torch.Tensor,
    ) -> torch.Tensor:
        chunks = [edge_index.long()]
        if self.use_multiradius:
            ca = X[:, _CA_COORD_INDEX].float()
            for radius in self.multiradius_values:
                chunks.append(
                    radius_graph(
                        ca,
                        r=float(radius),
                        batch=batch_id.long(),
                        loop=True,
                        max_num_neighbors=1000,
                    )
                )
        if self.use_sequence_edges:
            chunks.append(_build_sequence_edges(batch_id.long(), self.sequence_edge_span))
        if len(chunks) == 1:
            return chunks[0]
        return _unique_edges(torch.cat(chunks, dim=1))

    def _build_head_features(self, h_V: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        features = [h_V]
        if self.use_local_head or self.use_motif_pool_head:
            src, dst = edge_index.long()
        if self.use_local_head:
            local = scatter_mean(h_V[src], dst, dim=0, dim_size=h_V.size(0))
            features.append(local)
        if self.use_motif_pool_head:
            pooled = h_V
            for _ in range(self.motif_pool_steps):
                pooled = scatter_mean(pooled[src], dst, dim=0, dim_size=h_V.size(0))
            features.append(pooled)
        return torch.cat(features, dim=-1) if len(features) > 1 else h_V

    def forward(
        self,
        X: torch.Tensor,
        h_V: torch.Tensor,
        edge_index: torch.Tensor,
        seq: torch.Tensor,
        batch_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

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
        h_V = self._build_head_features(h_V, edge_index)
        out = F.elu(self.fc1(h_V))
        out = self.fc2(out)
        return out
