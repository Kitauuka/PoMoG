from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

_AA_VOCAB_SIZE = 21
_AA_UNKNOWN_INDEX = 20
_AA_ONEHOT_DIM = 21
_PHYS_BINARY_DIM = 6
_PHYS_CONT_DIM = 4
_CA_COORD_INDEX = 1
_NODE_VECTOR_CHANNELS = 3
_EDGE_VECTOR_CHANNELS = 1
_SEQ_BUCKET_DIM = 8
_EDGE_TYPE_DIM = 2


def _append_unknown(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values, values.new_zeros(1)], dim=0)


def _normalize_known(values: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(values.max() - values.min(), min=1e-6)
    return (values - values.min()) / denom


_AA_VOLUME_20 = torch.tensor(
    [
        88.6,   # A
        173.4,  # R
        114.1,  # N
        111.1,  # D
        108.5,  # C
        143.8,  # Q
        138.4,  # E
        60.1,   # G
        153.2,  # H
        166.7,  # I
        166.7,  # L
        168.6,  # K
        162.9,  # M
        189.9,  # F
        112.7,  # P
        89.0,   # S
        116.1,  # T
        227.8,  # W
        193.6,  # Y
        140.0,  # V
    ],
    dtype=torch.float32,
)
_AA_MASS_20 = torch.tensor(
    [
        89.09,   # A
        174.20,  # R
        132.12,  # N
        133.10,  # D
        121.15,  # C
        146.15,  # Q
        147.13,  # E
        75.07,   # G
        155.16,  # H
        131.17,  # I
        131.17,  # L
        146.19,  # K
        149.21,  # M
        165.19,  # F
        115.13,  # P
        105.09,  # S
        119.12,  # T
        204.23,  # W
        181.19,  # Y
        117.15,  # V
    ],
    dtype=torch.float32,
)
_AA_SIDECHAIN_PKA_20 = torch.tensor(
    [
        0.00,   # A
        12.48,  # R
        0.00,   # N
        3.65,   # D
        8.18,   # C
        0.00,   # Q
        4.25,   # E
        0.00,   # G
        6.00,   # H
        0.00,   # I
        0.00,   # L
        10.53,  # K
        0.00,   # M
        0.00,   # F
        0.00,   # P
        0.00,   # S
        0.00,   # T
        0.00,   # W
        10.07,  # Y
        0.00,   # V
    ],
    dtype=torch.float32,
)
_AA_HYDROPATHY_20 = torch.tensor(
    [
        1.8,   # A
        -4.5,  # R
        -3.5,  # N
        -3.5,  # D
        2.5,   # C
        -3.5,  # Q
        -3.5,  # E
        -0.4,  # G
        -3.2,  # H
        4.5,   # I
        3.8,   # L
        -3.9,  # K
        1.9,   # M
        2.8,   # F
        -1.6,  # P
        -0.8,  # S
        -0.7,  # T
        -0.9,  # W
        -1.3,  # Y
        4.2,   # V
    ],
    dtype=torch.float32,
)

_AA_BINARY_FEATURE_TABLE = torch.tensor(
    [
        [0, 0, 0, 0, 0, 1],  # A
        [1, 1, 1, 0, 0, 0],  # R
        [1, 0, 0, 0, 0, 0],  # N
        [1, 1, 0, 1, 0, 0],  # D
        [1, 0, 0, 0, 0, 1],  # C
        [1, 0, 0, 0, 0, 0],  # Q
        [1, 1, 0, 1, 0, 0],  # E
        [0, 0, 0, 0, 0, 0],  # G
        [1, 1, 1, 0, 1, 0],  # H
        [0, 0, 0, 0, 0, 1],  # I
        [0, 0, 0, 0, 0, 1],  # L
        [1, 1, 1, 0, 0, 0],  # K
        [0, 0, 0, 0, 0, 1],  # M
        [0, 0, 0, 0, 1, 1],  # F
        [0, 0, 0, 0, 0, 1],  # P
        [1, 0, 0, 0, 0, 0],  # S
        [1, 0, 0, 0, 0, 0],  # T
        [0, 0, 0, 0, 1, 1],  # W
        [1, 0, 0, 0, 1, 1],  # Y
        [0, 0, 0, 0, 0, 1],  # V
        [0, 0, 0, 0, 0, 0],  # unknown
    ],
    dtype=torch.float32,
)
_AA_CONT_FEATURE_TABLE = torch.stack(
    [
        _append_unknown(_normalize_known(_AA_VOLUME_20)),
        _append_unknown(_normalize_known(_AA_MASS_20)),
        _append_unknown(_normalize_known(_AA_SIDECHAIN_PKA_20)),
        _append_unknown(_normalize_known(_AA_HYDROPATHY_20)),
    ],
    dim=-1,
)


def _norm_no_nan(
    x: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True,
) -> torch.Tensor:
    out = torch.clamp(torch.sum(x.pow(2), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / _norm_no_nan(x, dim=-1, keepdim=True, eps=eps)


def _tuple_sum(*items: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    scalars, vectors = zip(*items)
    return sum(scalars), sum(vectors)


def _tuple_cat(*items: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    scalars, vectors = zip(*items)
    return torch.cat(scalars, dim=-1), torch.cat(vectors, dim=-2)


def _rbf_expand(
    distances: torch.Tensor,
    *,
    d_min: float = 0.0,
    d_max: float = 20.0,
    d_count: int = 16,
) -> torch.Tensor:
    centers = torch.linspace(d_min, d_max, d_count, device=distances.device, dtype=distances.dtype)
    width = max((d_max - d_min) / max(d_count, 1), 1e-6)
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2.0 * width ** 2))


def _build_backbone_edges(batch_id: torch.Tensor) -> torch.Tensor:
    if batch_id.numel() <= 1:
        return batch_id.new_empty((2, 0))

    same_graph = batch_id[:-1] == batch_id[1:]
    if not torch.any(same_graph):
        return batch_id.new_empty((2, 0))

    left = torch.arange(batch_id.numel() - 1, device=batch_id.device, dtype=batch_id.dtype)[same_graph]
    right = left + 1
    return torch.stack(
        [torch.cat([left, right], dim=0), torch.cat([right, left], dim=0)],
        dim=0,
    )


def _sequence_separation_bucket(seq_sep: torch.Tensor) -> torch.Tensor:
    seq_sep = seq_sep.long()
    bucket = torch.full_like(seq_sep, _SEQ_BUCKET_DIM - 1)
    bucket = torch.where(seq_sep <= 20, torch.full_like(seq_sep, 6), bucket)
    bucket = torch.where(seq_sep <= 10, torch.full_like(seq_sep, 5), bucket)
    bucket = torch.where(seq_sep <= 5, torch.full_like(seq_sep, 4), bucket)
    bucket = torch.where(seq_sep <= 3, seq_sep, bucket)
    return bucket


def _augment_edges(
    edge_index: torch.Tensor,
    batch_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = int(batch_id.size(0))
    if num_nodes == 0:
        return edge_index.long(), edge_index.new_zeros((0, _EDGE_TYPE_DIM), dtype=torch.float32)

    backbone_edges = _build_backbone_edges(batch_id)
    combined_edges = edge_index.long()
    if backbone_edges.numel() > 0:
        combined_edges = torch.cat([combined_edges, backbone_edges.long()], dim=1)

    if combined_edges.numel() == 0:
        return combined_edges, edge_index.new_zeros((0, _EDGE_TYPE_DIM), dtype=torch.float32)

    edge_keys = combined_edges[0].long() * num_nodes + combined_edges[1].long()
    unique_keys, inverse = torch.unique(edge_keys, sorted=True, return_inverse=True)
    aug_edge_index = torch.stack(
        [
            unique_keys.div(num_nodes, rounding_mode="floor"),
            unique_keys.remainder(num_nodes),
        ],
        dim=0,
    ).long()

    spatial_flag = torch.zeros(unique_keys.size(0), device=edge_index.device, dtype=torch.float32)
    backbone_flag = torch.zeros_like(spatial_flag)

    if edge_index.numel() > 0:
        one_spatial = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
        spatial_flag.index_add_(0, inverse[: edge_index.size(1)], one_spatial)

    if backbone_edges.numel() > 0:
        one_backbone = torch.ones(backbone_edges.size(1), device=edge_index.device, dtype=torch.float32)
        backbone_flag.index_add_(0, inverse[edge_index.size(1) :], one_backbone)

    edge_type_attr = torch.stack(
        [
            backbone_flag.clamp_(max=1.0),
            spatial_flag.clamp_(max=1.0),
        ],
        dim=-1,
    )
    return aug_edge_index, edge_type_attr


class ResidueNodeAugmenter(nn.Module):
    def __init__(
        self,
        aa_feature_mode: str = "embedding",
        aa_embedding_dim: int = 8,
        use_physchem_binary: bool = True,
        use_physchem_continuous: bool = True,
    ) -> None:
        super().__init__()
        mode = str(aa_feature_mode).lower()
        if mode not in {"none", "onehot", "embedding", "embed"}:
            raise ValueError(f"unsupported aa_feature_mode: {aa_feature_mode}")

        self.aa_feature_mode = "embedding" if mode == "embed" else mode
        self.aa_embedding_dim = int(aa_embedding_dim)
        self.use_physchem_binary = bool(use_physchem_binary)
        self.use_physchem_continuous = bool(use_physchem_continuous)
        self.register_buffer("physchem_binary_table", _AA_BINARY_FEATURE_TABLE.clone())
        self.register_buffer("physchem_cont_table", _AA_CONT_FEATURE_TABLE.clone())

        if self.aa_feature_mode == "embedding":
            self.aa_embedding = nn.Embedding(_AA_VOCAB_SIZE, self.aa_embedding_dim)
        else:
            self.aa_embedding = None

        extra_dim = 0
        if self.aa_feature_mode == "onehot":
            extra_dim += _AA_ONEHOT_DIM
        elif self.aa_feature_mode == "embedding":
            extra_dim += self.aa_embedding_dim
        if self.use_physchem_binary:
            extra_dim += _PHYS_BINARY_DIM
        if self.use_physchem_continuous:
            extra_dim += _PHYS_CONT_DIM
        self.output_dim = extra_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        safe_seq = seq.long().clamp(min=0, max=_AA_UNKNOWN_INDEX)
        features: list[torch.Tensor] = []

        if self.aa_feature_mode == "onehot":
            features.append(F.one_hot(safe_seq, num_classes=_AA_ONEHOT_DIM).float())
        elif self.aa_feature_mode == "embedding":
            assert self.aa_embedding is not None
            features.append(self.aa_embedding(safe_seq))

        if self.use_physchem_binary:
            features.append(self.physchem_binary_table[safe_seq])
        if self.use_physchem_continuous:
            features.append(self.physchem_cont_table[safe_seq])

        if not features:
            return self.physchem_binary_table.new_zeros((safe_seq.size(0), 0))
        return torch.cat(features, dim=-1)


class _VectorDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_rate <= 0 or x.size(-2) == 0:
            return x
        keep_rate = 1.0 - self.drop_rate
        mask = torch.bernoulli(x.new_full(x.shape[:-1], keep_rate)).unsqueeze(-1)
        return mask * x / max(keep_rate, 1e-6)


class GVPDropout(nn.Module):
    def __init__(self, drop_rate: float) -> None:
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate)
        self.vector_dropout = _VectorDropout(drop_rate)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        s, v = x
        return self.scalar_dropout(s), self.vector_dropout(v)


class GVPLayerNorm(nn.Module):
    def __init__(self, dims: tuple[int, int]) -> None:
        super().__init__()
        self.scalar_dim, self.vector_dim = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dim)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        s, v = x
        s = self.scalar_norm(s)
        if self.vector_dim == 0:
            return s, v

        v_norm_sq = _norm_no_nan(v, dim=-1, keepdim=True, sqrt=False)
        denom = torch.sqrt(torch.mean(v_norm_sq, dim=-2, keepdim=True))
        return s, v / denom


class GVP(nn.Module):
    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dims: tuple[int, int],
        *,
        h_dim: int | None = None,
        activations: tuple | None = (F.silu, torch.sigmoid),
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = bool(vector_gate)
        self.scalar_act, self.vector_act = activations if activations is not None else (None, None)

        if self.vi > 0:
            self.h_dim = int(h_dim or max(self.vi, self.vo, 1))
            self.vector_hidden = nn.Linear(self.vi, self.h_dim, bias=False)
            self.scalar_out = nn.Linear(self.si + self.h_dim, self.so)
            self.vector_out = nn.Linear(self.h_dim, self.vo, bias=False) if self.vo > 0 else None
            self.vector_gate_proj = nn.Linear(self.so, self.vo) if self.vector_gate and self.vo > 0 else None
        else:
            self.h_dim = 0
            self.vector_hidden = None
            self.scalar_out = nn.Linear(self.si, self.so)
            self.vector_out = None
            self.vector_gate_proj = None

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        s, v = x
        if self.vi > 0:
            v_hidden = self.vector_hidden(v.transpose(-1, -2))
            v_hidden_norm = _norm_no_nan(v_hidden, dim=-2)
            s_out = self.scalar_out(torch.cat([s, v_hidden_norm], dim=-1))
            if self.vo > 0:
                v_out = self.vector_out(v_hidden).transpose(-1, -2)
                if self.vector_gate_proj is not None:
                    gate_input = self.vector_act(s_out) if self.vector_act is not None else s_out
                    gate = torch.sigmoid(self.vector_gate_proj(gate_input)).unsqueeze(-1)
                    v_out = v_out * gate
                elif self.vector_act is not None:
                    v_scale = self.vector_act(_norm_no_nan(v_out, dim=-1, keepdim=True))
                    v_out = v_out * v_scale
            else:
                v_out = v.new_zeros(v.shape[:-2] + (0, 3))
        else:
            s_out = self.scalar_out(s)
            v_out = s.new_zeros(s.shape[:-1] + (self.vo, 3))

        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)
        return s_out, v_out


def _build_gvp_stack(
    in_dims: tuple[int, int],
    out_dims: tuple[int, int],
    *,
    num_layers: int,
    vector_gate: bool,
    final_activations: tuple | None = (None, None),
) -> nn.Sequential:
    if num_layers <= 1:
        return nn.Sequential(
            GVP(
                in_dims,
                out_dims,
                activations=final_activations,
                vector_gate=vector_gate,
            )
        )

    layers: list[nn.Module] = [
        GVP(in_dims, out_dims, activations=(F.silu, torch.sigmoid), vector_gate=vector_gate)
    ]
    for _ in range(num_layers - 2):
        layers.append(GVP(out_dims, out_dims, activations=(F.silu, torch.sigmoid), vector_gate=vector_gate))
    layers.append(GVP(out_dims, out_dims, activations=final_activations, vector_gate=vector_gate))
    return nn.Sequential(*layers)


class GVPConvLayer(nn.Module):
    def __init__(
        self,
        node_dims: tuple[int, int],
        edge_dims: tuple[int, int],
        *,
        num_message_gvps: int = 3,
        num_ff_gvps: int = 2,
        dropout: float = 0.2,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        node_scalar_dim, node_vector_dim = node_dims
        edge_scalar_dim, edge_vector_dim = edge_dims
        message_in_dims = (
            2 * node_scalar_dim + edge_scalar_dim,
            2 * node_vector_dim + edge_vector_dim,
        )
        ff_hidden_dims = (4 * node_scalar_dim, 2 * node_vector_dim)

        self.message_func = _build_gvp_stack(
            message_in_dims,
            node_dims,
            num_layers=num_message_gvps,
            vector_gate=vector_gate,
            final_activations=(None, None),
        )
        self.ff_func = _build_gvp_stack(
            node_dims,
            ff_hidden_dims,
            num_layers=max(num_ff_gvps - 1, 1),
            vector_gate=vector_gate,
        )
        self.ff_out = GVP(
            ff_hidden_dims,
            node_dims,
            activations=(None, None),
            vector_gate=vector_gate,
        )
        self.norm1 = GVPLayerNorm(node_dims)
        self.norm2 = GVPLayerNorm(node_dims)
        self.drop1 = GVPDropout(dropout)
        self.drop2 = GVPDropout(dropout)

    def _aggregate_messages(
        self,
        node_state: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_s, node_v = node_state
        edge_s, edge_v = edge_state
        src, dst = edge_index
        message_input = _tuple_cat(
            (node_s[src], node_v[src]),
            (edge_s, edge_v),
            (node_s[dst], node_v[dst]),
        )
        msg_s, msg_v = self.message_func(message_input)
        agg_s = scatter_mean(msg_s, dst, dim=0, dim_size=node_s.size(0))
        agg_v = scatter_mean(msg_v, dst, dim=0, dim_size=node_s.size(0))
        return agg_s, agg_v

    def forward(
        self,
        node_state: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self._aggregate_messages(node_state, edge_index, edge_state)
        node_state = self.norm1(_tuple_sum(node_state, self.drop1(delta)))
        delta = self.ff_out(self.ff_func(node_state))
        node_state = self.norm2(_tuple_sum(node_state, self.drop2(delta)))
        return node_state


class GVPGNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        hidden_vector_dim: int,
        num_layers: int = 4,
        dropout: float = 0.2,
        num_message_gvps: int = 3,
        num_ff_gvps: int = 2,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = (int(hidden_dim), int(hidden_vector_dim))
        node_in_dims = (int(node_in_dim), _NODE_VECTOR_CHANNELS)
        edge_in_dims = (int(edge_in_dim), _EDGE_VECTOR_CHANNELS)

        self.node_embed = nn.Sequential(
            GVPLayerNorm(node_in_dims),
            GVP(node_in_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate),
        )
        self.edge_embed = nn.Sequential(
            GVPLayerNorm(edge_in_dims),
            GVP(edge_in_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate),
        )
        self.layers = nn.ModuleList(
            GVPConvLayer(
                hidden_dims,
                hidden_dims,
                num_message_gvps=num_message_gvps,
                num_ff_gvps=num_ff_gvps,
                dropout=dropout,
                vector_gate=vector_gate,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        node_state: tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_state = self.node_embed(node_state)
        edge_state = self.edge_embed(edge_state)
        for layer in self.layers:
            node_state = layer(node_state, edge_index, edge_state)
        return node_state


class PoMoG(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 1024 + 9,
        edge_input_dim: int = 27,
        hidden_dim: int = 128,
        hidden_vector_dim: int = 16,
        num_layers: int = 4,
        dropout: float = 0.2,
        augment_eps: float = 0.1,
        edge_rbf_dim: int = 16,
        edge_rbf_max: float = 20.0,
        aa_feature_mode: str = "embedding",
        aa_embedding_dim: int = 8,
        use_physchem_binary: bool = True,
        use_physchem_continuous: bool = True,
        num_message_gvps: int = 3,
        num_ff_gvps: int = 2,
        vector_gate: bool = True,
    ) -> None:
        super().__init__()
        expected_edge_dim = 1 + int(edge_rbf_dim) + _SEQ_BUCKET_DIM + _EDGE_TYPE_DIM
        if int(edge_input_dim) != expected_edge_dim:
            raise ValueError(
                f"edge_input_dim={edge_input_dim} does not match expected GVP edge scalar dim {expected_edge_dim}"
            )

        self.augment_eps = float(augment_eps)
        self.edge_rbf_dim = int(edge_rbf_dim)
        self.edge_rbf_max = float(edge_rbf_max)
        self.node_augmenter = ResidueNodeAugmenter(
            aa_feature_mode=aa_feature_mode,
            aa_embedding_dim=aa_embedding_dim,
            use_physchem_binary=use_physchem_binary,
            use_physchem_continuous=use_physchem_continuous,
        )

        total_node_dim = int(node_input_dim) + int(self.node_augmenter.output_dim)
        self.encoder = GVPGNNEncoder(
            node_in_dim=total_node_dim,
            edge_in_dim=expected_edge_dim,
            hidden_dim=hidden_dim,
            hidden_vector_dim=hidden_vector_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_message_gvps=num_message_gvps,
            num_ff_gvps=num_ff_gvps,
            vector_gate=vector_gate,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def _build_node_inputs(
        self,
        X: torch.Tensor,
        node_scalar: torch.Tensor,
        seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_aug = self.node_augmenter(seq)
        if node_aug.size(-1) > 0:
            node_scalar = torch.cat([node_scalar.float(), node_aug], dim=-1)
        else:
            node_scalar = node_scalar.float()

        ca = X[:, _CA_COORD_INDEX].float()
        node_vector = torch.stack(
            [
                X[:, 0].float() - ca,
                X[:, 2].float() - ca,
                X[:, 3].float() - ca,
            ],
            dim=1,
        )
        return node_scalar, node_vector

    def _build_edge_inputs(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ca = X[:, _CA_COORD_INDEX].float()
        src, dst = edge_index
        ca_diff = ca[dst] - ca[src]
        distances = _norm_no_nan(ca_diff, dim=-1)
        seq_sep = (dst - src).abs()
        seq_bucket = F.one_hot(
            _sequence_separation_bucket(seq_sep),
            num_classes=_SEQ_BUCKET_DIM,
        ).float()

        edge_scalar = torch.cat(
            [
                distances.unsqueeze(-1),
                _rbf_expand(
                    distances,
                    d_min=0.0,
                    d_max=self.edge_rbf_max,
                    d_count=self.edge_rbf_dim,
                ),
                seq_bucket,
                edge_type_attr.float(),
            ],
            dim=-1,
        )
        edge_vector = _safe_normalize(ca_diff).unsqueeze(1)
        return edge_scalar, edge_vector

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

        aug_edge_index, edge_type_attr = _augment_edges(edge_index.long(), batch_id.long())
        node_state = self._build_node_inputs(X.float(), h_V, seq)
        edge_state = self._build_edge_inputs(X.float(), aug_edge_index, edge_type_attr)
        scalar_state, _ = self.encoder(node_state, aug_edge_index, edge_state)

        out = F.elu(self.fc1(scalar_state))
        out = self.fc2(out)
        return out
