from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_mean


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor, edge_index: torch.Tensor, h_E: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h_EV = torch.cat([h_V[src], h_E, h_V[dst]], dim=-1)
        h_msg = self.W12(self.act(self.W11(h_EV)))
        return self.norm(h_E + self.dropout(h_msg))


class Context(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

    def forward(self, h_V: torch.Tensor, batch_id: torch.Tensor) -> torch.Tensor:
        c_V = scatter_mean(h_V, batch_id, dim=0)
        return h_V * self.gate(c_V[batch_id])


class GNNLayer(nn.Module):
    def __init__(self, num_hidden: int, dropout: float = 0.2, num_heads: int = 4) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.attention = TransformerConv(
            in_channels=num_hidden,
            out_channels=num_hidden // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=num_hidden,
            root_weight=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V: torch.Tensor, edge_index: torch.Tensor, h_E: torch.Tensor, batch_id: torch.Tensor):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.ffn(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        h_E = self.edge_update(h_V, edge_index, h_E)
        h_V = self.context(h_V, batch_id)
        return h_V, h_E


class Graph_encoder(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        seq_in: bool = False,
        num_layers: int = 4,
        drop_rate: float = 0.2,
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
        self.layers = nn.ModuleList(
            GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers)
        )

    def forward(self, h_V: torch.Tensor, edge_index: torch.Tensor, h_E: torch.Tensor, seq: torch.Tensor | None, batch_id: torch.Tensor) -> torch.Tensor:
        if self.seq_in and seq is not None:
            seq_emb = self.W_s(seq)
            h_V = torch.cat([h_V, seq_emb], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        return h_V
