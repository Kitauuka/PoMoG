"""
 ——  PoMoG/Features/features.py 。

 X([L, 5, 3])  edge_index([2, E]) 
：

   [L, 184]:
    _get_angle()         → 12   /（φ ψ ω ）
    _get_distance() N    → 160   10  × 16 RBF 
    _get_direction_orientation()  → 12   

   [E, 450]:
    _positional_embeddings()  → 16   
    _quaternions()             → 4    （）
    _get_distance()     → 400   25  × 16 RBF 
    _get_direction_orientation()  → 32   

  ：12+160+12 = 184（）  16+4+400+32 = 452 ≈ 450（， 450）
"""
from __future__ import annotations

import torch
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
#  → 
#  PoMoG ，20  0‒19，
#  (nn.Embedding(21, embed_dim))
# ──────────────────────────────────────────────────────────────────────────────
LETTER_TO_NUM = {
    "C": 4, "D": 3, "S": 15, "Q": 5, "K": 11, "I": 9,
    "P": 14, "T": 16, "F": 13, "A": 0, "G": 7, "H": 8,
    "E": 6, "L": 10, "R": 1, "W": 17, "V": 19, "N": 2,
    "Y": 18, "M": 12,
}

# ： LETTER_TO_NUM  NUM_TO_LETTER（）
NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}


def get_geo_feat(
    X: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ，。

    Args:
        X          : [L, 5, 3]   5 （N, Cα, C, O, R）
        edge_index : [2, E]     ， 0 ， 1 

    Returns:
        node_feat : [L, 184]   （ 12 +  160 +  12）
        edge_feat : [E, 450]   （ 16 +  4 +  400 +  32）
    """
    X = X.float()  #  float32， float64 
    E = edge_index.shape[1]  # 

    #  Cα （X[:, 1]）
    X_ca = X[:, 1]  # [L, 3]

    # ──  ──────────────────────────────────────────────
    angle_feat = _get_angle(X)                          # [L, 12]  φ/ψ/ω 
    node_dist, edge_dist = _get_distance(X, edge_index) # [L, 160], [E, 400]
    node_dir, edge_dir, q = _get_direction_orientation(X, edge_index)  # [L,12], [E,32], [E,4]

    node_feat = torch.cat([angle_feat, node_dist, node_dir], dim=-1)   # [L, 184]

    # ──  ────────────────────────────────────────────────
    pos_emb = _positional_embeddings(edge_index, num_embeddings=8)    # [E, 16]  sin/cos 
    # ：_positional_embeddings(num_embeddings=8)  2×8=16 

    edge_feat = torch.cat([pos_emb, q, edge_dist, edge_dir], dim=-1)  # [E, 452]
    edge_feat = edge_feat[:, :450]  #  450  PoMoG 

    return node_feat, edge_feat


# ──────────────────────────────────────────────────────────────────────────────
# 
# ──────────────────────────────────────────────────────────────────────────────

def _positional_embeddings(
    edge_index: torch.Tensor,
    num_embeddings: int = 8,
    period_range: tuple[int, int] = (2, 1000),
) -> torch.Tensor:
    """
    ，。

    ：（）。
    /， Transformer
    。

    ：
      d = i - j （ - ）
      PE[k] = sin(d / base^(2k/D))  （）
      PE[k] = cos(d / base^(2k/D))  （）
      D = num_embeddings, base  period_range 

    Args:
        edge_index    : [2, E]    
        num_embeddings: ， = 2 × num_embeddings
        period_range  : ，

    Returns:
        [E, 2 × num_embeddings]
    """
    #  d =  - ，（/）
    d = edge_index[0].float() - edge_index[1].float()  # [E]

    # （， period_range ）
    # ""
    frequency = torch.exp(
        torch.arange(0, num_embeddings, dtype=torch.float, device=edge_index.device)
        * -(np.log(period_range[1] / period_range[0]) / (num_embeddings - 1))
        + np.log(1 / period_range[0])
    )  # [num_embeddings]

    # ： d 
    angles = d.unsqueeze(-1) * frequency.unsqueeze(0)  # [E, num_embeddings]

    # sin + cos ， → 2 × num_embeddings = 16 
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # [E, 16]
    return emb


def _angle_between(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """（）， [..., 3]。"""
    v1n = _normalize(v1, eps)
    v2n = _normalize(v2, eps)
    cosv = torch.clamp((v1n * v2n).sum(dim=-1), -1.0, 1.0)
    return torch.acos(cosv)


def _dihedral(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """（）， [..., 3]。"""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1n = _normalize(b1, eps)
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _get_angle(X: torch.Tensor) -> torch.Tensor:
    """
    （12 ）。

     6 （phi/psi/omega + 3 ） cos/sin， [L, 12]。
    """
    L = X.shape[0]
    device = X.device

    # 
    if L == 0:
        return torch.zeros((0, 12), dtype=X.dtype, device=device)

    N = X[:, 0]
    CA = X[:, 1]
    C = X[:, 2]

    phi = torch.zeros(L, dtype=X.dtype, device=device)
    psi = torch.zeros(L, dtype=X.dtype, device=device)
    omega = torch.zeros(L, dtype=X.dtype, device=device)

    ang1 = torch.zeros(L, dtype=X.dtype, device=device)  # ∠N-CA-C
    ang2 = torch.zeros(L, dtype=X.dtype, device=device)  # ∠CA-C-N(next)
    ang3 = torch.zeros(L, dtype=X.dtype, device=device)  # ∠C(prev)-N-CA

    # 
    if L >= 2:
        phi[1:] = _dihedral(C[:-1], N[1:], CA[1:], C[1:])
        psi[:-1] = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
        omega[1:] = _dihedral(CA[:-1], C[:-1], N[1:], CA[1:])

    # 
    ang1 = _angle_between(N - CA, C - CA)
    if L >= 2:
        ang2[:-1] = _angle_between(CA[:-1] - C[:-1], N[1:] - C[:-1])
        ang3[1:] = _angle_between(C[:-1] - N[1:], CA[1:] - N[1:])

    angles = torch.stack([phi, psi, omega, ang1, ang2, ang3], dim=-1)  # [L, 6]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)   # [L, 12]

def _rbf(
    D: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    D_count: int = 16,
) -> torch.Tensor:
    """
    （RBF）。

     RBF ？
    - ，"8Å  9Å "
    - RBF  16 （）
    - " 5Å"" 10Å"
    -  μ_k = D_min + k*(D_max-D_min)/(D_count-1)
    - σ ， RBF 

    Args:
        D       : [...] 
        D_min   : （ 0  RBF ）
        D_max   : （0~20 Å ）
        D_count : RBF （）， = D_count

    Returns:
        [..., D_count]   RBF 
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_sigma = (D_max - D_min) / D_count  #  σ

    # ：exp(-(d - μ)² / (2σ²))
    return torch.exp(-((D.unsqueeze(-1) - D_mu) ** 2) / (2 * D_sigma ** 2))


def _get_distance(
    X: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    （）（）， RBF 。

    （）：
       5 （N, Cα, C, O, R） C(5,2) = 10 
      10  × 16 RBF = 160 

    （）：
       5  5 ， 25 （）
      25  × 16 RBF = 400 

    Returns:
        node_dist : [L, 160]
        edge_dist : [E, 400]
    """
    L = X.shape[0]
    # ─ （）─────────────────────────────────────
    #  10 
    atom_pairs_node = [(i, j) for i in range(5) for j in range(i + 1, 5)]  # 10 
    dists_node = []
    for i, j in atom_pairs_node:
        d = torch.norm(X[:, i] - X[:, j], dim=-1)  # [L]
        dists_node.append(_rbf(d).float())           # [L, 16]

    node_dist = torch.cat(dists_node, dim=-1)  # [L, 10×16=160]

    # ─ （）─────────────────────────────────────
    src, dst = edge_index[0], edge_index[1]
    atom_pairs_edge = [(i, j) for i in range(5) for j in range(5)]  # 25 （）
    dists_edge = []
    for i, j in atom_pairs_edge:
        d = torch.norm(X[src, i] - X[dst, j], dim=-1)  # [E]
        dists_edge.append(_rbf(d).float())               # [E, 16]

    edge_dist = torch.cat(dists_edge, dim=-1)  # [E, 25×16=400]

    return node_dist, edge_dist


def _get_direction_orientation(
    X: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ，/（）。

    ， Cα ，
    N→Cα  Cα→C  X/Y/Z ：
      u = Cα - N（）
      n = u × (C - N)（， N/Cα/C ）
      b = n × u（ u ）
     (u, b, n) ，。

    （，12 ）：
      ， Cα→C （3 ）
      + / Cα （ 3 ） →  3×3=9  +  3 

    （，32 ）：
       src  dst  8 （ 4 ）

    （，4 ）：
      src  dst ；
      （），（4 vs 9 ）

    Returns:
        node_dir : [L, 12]
        edge_dir : [E, 32]
        q        : [E, 4]
    """
    X_ca = X[:, 1]  # Cα  [L, 3]
    X_n  = X[:, 0]  # N   [L, 3]
    X_c  = X[:, 2]  # C   [L, 3]

    # ─ （）────────────────────────────
    u = _normalize(X_ca - X_n)   # Cα → N （）[L, 3]
    v = _normalize(X_c  - X_n)   # N  → C               [L, 3]
    n = _normalize(torch.cross(u, v, dim=-1))  # ， u/v 
    b = torch.cross(n, u, dim=-1)              # u/n/b 

    # ：
    # Q.shape = [L, 3, 3]，Q[i]  i 
    Q = torch.stack([u, b, n], dim=-1)  # [L, 3, 3]  ：u, b, n

    # ─  ─────────────────────────────────────────────
    #  i ， Cα 
    #  Cα→C 
    #  12  = 33（Cα→, CαC, Cα→）
    # ：（/） 0 
    L = X.shape[0]
    #  Cα （0 ，）
    X_prev = torch.cat([X_ca[:1], X_ca[:-1]], dim=0)
    X_next = torch.cat([X_ca[1:], X_ca[-1:]], dim=0)

    d_prev = _normalize(X_prev - X_ca)  # → 
    d_next = _normalize(X_next - X_ca)  # → 
    d_ca_c = _normalize(X_c - X_ca)     # Cα → C

    # ：
    # (Q^T @ d)：Q ，Q^T 
    def local_project(d):
        # d: [L, 3]   Q: [L, 3, 3]
        return (Q.transpose(-1, -2) @ d.unsqueeze(-1)).squeeze(-1)  # [L, 3]

    node_dir = torch.cat([
        local_project(d_prev),
        local_project(d_ca_c),
        local_project(d_next),
    ], dim=-1)  # [L, 9]

    #  3 （ 12 ）： b 
    # （b ，）
    node_dir = torch.cat([node_dir, b], dim=-1)  # [L, 12]

    # ─ （）────────────────────────────────────
    # Q_src: src   Q_dst: dst 
    src, dst = edge_index[0], edge_index[1]
    dR = Q[src].transpose(-1, -2) @ Q[dst]  #  [E, 3, 3]
    q = _quaternions(dR)                      # →       [E, 4]

    # ─  ────────────────────────────────────────────────
    #  src→dst ， src （）
    # 8  × 4 （3 + 1 ）= 32 
    atom_pairs = [(1, 1), (0, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1), (0, 2)]
    dirs = []
    for ai, aj in atom_pairs:
        diff = X[dst, aj] - X[src, ai]            # [E, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-8)
        unit = diff / dist                          # 
        #  src 
        local_d = (Q[src].transpose(-1, -2) @ unit.unsqueeze(-1)).squeeze(-1)
        dirs.append(torch.cat([local_d, dist / 20.0], dim=-1))  # [E, 4]

    edge_dir = torch.cat(dirs, dim=-1)  # [E, 32]

    return node_dir, edge_dir, q


def _normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2 ， 0  0。"""
    norm = tensor.norm(dim=-1, keepdim=True).clamp(min=eps)
    return tensor / norm


def _quaternions(R: torch.Tensor) -> torch.Tensor:
    """
     R ([..., 3, 3])  ([..., 4])。

    ：
    -  9 ， 3 （ + ）
    - （Gimbal Lock），
    -  (q0, q1, q2, q3)  4 
    -  |q|=1，

     Shepperd :
      q0² = (1 + tr(R)) / 4
      q1² = (1 + R00 - R11 - R22) / 4  
    """
    # 
    # R[..., i, j]  i  j 
    diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [..., 3]  (R00, R11, R22)
    q0 = 0.5 * torch.sqrt(torch.clamp(1 + diag.sum(-1), min=0)).unsqueeze(-1)  # [..., 1]

    q123 = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],  # 4*q0*q1
        R[..., 0, 2] - R[..., 2, 0],  # 4*q0*q2
        R[..., 1, 0] - R[..., 0, 1],  # 4*q0*q3
    ], dim=-1)  # [..., 3]

    denom = (4 * q0).clamp(min=1e-8)
    q123 = q123 / denom  #  4*q0  q1, q2, q3

    q = torch.cat([q0, q123], dim=-1)  # [..., 4]
    return _normalize(q)  #  |q| = 1
