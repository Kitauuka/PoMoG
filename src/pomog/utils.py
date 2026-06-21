from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 2023) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def padding_ver1(x: torch.Tensor, batch_id: torch.Tensor, feature_dim: int, activate_site: torch.Tensor):
    counts = torch.unique(batch_id, return_counts=True)[1]
    batch_size = int(counts.shape[0])
    max_len = int(counts.max().item())
    batch_data = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len)
    batch_activate_site = torch.zeros(batch_size, max_len, 1)
    offset = 0
    for i in range(batch_size):
        length = int(counts[i].item())
        batch_data[i, :length] = x[offset: offset + length]
        batch_activate_site[i, :length] = activate_site[offset: offset + length]
        mask[i, :length] = 1
        offset += length
    return batch_data, mask, batch_activate_site
