from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _model_forward(model: nn.Module, batch, *, return_projection: bool = False):
    return model(
        batch.X,
        batch.node_feat,
        batch.edge_index,
        batch.seq,
        batch.batch,
        batch.residue_pocket_feature,
        batch.context_node_type,
        batch.context_node_raw_feature,
        batch.context_node_member_index,
        batch.context_graph_edge_index,
        batch.context_graph_edge_attr,
        batch.context_edge_index,
        getattr(batch, "context_node_label", None),
        getattr(batch, "structure_prior_feat", None),
        getattr(batch, "structure_prior_mask", None),
        getattr(batch, "structure_prior_detector_missing", None),
        return_projection=return_projection,
        structure_prior_pocket_feature=getattr(batch, "structure_prior_pocket_feature", None),
        structure_prior_residue_to_pocket_index=getattr(batch, "structure_prior_residue_to_pocket_index", None),
        structure_prior_residue_to_pocket_attr=getattr(batch, "structure_prior_residue_to_pocket_attr", None),
    )


def _amp_mode(config: Dict[str, Any], device: torch.device) -> str:
    if device.type != "cuda" or not bool(config.get("use_amp", False)):
        return "off"
    mode = str(config.get("amp_dtype", "fp16")).lower()
    return mode if mode in {"fp16", "bf16"} else "fp16"


def _autocast(mode: str):
    if mode == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if mode == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(enabled=False)
