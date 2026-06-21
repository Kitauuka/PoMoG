from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from pomog.data.residue_dataset import PoMoGDataset
from pomog.metrics import compute_mcc_best_thresh
from pomog.models.residue_encoder import PoMoG
from pomog.train.fold_utils import build_training_summary, parse_selected_folds, maybe_enable_direct_run_logging, write_training_summary
from pomog.utils import seed_everything


def _enable_line_buffered_output() -> None:
    """Make redirected training logs visible in near real time."""
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass


_enable_line_buffered_output()

DEFAULT_CONFIG: Dict[str, Any] = {
    "node_input_dim": 1024 + 9,
    "edge_input_dim": 450,
    "hidden_dim": 128,
    "hidden_vector_dim": 16,
    "num_layers": 4,
    "augment_eps": 0.1,
    "dropout": 0.2,
    "geometry_use_attention_adapter": False,
    "geometry_use_edge_adapter": True,
    "geometry_use_node_adapter": False,
    "geometry_use_gvp_lite": False,
    "geometry_use_edgemlp_geom": False,
    "geometry_use_confidence_dropout": False,
    "geometry_use_ffn_film": False,
    "geometry_use_sidechain_scalar": False,
    "geometry_use_edge_group_gate": False,
    "geometry_use_edge_memory": False,
    "geometry_use_coordinate_adapter": False,
    "geometry_use_distance_drift": False,
    "geometry_use_geo_context": False,
    "geometry_use_local_head": False,
    "geometry_use_motif_pool_head": False,
    "geometry_motif_pool_steps": 1,
    "geometry_use_multiradius": False,
    "geometry_multiradius_values": [6.0, 8.0, 12.0],
    "geometry_use_sequence_edges": False,
    "geometry_sequence_edge_span": 1,
    "geometry_attention_gate_init": -3.0,
    "geometry_edge_gate_init": -3.0,
    "geometry_node_gate_init": -3.0,
    "geometry_gvp_gate_init": -3.0,
    "geometry_geom_gate_init": -3.0,
    "geometry_confidence_dropout_base": 0.03,
    "geometry_confidence_dropout_alpha": 0.12,
    "geometry_confidence_dropout_max": 0.20,
    "geometry_branch_dropout": 0.0,
    "geometry_num_message_gvps": 1,
    "geometry_num_ff_gvps": 1,
    "geometry_gvp_lite_start_layer": 0,
    "geometry_vector_gate": True,
    "geometry_edge_vector_mode": "ca",
    "adapter_lr_mult": 1.0,
    "two_stage_freeze_geometry_epochs": 0,
    "confidence_loss_weighting": False,
    "confidence_loss_floor": 0.35,
    "batch_size": 4,
    "lr": 1e-4,
    "lr_scheduler": "none",
    "lr_min": 1e-6,
    "epochs": 50,
    "patience": 5,
    "early_stop": True,
    "folds": 5,
    "seed": 2023,
    "num_workers": 4,
    "pos_weight": 20.0,
    "dice_weight": 0.0,
    "focal_weight": 0.0,
    "focal_gamma": 2.0,
    "focal_alpha": -1.0,
    "model_save_dir": "./PoMoG/model/",
}


def _model_kwargs(config: Dict[str, Any], *, augment_eps: float | None = None) -> Dict[str, Any]:
    return {
        "node_input_dim": int(config["node_input_dim"]),
        "edge_input_dim": int(config["edge_input_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "hidden_vector_dim": int(config.get("hidden_vector_dim", 16)),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "augment_eps": float(config["augment_eps"] if augment_eps is None else augment_eps),
        "geometry_use_attention_adapter": bool(config.get("geometry_use_attention_adapter", False)),
        "geometry_use_edge_adapter": bool(config.get("geometry_use_edge_adapter", True)),
        "geometry_use_node_adapter": bool(config.get("geometry_use_node_adapter", False)),
        "geometry_use_gvp_lite": bool(config.get("geometry_use_gvp_lite", False)),
        "geometry_use_edgemlp_geom": bool(config.get("geometry_use_edgemlp_geom", False)),
        "geometry_use_confidence_dropout": bool(config.get("geometry_use_confidence_dropout", False)),
        "geometry_use_ffn_film": bool(config.get("geometry_use_ffn_film", False)),
        "geometry_use_sidechain_scalar": bool(config.get("geometry_use_sidechain_scalar", False)),
        "geometry_use_edge_group_gate": bool(config.get("geometry_use_edge_group_gate", False)),
        "geometry_use_edge_memory": bool(config.get("geometry_use_edge_memory", False)),
        "geometry_use_coordinate_adapter": bool(config.get("geometry_use_coordinate_adapter", False)),
        "geometry_use_distance_drift": bool(config.get("geometry_use_distance_drift", False)),
        "geometry_use_geo_context": bool(config.get("geometry_use_geo_context", False)),
        "geometry_use_local_head": bool(config.get("geometry_use_local_head", False)),
        "geometry_use_motif_pool_head": bool(config.get("geometry_use_motif_pool_head", False)),
        "geometry_motif_pool_steps": int(config.get("geometry_motif_pool_steps", 1)),
        "geometry_use_multiradius": bool(config.get("geometry_use_multiradius", False)),
        "geometry_multiradius_values": config.get("geometry_multiradius_values", [6.0, 8.0, 12.0]),
        "geometry_use_sequence_edges": bool(config.get("geometry_use_sequence_edges", False)),
        "geometry_sequence_edge_span": int(config.get("geometry_sequence_edge_span", 1)),
        "geometry_attention_gate_init": float(config.get("geometry_attention_gate_init", -3.0)),
        "geometry_edge_gate_init": float(config.get("geometry_edge_gate_init", -3.0)),
        "geometry_node_gate_init": float(config.get("geometry_node_gate_init", -3.0)),
        "geometry_gvp_gate_init": float(config.get("geometry_gvp_gate_init", -3.0)),
        "geometry_geom_gate_init": float(config.get("geometry_geom_gate_init", -3.0)),
        "geometry_confidence_dropout_base": float(config.get("geometry_confidence_dropout_base", 0.03)),
        "geometry_confidence_dropout_alpha": float(config.get("geometry_confidence_dropout_alpha", 0.12)),
        "geometry_confidence_dropout_max": float(config.get("geometry_confidence_dropout_max", 0.20)),
        "geometry_branch_dropout": float(config.get("geometry_branch_dropout", 0.0)),
        "geometry_num_message_gvps": int(config.get("geometry_num_message_gvps", 1)),
        "geometry_num_ff_gvps": int(config.get("geometry_num_ff_gvps", 1)),
        "geometry_gvp_lite_start_layer": int(config.get("geometry_gvp_lite_start_layer", 0)),
        "geometry_vector_gate": bool(config.get("geometry_vector_gate", True)),
        "geometry_edge_vector_mode": str(config.get("geometry_edge_vector_mode", "ca")),
    }


def _load_pickle_compat(path: str):
    """Load pickle files saved by either NumPy 1.x or 2.x runtimes."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            import numpy.core as _np_core

            sys.modules.setdefault("numpy._core", _np_core)
            f.seek(0)
            return pickle.load(f)


def load_config(base: Dict[str, Any], config_path: str | None = None) -> Dict[str, Any]:
    cfg = dict(base)
    if not config_path:
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        override = json.load(f)
    if not isinstance(override, dict):
        raise ValueError("config file must be a JSON object")
    cfg.update(override)
    return cfg


def _build_lr_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    scheduler_name = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_name in {"", "none", "off", "null"}:
        return None
    if scheduler_name == "cosine":
        t_max = int(config.get("cosine_t_max", config.get("epochs", 1)))
        eta_min = float(config.get("lr_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=eta_min,
        )
    raise ValueError(f"unsupported lr_scheduler: {scheduler_name}")


def _is_geometry_parameter(name: str) -> bool:
    tokens = (
        "adapter",
        "gvp_lite",
        "ffn_film",
        "edge_group_gate",
        "edge_memory",
        "edge_memory_gate",
        "geo_context",
        "coordinate",
    )
    return any(token in name for token in tokens)


def _set_geometry_trainable(model: nn.Module, trainable: bool) -> None:
    target = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in target.named_parameters():
        if _is_geometry_parameter(name):
            param.requires_grad = bool(trainable)


def _build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    lr = float(config["lr"])
    adapter_lr_mult = float(config.get("adapter_lr_mult", 1.0))
    if abs(adapter_lr_mult - 1.0) < 1e-8:
        return optim.Adam(model.parameters(), lr=lr)

    base_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if _is_geometry_parameter(name):
            adapter_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": lr})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": lr * adapter_lr_mult})
    return optim.Adam(groups, lr=lr)


def _confidence_sample_weight(batch, config: Dict[str, Any]) -> torch.Tensor | None:
    if not bool(config.get("confidence_loss_weighting", False)):
        return None
    X = batch.X.float()
    edge_index = batch.edge_index.long()
    ca = X[:, 1]
    src, dst = edge_index
    dist = torch.linalg.norm(ca[dst] - ca[src], dim=-1, keepdim=True)
    edge_conf = torch.exp(-torch.clamp(dist, min=0.0, max=30.0) / 10.0)
    node_conf = scatter_mean(edge_conf, dst, dim=0, dim_size=X.size(0))
    floor = float(config.get("confidence_loss_floor", 0.35))
    return (floor + (1.0 - floor) * node_conf).clamp(min=0.05, max=1.0)


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = -1.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(float(gamma))
    loss = modulating * bce

    if float(alpha) >= 0.0:
        alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
        loss = alpha_t * loss

    return loss.mean()


def _binary_loss_components(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    config: Dict[str, Any],
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if sample_weight is not None:
        pos_weight = getattr(criterion, "pos_weight", None)
        raw_bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        sample_weight = sample_weight.to(device=logits.device, dtype=logits.dtype)
        bce_loss = (raw_bce * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    else:
        bce_loss = criterion(logits, targets)
    dice_weight = float(config.get("dice_weight", 0.0))
    focal_weight = float(config.get("focal_weight", 0.0))

    dice_loss = _dice_loss_from_logits(logits, targets) if dice_weight > 0 else logits.new_tensor(0.0)
    focal_loss = (
        _sigmoid_focal_loss(
            logits,
            targets,
            gamma=float(config.get("focal_gamma", 2.0)),
            alpha=float(config.get("focal_alpha", -1.0)),
        )
        if focal_weight > 0
        else logits.new_tensor(0.0)
    )

    total = bce_loss + dice_weight * dice_loss + focal_weight * focal_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "bce": float(bce_loss.detach().cpu()),
        "dice": float(dice_loss.detach().cpu()),
        "focal": float(focal_loss.detach().cpu()),
        "dice_w": dice_weight,
        "focal_w": focal_weight,
    }


def _compute_loss_components(*args, **kwargs) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute the core binary loss with a PoMoG-compatible call signature."""
    sample_weight = kwargs.get("sample_weight")
    if len(args) >= 6:
        logits, targets, _projection, _seq, criterion, config = args[:6]
    else:
        logits, targets, criterion, config = args[:4]
    total, stats = _binary_loss_components(logits, targets, criterion, config, sample_weight)
    stats.update(
        {
            "contrastive": 0.0,
            "contrastive_raw": 0.0,
            "contrastive_w": 0.0,
            "contrastive_sampled": 0.0,
            "contrastive_valid_anchors": 0.0,
            "contrastive_types": 0.0,
            "contrastive_positive": 0.0,
            "contrastive_negative": 0.0,
            "contrastive_pos_anchors": 0.0,
            "contrastive_neg_anchors": 0.0,
        }
    )
    return total, stats


def _train_epoch(model, loader, optimizer, criterion, device, config: Dict[str, Any]):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_focal = 0.0
    fetch_seconds = 0.0
    compute_seconds = 0.0
    steps = 0

    loader_iter = iter(loader)
    progress = tqdm(total=len(loader), desc="  train", leave=False, disable=True)
    while True:
        fetch_start = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        fetch_seconds += time.time() - fetch_start

        compute_start = time.time()
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch)
        target = batch.activate_site.float().unsqueeze(1)
        sample_weight = _confidence_sample_weight(batch, config)
        loss, loss_comp = _binary_loss_components(pred, target, criterion, config, sample_weight)
        loss.backward()
        optimizer.step()
        compute_seconds += time.time() - compute_start

        total_loss += loss.item()
        total_bce += loss_comp["bce"]
        total_dice += loss_comp["dice"]
        total_focal += loss_comp["focal"]
        steps += 1
        progress.update(1)

    progress.close()
    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "dice": total_dice / denom,
        "focal": total_focal / denom,
        "fetch_seconds": fetch_seconds,
        "compute_seconds": compute_seconds,
    }


def _eval_epoch(model, loader, criterion, device, config: Dict[str, Any]):
    model.eval()
    preds, targets = [], []
    loss_total, loss_bce, loss_dice, loss_focal = [], [], [], []
    fetch_seconds = 0.0
    compute_seconds = 0.0

    with torch.no_grad():
        loader_iter = iter(loader)
        while True:
            fetch_start = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            fetch_seconds += time.time() - fetch_start

            compute_start = time.time()
            batch = batch.to(device)
            pred = model(batch.X, batch.node_feat, batch.edge_index, batch.seq, batch.batch)
            target = batch.activate_site.float().unsqueeze(1)
            sample_weight = _confidence_sample_weight(batch, config)
            loss, loss_comp = _compute_loss_components(pred, target, criterion, config, sample_weight)
            preds.append(torch.sigmoid(pred).cpu().numpy())
            targets.append(target.cpu().numpy())
            loss_total.append(loss_comp["loss"])
            loss_bce.append(loss_comp["bce"])
            loss_dice.append(loss_comp["dice"])
            loss_focal.append(loss_comp["focal"])
            compute_seconds += time.time() - compute_start

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    flat_preds = preds.reshape(-1)
    flat_targets = targets.reshape(-1)
    try:
        val_auc = roc_auc_score(flat_targets, flat_preds)
        precision, recall, _ = precision_recall_curve(flat_targets, flat_preds)
        val_aupr = auc(recall, precision)
    except ValueError:
        val_auc = val_aupr = 0.0
    val_mcc, decision_cutoff = compute_mcc_best_thresh(flat_targets, flat_preds)
    val_binary_preds = (flat_preds >= decision_cutoff).astype(int)
    val_f1 = float(f1_score(flat_targets, val_binary_preds, zero_division=0))
    return {
        "auc": float(val_auc),
        "aupr": float(val_aupr),
        "f1": val_f1,
        "mcc": float(val_mcc),
        "_decision_cutoff": float(decision_cutoff),
        "loss": float(np.mean(loss_total) if loss_total else 0.0),
        "bce": float(np.mean(loss_bce) if loss_bce else 0.0),
        "dice": float(np.mean(loss_dice) if loss_dice else 0.0),
        "focal": float(np.mean(loss_focal) if loss_focal else 0.0),
        "fetch_seconds": fetch_seconds,
        "compute_seconds": compute_seconds,
    }


def train_pomog(args, config: Dict[str, Any] | None = None) -> None:
    if config is None:
        config = load_config(DEFAULT_CONFIG, getattr(args, "config", None))

    seed_everything(config["seed"])
    selected_folds = parse_selected_folds(getattr(args, "fold_indices", None), int(config["folds"]))

    gpu_ids = None
    if getattr(args, "gpu_ids", None):
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    elif getattr(args, "gpu", None) is not None:
        gpu_ids = [int(args.gpu)]

    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    if gpu_ids:
        print(f"GPU IDs: {gpu_ids}")
    print(f"Selected folds: {selected_folds}")

    save_dir = config["model_save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    train_start_time = time.time()
    fold_summaries = []

    print(f"[STAGE] loading_data path={args.data_path}")
    data_load_start = time.time()
    full_data = _load_pickle_compat(args.data_path)
    print(f"[STAGE] data_loaded samples={len(full_data)} elapsed={time.time() - data_load_start:.2f}s")

    epoch_test_loader = None
    if bool(config.get("log_epoch_test_metrics", False)):
        epoch_test_data_path = str(getattr(args, "epoch_test_data_path", "") or "")
        epoch_test_feature_dir = str(getattr(args, "epoch_test_feature_dir", "") or "")
        if not epoch_test_data_path or not epoch_test_feature_dir:
            raise ValueError(
                "log_epoch_test_metrics=True requires --epoch_test_data_path and --epoch_test_feature_dir"
            )
        print(f"[STAGE] loading_epoch_test_data path={epoch_test_data_path}")
        epoch_test_load_start = time.time()
        epoch_test_data = _load_pickle_compat(epoch_test_data_path)
        epoch_test_ids = list(epoch_test_data.keys())
        epoch_test_ds = PoMoGDataset(
            epoch_test_data,
            epoch_test_ids,
            root_dir=epoch_test_feature_dir,
            mode="infer",
        )
        epoch_test_loader = DataLoader(
            epoch_test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        print(
            f"[STAGE] epoch_test_ready samples={len(epoch_test_ds)} "
            f"elapsed={time.time() - epoch_test_load_start:.2f}s"
        )

    all_keys = list(full_data.keys())
    kf = KFold(n_splits=config["folds"], shuffle=True, random_state=config["seed"])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_keys)):
        if fold not in selected_folds:
            continue
        print(f"\n{'=' * 15} Fold {fold} {'=' * 15}")
        train_ids = [all_keys[i] for i in train_idx]
        val_ids = [all_keys[i] for i in val_idx]

        fold_prep_start = time.time()
        print(f"[STAGE] fold={fold} building_dataset train={len(train_ids)} cv={len(val_ids)}")
        train_ds = PoMoGDataset(full_data, train_ids, root_dir=args.feature_dir, mode="train")
        val_ds = PoMoGDataset(full_data, val_ids, root_dir=args.feature_dir, mode="train")
        print(
            f"[STAGE] fold={fold} dataset_ready train_len={len(train_ds)} cv_len={len(val_ds)} "
            f"elapsed={time.time() - fold_prep_start:.2f}s"
        )

        loader_prep_start = time.time()
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )
        print(f"[STAGE] fold={fold} dataloader_ready elapsed={time.time() - loader_prep_start:.2f}s")

        model = PoMoG(**_model_kwargs(config)).to(device)
        if torch.cuda.is_available() and gpu_ids and len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

        freeze_geometry_epochs = int(config.get("two_stage_freeze_geometry_epochs", 0))
        _set_geometry_trainable(model, trainable=freeze_geometry_epochs <= 0)
        optimizer = _build_optimizer(model, config)
        scheduler = _build_lr_scheduler(optimizer, config)
        pos_weight = torch.tensor([config["pos_weight"]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_aupr = 0.0
        best_auc = 0.0
        best_cv_loss = float("inf")
        best_epoch = -1
        best_ckpt_path = os.path.join(save_dir, f"fold{fold}.ckpt")
        patience_counter = 0

        for epoch in range(config["epochs"]):
            if freeze_geometry_epochs > 0:
                _set_geometry_trainable(model, trainable=epoch >= freeze_geometry_epochs)
            epoch_start = time.time()
            train_metrics = _train_epoch(
                model, train_loader, optimizer, criterion, device, config
            )
            val_metrics = _eval_epoch(model, val_loader, criterion, device, config)
            test_metrics = None
            if epoch_test_loader is not None:
                test_metrics = _eval_epoch(model, epoch_test_loader, criterion, device, config)

            print(
                f"  Epoch {epoch:02d} | "
                f"Train Loss {train_metrics['loss']:.4f} "
                f"(bce={train_metrics['bce']:.4f}, dice={train_metrics['dice']:.4f}, focal={train_metrics['focal']:.4f}) | "
                f"CV AUC {val_metrics['auc']:.4f} | CV AUPR {val_metrics['aupr']:.4f} | "
                f"CV F1 {val_metrics['f1']:.4f} | CV MCC {val_metrics['mcc']:.4f} | "
                f"CV Loss {val_metrics['loss']:.4f}"
            )
            if test_metrics is not None:
                metric_prefix = f"fold={fold} epoch={epoch:02d}"
                print(
                    f"  [EPOCH_CV  ] {metric_prefix} | "
                    f"AUC {val_metrics['auc']:>7.4f} | AUPR {val_metrics['aupr']:>7.4f} | "
                    f"F1 {val_metrics['f1']:>7.4f} | MCC {val_metrics['mcc']:>7.4f} | "
                    f"Loss {val_metrics['loss']:>7.4f} | checkpoint_selection=cv_aupr"
                )
                print(
                    f"  [EPOCH_TEST] {metric_prefix} | "
                    f"AUC {test_metrics['auc']:>7.4f} | AUPR {test_metrics['aupr']:>7.4f} | "
                    f"F1 {test_metrics['f1']:>7.4f} | MCC {test_metrics['mcc']:>7.4f} | "
                    f"Loss {test_metrics['loss']:>7.4f} | checkpoint_selection=cv_only"
                )
            print(
                f"  [TIMING] epoch={epoch:02d} total={time.time() - epoch_start:.2f}s "
                f"train_fetch={train_metrics['fetch_seconds']:.2f}s train_compute={train_metrics['compute_seconds']:.2f}s "
                f"cv_fetch={val_metrics['fetch_seconds']:.2f}s cv_compute={val_metrics['compute_seconds']:.2f}s"
                + (
                    f" test_fetch={test_metrics['fetch_seconds']:.2f}s test_compute={test_metrics['compute_seconds']:.2f}s"
                    if test_metrics is not None
                    else ""
                )
            )
            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  [OPT] epoch={epoch:02d} lr={current_lr:.8f}")

            if val_metrics["aupr"] > best_aupr:
                best_aupr = val_metrics["aupr"]
                best_auc = val_metrics["auc"]
                best_cv_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), best_ckpt_path)
                print(f"  Saved → {best_ckpt_path}")
            else:
                patience_counter += 1
                if bool(config.get("early_stop", True)) and patience_counter >= config["patience"]:
                    print(f"  Early stop at epoch {epoch}.")
                    break

        fold_summary = {
            "fold": fold,
            "train_samples": len(train_ids),
            "cv_samples": len(val_ids),
            "best_epoch": best_epoch,
            "best_cv_auc": float(best_auc),
            "best_cv_aupr": float(best_aupr),
            "best_cv_loss": float(best_cv_loss),
            "best_ckpt": best_ckpt_path,
        }
        fold_summaries.append(fold_summary)
        print(
            f"[FOLD {fold}] best_epoch={best_epoch}, "
            f"best_cv_auc={best_auc:.4f}, best_cv_aupr={best_aupr:.4f}, ckpt={best_ckpt_path}"
        )

    total_seconds = time.time() - train_start_time
    summary = build_training_summary(
        task="pomog_train",
        data_path=args.data_path,
        feature_dir=args.feature_dir,
        device=str(device),
        gpu_ids=gpu_ids,
        config=config,
        total_seconds=total_seconds,
        fold_results=fold_summaries,
        selected_folds=selected_folds,
        total_config_folds=int(config["folds"]),
    )

    summary_path = getattr(args, "summary_path", None) or os.path.join(save_dir, "training_summary.json")
    write_training_summary(summary, summary_path)

    if fold_summaries:
        print(
            f"\n[SUMMARY] folds={len(fold_summaries)} "
            f"avg_cv_auc={summary['avg_best_cv_auc']:.4f} "
            f"avg_cv_aupr={summary['avg_best_cv_aupr']:.4f}"
        )
    print(f"[SUMMARY] total_seconds={total_seconds:.2f}")
    print(f"[SUMMARY] saved: {summary_path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train PoMoG")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--feature_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu_ids", type=str, default=None, help="GPU list for multi-GPU training, e.g. 0,1")
    p.add_argument("--fold_indices", type=str, default=None, help="selected folds, e.g. 0,2,4")
    p.add_argument("--summary_path", type=str, default=None, help="training summary output path")
    p.add_argument("--config", type=str, default=None, help="JSON config override file")
    p.add_argument("--epoch_test_data_path", type=str, default="", help="optional reference evaluation after each epoch")
    p.add_argument("--epoch_test_feature_dir", type=str, default="", help="optional reference evaluation after each epoch")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(DEFAULT_CONFIG, args.config)
    maybe_enable_direct_run_logging(args, config)
    train_pomog(args, config=config)
