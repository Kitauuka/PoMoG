from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from pomog.data.pocket_dataset import PoMoGPocketDataset
from pomog.models.network import PoMoG
from pomog.train.config import DEFAULT_CONFIG, _dataset_kwargs, _model_kwargs, load_config
from pomog.train.dataloader_utils import build_dataloader_kwargs
from pomog.train.epoch_loop import _eval_epoch, _train_epoch
from pomog.train.fold_utils import (
    build_training_summary,
    maybe_enable_direct_run_logging,
    parse_selected_folds,
    write_training_summary,
)
from pomog.train.runtime import _model_forward, _unwrap
from pomog.train.train_utils import (
    _build_lr_scheduler,
    _build_optimizer,
    _load_pickle_compat,
    _set_geometry_trainable,
)
from pomog.utils import seed_everything


def _log_branch_config(config: Dict[str, Any]) -> None:
    keys = [
        "model_name",
        "partition_method",
        "pocket_method",
        "component_edge_mode",
        "writeback_mode",
        "loss_mode",
    ]
    payload = {key: config.get(key) for key in keys}
    print(f"[BRANCH_CONFIG] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def train_pomog(args, config: Dict[str, Any] | None = None) -> None:
    cfg = config or load_config(DEFAULT_CONFIG, getattr(args, "config", None))
    _log_branch_config(cfg)
    seed_everything(cfg["seed"])
    selected_folds = parse_selected_folds(getattr(args, "fold_indices", None), int(cfg["folds"]))

    gpu_ids = None
    if getattr(args, "gpu_ids", None):
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    elif getattr(args, "gpu", None) is not None:
        gpu_ids = [int(args.gpu)]

    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu")
    print(f"Device: {device}")
    if gpu_ids:
        print(f"GPU IDs: {gpu_ids}")
    print(f"Selected folds: {selected_folds}")

    save_dir = cfg["model_save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    train_start_time = time.time()
    fold_summaries = []

    print(f"[STAGE] loading_data path={args.data_path}")
    data_load_start = time.time()
    full_data = _load_pickle_compat(args.data_path)
    print(f"[STAGE] data_loaded samples={len(full_data)} elapsed={time.time() - data_load_start:.2f}s")

    dataset_kwargs = _dataset_kwargs(cfg)
    epoch_test_loader = None
    if bool(cfg.get("log_epoch_test_metrics", False)):
        epoch_test_data_path = str(getattr(args, "epoch_test_data_path", "") or "")
        epoch_test_feature_dir = str(getattr(args, "epoch_test_feature_dir", "") or "")
        if not epoch_test_data_path or not epoch_test_feature_dir:
            raise ValueError("log_epoch_test_metrics=True requires --epoch_test_data_path and --epoch_test_feature_dir")
        epoch_test_data = _load_pickle_compat(epoch_test_data_path)
        epoch_test_ids = list(epoch_test_data.keys())
        epoch_test_ds = PoMoGPocketDataset(
            epoch_test_data,
            epoch_test_ids,
            root_dir=epoch_test_feature_dir,
            mode="infer",
            **dataset_kwargs,
        )
        epoch_test_loader = DataLoader(
            epoch_test_ds,
            **build_dataloader_kwargs(cfg, shuffle=False),
        )

    all_keys = list(full_data.keys())
    kf = KFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_keys)):
        if fold not in selected_folds:
            continue
        print(f"\n{'=' * 15} Fold {fold} {'=' * 15}")
        train_ids = [all_keys[i] for i in train_idx]
        val_ids = [all_keys[i] for i in val_idx]

        fold_prep_start = time.time()
        print(f"[STAGE] fold={fold} building_dataset train={len(train_ids)} cv={len(val_ids)}")
        train_ds = PoMoGPocketDataset(
            full_data,
            train_ids,
            root_dir=args.feature_dir,
            mode="train",
            **dataset_kwargs,
        )
        val_ds = PoMoGPocketDataset(
            full_data,
            val_ids,
            root_dir=args.feature_dir,
            mode="train",
            **dataset_kwargs,
        )
        print(
            f"[STAGE] fold={fold} dataset_ready train_len={len(train_ds)} cv_len={len(val_ds)} "
            f"elapsed={time.time() - fold_prep_start:.2f}s"
        )

        train_loader = DataLoader(
            train_ds,
            **build_dataloader_kwargs(cfg, shuffle=True),
        )
        val_loader = DataLoader(
            val_ds,
            **build_dataloader_kwargs(cfg, shuffle=False),
        )

        model = PoMoG(**_model_kwargs(cfg)).to(device)
        if torch.cuda.is_available() and gpu_ids and len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

        freeze_geometry_epochs = int(cfg.get("two_stage_freeze_geometry_epochs", 0))
        _set_geometry_trainable(model, trainable=freeze_geometry_epochs <= 0)
        optimizer = _build_optimizer(model, cfg)
        scheduler = _build_lr_scheduler(optimizer, cfg)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]]).to(device))

        best_aupr = 0.0
        best_auc = 0.0
        best_cv_loss = float("inf")
        best_epoch = -1
        best_ckpt_path = os.path.join(save_dir, f"fold{fold}.ckpt")
        patience_counter = 0

        for epoch in range(cfg["epochs"]):
            if freeze_geometry_epochs > 0:
                _set_geometry_trainable(model, trainable=epoch >= freeze_geometry_epochs)
            epoch_start = time.time()
            train_metrics = _train_epoch(model, train_loader, optimizer, criterion, device, cfg, epoch_idx=epoch)
            val_metrics = _eval_epoch(model, val_loader, criterion, device, cfg, epoch_idx=epoch)
            test_metrics = None
            if epoch_test_loader is not None:
                test_metrics = _eval_epoch(model, epoch_test_loader, criterion, device, cfg, epoch_idx=epoch)

            print(
                f"  Epoch {epoch:02d} | "
                f"Train Loss {train_metrics['loss']:.4f} "
                f"(bce={train_metrics['bce']:.4f}, dice={train_metrics['dice']:.4f}, "
                f"focal={train_metrics['focal']:.4f}, contrast={train_metrics['contrastive']:.4f}) | "
                f"CV AUC {val_metrics['auc']:.4f} | CV AUPR {val_metrics['aupr']:.4f} | "
                f"CV F1 {val_metrics['f1']:.4f} | CV MCC {val_metrics['mcc']:.4f} | "
                f"CV Precision {val_metrics['precision']:.4f} | CV FP {val_metrics['false_positive']} | "
                f"CV Loss {val_metrics['loss']:.4f}"
            )
            if test_metrics is not None:
                print(
                    f"  [EPOCH_TEST] fold={fold} epoch={epoch:02d} | "
                    f"AUC {test_metrics['auc']:.4f} | AUPR {test_metrics['aupr']:.4f} | "
                    f"F1 {test_metrics['f1']:.4f} | MCC {test_metrics['mcc']:.4f} | "
                    f"Precision {test_metrics['precision']:.4f} | FP {test_metrics['false_positive']}"
                )
            print(
                f"  [TIMING] epoch={epoch:02d} total={time.time() - epoch_start:.2f}s "
                f"train_fetch={train_metrics['fetch_seconds']:.2f}s train_compute={train_metrics['compute_seconds']:.2f}s "
                f"cv_fetch={val_metrics['fetch_seconds']:.2f}s cv_compute={val_metrics['compute_seconds']:.2f}s"
            )

            if val_metrics["aupr"] > best_aupr:
                best_aupr = val_metrics["aupr"]
                best_auc = val_metrics["auc"]
                best_cv_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), best_ckpt_path)
                print(f"  Saved -> {best_ckpt_path}")
            else:
                patience_counter += 1
                if bool(cfg.get("early_stop", True)) and patience_counter >= cfg["patience"]:
                    print(f"  Early stop at epoch {epoch}.")
                    break

            if scheduler is not None:
                scheduler.step()

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
        config=cfg,
        total_seconds=total_seconds,
        fold_results=fold_summaries,
        selected_folds=selected_folds,
        total_config_folds=int(cfg["folds"]),
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
    parser = argparse.ArgumentParser(description="Train PoMoG")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_ids", type=str, default=None, help="GPU list, e.g. 0,1")
    parser.add_argument("--fold_indices", type=str, default=None, help="Selected folds, e.g. 0,2,4")
    parser.add_argument("--summary_path", type=str, default=None, help="Training summary output path")
    parser.add_argument("--config", type=str, default=None, help="JSON config override file")
    parser.add_argument("--epoch_test_data_path", type=str, default="", help="Optional epoch-level test data")
    parser.add_argument("--epoch_test_feature_dir", type=str, default="", help="Optional epoch-level test features")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    config = load_config(DEFAULT_CONFIG, args.config)
    maybe_enable_direct_run_logging(args, config)
    train_pomog(args, config=config)
