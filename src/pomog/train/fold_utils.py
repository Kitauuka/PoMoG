from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO



_LOG_CAPTURE_ACTIVE = False
_LOG_CAPTURE_HANDLES: list[TextIO] = []


class _TeeStream:
    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._mirror.write(data)
        if "\n" in data:
            self._mirror.flush()
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _infer_summary_path(args: Any, config: dict[str, Any]) -> Path:
    summary_path = getattr(args, "summary_path", None)
    if summary_path:
        return Path(summary_path)
    return Path(str(config.get("model_save_dir", "./outputs/train/PoMoG/model"))) / "training_summary.json"


def _infer_run_dir_from_summary_path(path: Path) -> Path | None:
    if path.parent.name != "model":
        return None
    return path.parent.parent


def _infer_default_log_path(summary_path: Path) -> Path | None:
    run_dir = _infer_run_dir_from_summary_path(summary_path)
    if run_dir is None:
        return None
    if summary_path.stem.startswith("training_summary.worker"):
        worker_suffix = summary_path.stem.replace("training_summary.worker", "", 1)
        if worker_suffix.isdigit():
            return run_dir / f"train.worker{worker_suffix}.log"
    return run_dir / "train.log"


def maybe_enable_direct_run_logging(args: Any, config: dict[str, Any]) -> Path | None:
    global _LOG_CAPTURE_ACTIVE
    if _LOG_CAPTURE_ACTIVE:
        return None
    if str(Path(sys.argv[0]).name) == "run_pomog_experiment.py":
        return None
    if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        return None

    summary_path = _infer_summary_path(args, config)
    log_path = _infer_default_log_path(summary_path)
    if log_path is None:
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    _LOG_CAPTURE_HANDLES.append(handle)
    sys.stdout = _TeeStream(sys.stdout, handle)
    sys.stderr = _TeeStream(sys.stderr, handle)
    _LOG_CAPTURE_ACTIVE = True
    print(f"[LOG] tee stdout/stderr -> {log_path}")
    return log_path


def _curve_log_paths_for_summary(summary_path: Path) -> list[Path]:
    if summary_path.stem.startswith("training_summary.worker"):
        return []
    run_dir = _infer_run_dir_from_summary_path(summary_path)
    if run_dir is None:
        return []
    worker_logs = sorted(run_dir.glob("train.worker*.log"))
    if worker_logs:
        return worker_logs
    train_log = run_dir / "train.log"
    if train_log.exists():
        return [train_log]
    return []


def maybe_export_training_curves(summary_path: str | Path) -> dict[str, str] | None:
    summary_path = Path(summary_path)
    log_paths = _curve_log_paths_for_summary(summary_path)
    if not log_paths:
        return None

    run_dir = _infer_run_dir_from_summary_path(summary_path)
    if run_dir is None:
        return None

    curves_dir = run_dir / "curves"
    curve_log = run_dir / "curve_export.log"
    plot_script = _repo_root() / "scripts" / "plot_pomog_train_curves.py"
    command = [
        sys.executable,
        str(plot_script),
        "--output_dir",
        str(curves_dir),
        "--log_paths",
        *[str(path) for path in log_paths],
    ]
    curve_log.parent.mkdir(parents=True, exist_ok=True)
    with curve_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[COMMAND] {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=str(_repo_root()),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    return {
        "curves_dir": str(curves_dir),
        "avg_csv_path": str(curves_dir / "train.avg.csv"),
        "avg_png_path": str(curves_dir / "train.avg.png"),
        "interactive_html_path": str(curves_dir / "train.interactive.html"),
        "curve_export_log_path": str(curve_log),
    }


def parse_selected_folds(raw: str | None, total_folds: int) -> list[int]:
    """Parse an optional comma-separated fold list into sorted unique indices."""
    if int(total_folds) <= 0:
        raise ValueError(f"folds must be positive, got: {total_folds}")

    if raw is None or not str(raw).strip():
        return list(range(int(total_folds)))

    selected: set[int] = set()
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            idx = int(chunk)
        except ValueError as exc:
            raise ValueError(f"invalid fold index: {chunk}") from exc
        if idx < 0 or idx >= int(total_folds):
            raise ValueError(f"fold index out of range: {idx} not in [0, {int(total_folds) - 1}]")
        selected.add(idx)

    if not selected:
        raise ValueError("at least one fold index must be selected")
    return sorted(selected)


def build_training_summary(
    *,
    task: str,
    data_path: str,
    feature_dir: str,
    device: str,
    gpu_ids: Sequence[int] | None,
    config: dict[str, Any],
    total_seconds: float,
    fold_results: Sequence[dict[str, Any]],
    selected_folds: Iterable[int],
    total_config_folds: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard training summary for full or partial fold runs."""
    payload: dict[str, Any] = {
        "task": task,
        "data_path": data_path,
        "feature_dir": feature_dir,
        "device": device,
        "gpu_ids": list(gpu_ids) if gpu_ids else None,
        "config": config,
        "total_seconds": float(total_seconds),
        "fold_results": list(fold_results),
        "selected_folds": list(selected_folds),
        "total_config_folds": int(total_config_folds),
    }
    if fold_results:
        payload["avg_best_cv_auc"] = float(
            sum(float(item["best_cv_auc"]) for item in fold_results) / float(len(fold_results))
        )
        payload["avg_best_cv_aupr"] = float(
            sum(float(item["best_cv_aupr"]) for item in fold_results) / float(len(fold_results))
        )
    if extra:
        payload.update(extra)
    return payload


def write_training_summary(summary: dict[str, Any], path_text: str) -> Path:
    """Write a training summary JSON to disk."""
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key in list(summary):
        if "threshold" in key.lower():
            summary.pop(key, None)
    for item in summary.get("fold_results", []):
        if isinstance(item, dict):
            for key in list(item):
                if "threshold" in key.lower():
                    item.pop(key, None)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        maybe_export_training_curves(path)
    except Exception as exc:
        print(f"[WARN] failed to export training curves for {path}: {exc}")
    return path
