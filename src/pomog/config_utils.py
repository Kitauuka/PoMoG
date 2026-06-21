from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_POMOG_ALIASES = {
    "geometry_use_vector_layers": "geometry_use_gvp_lite",
    "geometry_use_edge_mlp": "geometry_use_edgemlp_geom",
    "geometry_use_multiradius": "geometry_use_multiradius",
    "geometry_multiradius_values": "geometry_multiradius_values",
    "geometry_edge_vector_mode": "geometry_edge_vector_mode",
    "geometry_use_confidence_dropout": "geometry_use_confidence_dropout",
    "geometry_confidence_dropout_base": "geometry_confidence_dropout_base",
    "geometry_confidence_dropout_alpha": "geometry_confidence_dropout_alpha",
    "geometry_confidence_dropout_max": "geometry_confidence_dropout_max",
    "geometry_vector_gate_init": "geometry_gvp_gate_init",
    "local_segment_aux_weight": "micro_motif_aux_weight",
    "merge_short_segments": "merge_short_motifs",
    "min_segment_length": "min_motif_length",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_text: str | Path, *, root: Path | None = None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (root or repo_root()) / path


def load_json_config(path_text: str | Path) -> dict[str, Any]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return payload


def with_internal_aliases(config: Mapping[str, Any]) -> dict[str, Any]:
    """Map release-facing PoMoG keys to the checkpoint-compatible internals."""
    cfg = dict(config)
    for public_key, internal_key in _POMOG_ALIASES.items():
        if public_key in cfg:
            cfg[internal_key] = cfg[public_key]
    return cfg


def resolve_json_config(
    config_path: str | Path | None,
    *,
    base: Mapping[str, Any] | None = None,
    root: Path | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(base or {})
    if not config_path:
        return cfg

    root = root or repo_root()
    path = resolve_path(config_path, root=root).resolve()
    seen = set(_seen or set())
    if path in seen:
        chain = " -> ".join(str(item) for item in [*seen, path])
        raise ValueError(f"recursive base_config cycle detected: {chain}")
    seen.add(path)

    override = load_json_config(path)
    parent_path = override.get("base_config")
    if parent_path:
        cfg = resolve_json_config(parent_path, base=base, root=root, _seen=seen)
    cfg.update(override)
    return with_internal_aliases(cfg)


def load_training_run_config(
    model_dir: str | Path,
    default_config: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(model_dir).expanduser().resolve().parent
    resolved_path = run_dir / "config.train.resolved.json"
    if resolved_path.exists():
        cfg = dict(default_config)
        cfg.update(load_json_config(resolved_path))
        return with_internal_aliases(cfg)

    config_path = run_dir / "config.train.json"
    if config_path.exists():
        return resolve_json_config(config_path, base=default_config, root=root or repo_root())
    return with_internal_aliases(default_config)
