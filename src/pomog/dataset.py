"""Public PoMoG dataset entrypoint."""

from __future__ import annotations

from pomog.data.pocket_dataset import PoMoGPocketDataset as PoMoGDataset

PoMoGDataset.__name__ = "PoMoGDataset"
PoMoGDataset.__qualname__ = "PoMoGDataset"

__all__ = ["PoMoGDataset"]
