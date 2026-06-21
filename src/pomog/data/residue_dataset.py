"""Residue graph dataset used by PoMoG.

Each sample is returned as a ``torch_geometric.data.Data`` object with:

  name          : protein identifier
  seq           : [L] residue indices
  X             : [L, 5, 3] backbone/side-chain anchor coordinates
  node_feat     : [L, 1033] ProtTrans(1024) + DSSP(9)
  edge_index    : [2, E] radius graph edges
  activate_site : [L] binary active-site labels
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

import numpy as np
import torch
import torch.utils.data as data
import torch_geometric
import torch_geometric.nn

from pomog.data.graph import LETTER_TO_NUM


class PoMoGDataset(data.Dataset):
    """Load per-protein feature tensors and build residue graphs.

    Args:
        dataset   : {protein_id: [sequence_str, label_array]}
                    label_array has length L and contains 0/1 labels.
        index     : protein IDs selected for this dataset instance.
        root_dir  : feature root containing Structures/, ProtTrans/, and DSSP/.
        radius    : C-alpha radius graph cutoff in Angstrom.
        mode      : "train" or "infer".
    """

    def __init__(
        self,
        dataset: Dict[str, list],
        index: Sequence,
        root_dir: str,
        radius: float = 10,
        mode: str = "train",
    ) -> None:
        super().__init__()
        index_set = set(index)
        self.dataset = {k: v for k, v in dataset.items() if k in index_set}
        self.IDs = list(self.dataset.keys())
        self.root_dir = root_dir
        self.radius = radius
        self.mode = mode
        self.radius_graph_workers = max(1, int(os.environ.get("POMOG_RADIUS_GRAPH_WORKERS", "4")))

        self.max_prot: torch.Tensor | None = None
        self.min_prot: torch.Tensor | None = None
        max_path = os.path.join(root_dir, "ProtTrans", "Max_ProtTrans.npy")
        min_path = os.path.join(root_dir, "ProtTrans", "Min_ProtTrans.npy")
        if os.path.exists(max_path) and os.path.exists(min_path):
            try:
                self.max_prot = torch.from_numpy(np.load(max_path)).float()
                self.min_prot = torch.from_numpy(np.load(min_path)).float()
            except Exception as e:
                print(f"[PoMoGDataset] failed to load ProtTrans normalization stats: {e}")
        else:
            print("[PoMoGDataset] ProtTrans normalization stats were not found.")

    def __len__(self) -> int:
        return len(self.IDs)

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        return self._featurize_graph(idx)

    def _find_file(self, subdir: str, name: str) -> str | None:
        """Find a tensor file using original, upper-case, or lower-case names."""
        for candidate in [name, name.upper(), name.lower()]:
            p = os.path.join(self.root_dir, subdir, candidate + ".tensor")
            if os.path.exists(p):
                return p
        return None

    def _assert_feature_integrity(
        self,
        name: str,
        seq_str: str,
        X: torch.Tensor,
        prottrans_feat: torch.Tensor,
        dssp_feat: torch.Tensor,
        activate_site: torch.Tensor,
    ) -> None:
        """Validate feature shapes, residue counts, labels, and finite values."""
        if X.ndim != 3 or X.shape[1] != 5 or X.shape[2] != 3:
            raise ValueError(
                f"[PoMoGDataset] invalid structure shape for {name}: {tuple(X.shape)}; expected [L, 5, 3]"
            )

        if prottrans_feat.ndim != 2 or prottrans_feat.shape[1] != 1024:
            raise ValueError(
                f"[PoMoGDataset] invalid ProtTrans shape for {name}: {tuple(prottrans_feat.shape)}; expected [L, 1024]"
            )

        if dssp_feat.ndim != 2 or dssp_feat.shape[1] != 9:
            raise ValueError(
                f"[PoMoGDataset] invalid DSSP shape for {name}: {tuple(dssp_feat.shape)}; expected [L, 9]"
            )

        seq_len = len(seq_str)
        x_len = int(X.shape[0])
        prot_len = int(prottrans_feat.shape[0])
        dssp_len = int(dssp_feat.shape[0])

        if not (seq_len == x_len == prot_len == dssp_len):
            raise ValueError(
                f"[PoMoGDataset] length mismatch for {name}: seq={seq_len}, X={x_len}, ProtTrans={prot_len}, DSSP={dssp_len}"
            )

        if self.mode == "train":
            label_len = int(activate_site.shape[0])
            if label_len != seq_len:
                raise ValueError(
                    f"[PoMoGDataset] label length mismatch for {name}: label={label_len}, seq={seq_len}"
                )

        for tensor_name, tensor_value in {
            "X": X,
            "ProtTrans": prottrans_feat,
            "DSSP": dssp_feat,
            "activate_site": activate_site,
        }.items():
            if not torch.isfinite(tensor_value).all():
                raise ValueError(
                    f"[PoMoGDataset] non-finite values in {tensor_name} for {name}"
                )

    def _featurize_graph(self, idx: int) -> torch_geometric.data.Data:
        """Build a PyG graph from structure, sequence, ProtTrans, and DSSP features."""
        name = self.IDs[idx]

        with torch.no_grad():
            struct_path = self._find_file("Structures", name)
            if struct_path is None:
                raise FileNotFoundError(f"[PoMoGDataset] missing structure tensor: {name}")
            X = torch.load(struct_path)

            seq_str = self.dataset[name][0]
            seq = torch.tensor(
                [LETTER_TO_NUM.get(aa, 20) for aa in seq_str], dtype=torch.long
            )

            prot_path = self._find_file("ProtTrans", name)
            if prot_path is None:
                raise FileNotFoundError(f"[PoMoGDataset] missing ProtTrans tensor: {name}")
            prottrans_feat = torch.load(prot_path)

            if self.max_prot is not None:
                denom = self.max_prot - self.min_prot
                denom[denom == 0] = 1.0
                prottrans_feat = (prottrans_feat - self.min_prot) / denom

            dssp_path = self._find_file("DSSP", name)
            dssp_feat = torch.load(dssp_path) if dssp_path is not None \
                else torch.zeros((prottrans_feat.shape[0], 9))

            node_feat = torch.cat([prottrans_feat, dssp_feat], dim=-1)

            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(
                X_ca,
                r=self.radius,
                loop=True,
                max_num_neighbors=1000,
                num_workers=self.radius_graph_workers,
            )

            if len(self.dataset[name]) >= 2:
                activate_site = torch.from_numpy(
                    np.array(self.dataset[name][1])
                ).float()
            else:
                activate_site = torch.zeros(seq.shape[0]).float()

            self._assert_feature_integrity(
                name=name,
                seq_str=seq_str,
                X=X,
                prottrans_feat=prottrans_feat,
                dssp_feat=dssp_feat,
                activate_site=activate_site,
            )

        return torch_geometric.data.Data(
            name=name,
            seq=seq,
            X=X,
            node_feat=node_feat,
            edge_index=edge_index,
            activate_site=activate_site,
        )
