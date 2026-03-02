"""
Data loading, Materials Project queries, dataset splitting, and PyG dataset classes.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, InMemoryDataset
from pymatgen.core import Structure

from src.utils import structure_to_graph, get_chemistry_family, set_seed


# ---------------------------------------------------------------------------
# Materials Project queries
# ---------------------------------------------------------------------------

def query_li_battery_data(api_key: str, save_path: Optional[str] = None,
                           max_entries: int = 10000) -> list[dict]:
    """
    Query the Materials Project battery database for Li insertion electrodes.

    Returns a list of dicts with keys:
        battery_id, formula, average_voltage, capacity_grav, capacity_vol,
        chemistry_family, charged_structure, discharged_structure, num_steps
    """
    from mp_api.client import MPRester

    print("Connecting to Materials Project API...")
    results = []

    with MPRester(api_key=api_key) as mpr:
        electrode_docs = mpr.electrodes.search(
            working_ion="Li",
            fields=[
                "battery_id",
                "average_voltage",
                "capacity_grav",
                "capacity_vol",
                "working_ion",
                "num_steps",
                "max_delta_volume",
                "adj_pairs",
                "framework_formula",
            ],
        )

    print(f"Retrieved {len(electrode_docs)} electrode entries.")

    for doc in electrode_docs[:max_entries]:
        try:
            avg_v = getattr(doc, "average_voltage", None)
            if avg_v is None or avg_v <= 0 or avg_v > 6.0:
                continue

            formula = getattr(doc, "framework_formula", "Unknown")
            chem_family = get_chemistry_family(formula)

            adj_pairs = getattr(doc, "adj_pairs", [])
            charged_struct = None
            discharged_struct = None

            if adj_pairs:
                pair = adj_pairs[0]
                charged_struct = getattr(pair, "charge_structure", None)
                discharged_struct = getattr(pair, "insertion_structure", None)

            if charged_struct is None and discharged_struct is None:
                continue

            entry = {
                "battery_id": str(getattr(doc, "battery_id", "")),
                "formula": formula,
                "average_voltage": float(avg_v),
                "capacity_grav": float(getattr(doc, "capacity_grav", 0) or 0),
                "capacity_vol": float(getattr(doc, "capacity_vol", 0) or 0),
                "num_steps": int(getattr(doc, "num_steps", 1) or 1),
                "max_delta_volume": float(getattr(doc, "max_delta_volume", 0) or 0),
                "chemistry_family": chem_family,
                "charged_structure": charged_struct.as_dict() if charged_struct else None,
                "discharged_structure": discharged_struct.as_dict() if discharged_struct else None,
            }
            results.append(entry)

        except Exception as e:
            continue

    print(f"Processed {len(results)} valid entries with voltages and structures.")

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved dataset to {save_path}")

    return results


def load_dataset(json_path: str) -> list[dict]:
    """Load a previously saved dataset from JSON."""
    with open(json_path) as f:
        return json.load(f)


def split_dataset(entries: list[dict], train_frac: float = 0.80,
                  val_frac: float = 0.10, seed: int = 42) -> tuple[list, list, list]:
    """
    Stratified split by chemistry_family (80/10/10 by default).
    Returns (train_entries, val_entries, test_entries).
    """
    set_seed(seed)

    families = sorted(set(e["chemistry_family"] for e in entries))
    train, val, test = [], [], []

    for family in families:
        subset = [e for e in entries if e["chemistry_family"] == family]
        np.random.shuffle(subset)
        n = len(subset)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(subset[:n_train])
        val.extend(subset[n_train:n_train + n_val])
        test.extend(subset[n_train + n_val:])

    print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


# ---------------------------------------------------------------------------
# PyTorch Geometric Dataset
# ---------------------------------------------------------------------------

class VoltageGraphDataset(InMemoryDataset):
    """
    PyG InMemoryDataset that converts pymatgen structures to graphs and stores
    the average voltage as the regression target.

    Args:
        entries:     list of dicts from load_dataset / split_dataset
        use_charged: if True use the charged structure; else use discharged
        cutoff:      neighbor cutoff in angstroms for graph construction
        n_gbf_bins:  number of Gaussian basis function bins for edge features
    """

    def __init__(self, entries: list[dict], use_charged: bool = True,
                 cutoff: float = 5.0, n_gbf_bins: int = 64):
        super().__init__(root=None)
        self.entries = entries
        self.use_charged = use_charged
        self.cutoff = cutoff
        self.n_gbf_bins = n_gbf_bins
        self._data_list = self._process_entries()
        self.data, self.slices = self.collate(self._data_list)

    def _process_entries(self) -> list[Data]:
        data_list = []
        struct_key = "charged_structure" if self.use_charged else "discharged_structure"

        for entry in self.entries:
            struct_dict = entry.get(struct_key)
            if struct_dict is None:
                struct_dict = entry.get("discharged_structure") or entry.get("charged_structure")
            if struct_dict is None:
                continue
            try:
                structure = Structure.from_dict(struct_dict)
                graph = structure_to_graph(structure, self.cutoff, self.n_gbf_bins)
                graph.y = torch.tensor([entry["average_voltage"]], dtype=torch.float32)
                graph.battery_id = entry["battery_id"]
                graph.formula = entry["formula"]
                graph.chemistry_family = entry["chemistry_family"]
                data_list.append(graph)
            except Exception:
                continue

        print(f"Built {len(data_list)} graphs.")
        return data_list

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def save_processed(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._data_list, f)

    @classmethod
    def from_processed(cls, path: str) -> "VoltageGraphDataset":
        with open(path, "rb") as f:
            data_list = pickle.load(f)
        obj = cls.__new__(cls)
        obj._data_list = data_list
        obj.data, obj.slices = InMemoryDataset.collate(data_list)
        return obj


# ---------------------------------------------------------------------------
# Matminer feature matrix (for Random Forest baseline)
# ---------------------------------------------------------------------------

def build_matminer_features(entries: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract compositional and structural descriptors using matminer.

    Returns:
        X:             (N, n_features) feature matrix
        y:             (N,) voltage targets
        feature_names: list of feature name strings
    """
    from matminer.featurizers.composition import ElementProperty, IonProperty
    from matminer.featurizers.base import MultipleFeaturizer
    from pymatgen.core import Composition

    ep = ElementProperty.from_preset("magpie")
    ip = IonProperty()
    ip.fast = True

    featurizer = MultipleFeaturizer([ep, ip])
    feature_names = featurizer.feature_labels()

    X_rows, y_vals = [], []

    for entry in entries:
        try:
            comp = Composition(entry["formula"])
            feats = featurizer.featurize(comp)
            X_rows.append(feats)
            y_vals.append(entry["average_voltage"])
        except Exception:
            continue

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)

    # Replace NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names
