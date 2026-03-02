"""
Utility functions for atom featurization, graph construction, and helpers.
"""

from __future__ import annotations

import numpy as np
import torch
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Atom-level property lookup tables
# ---------------------------------------------------------------------------

# Pauling electronegativity; 0.0 for noble gases / undefined
ELECTRONEGATIVITY: dict[str, float] = {
    "H": 2.20, "He": 0.00, "Li": 0.98, "Be": 1.57, "B": 2.04,
    "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.00,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19,
    "S": 2.58, "Cl": 3.16, "Ar": 0.00, "K": 0.82, "Ca": 1.00,
    "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Kr": 0.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 0.00, "Cs": 0.79,
    "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10,
    "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10,
    "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90,
    "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.00, "At": 2.20,
    "Rn": 0.00, "Fr": 0.70, "Ra": 0.90,
}

# Effective ionic radii (angstroms) for the most common oxidation state; -1 = unknown
IONIC_RADIUS: dict[str, float] = {
    "H": 1.40, "Li": 0.76, "Be": 0.45, "B": 0.27, "C": 0.16,
    "N": 1.46, "O": 1.40, "F": 1.33, "Na": 1.02, "Mg": 0.72,
    "Al": 0.535, "Si": 0.40, "P": 0.38, "S": 1.84, "Cl": 1.81,
    "K": 1.38, "Ca": 1.00, "Sc": 0.745, "Ti": 0.605, "V": 0.54,
    "Cr": 0.615, "Mn": 0.645, "Fe": 0.645, "Co": 0.545, "Ni": 0.69,
    "Cu": 0.73, "Zn": 0.74, "Ga": 0.62, "Ge": 0.53, "As": 0.335,
    "Se": 1.98, "Br": 1.96, "Rb": 1.52, "Sr": 1.18, "Y": 0.90,
    "Zr": 0.72, "Nb": 0.64, "Mo": 0.59, "Ru": 0.62, "Rh": 0.665,
    "Pd": 0.86, "Ag": 1.15, "Cd": 0.95, "In": 0.80, "Sn": 0.69,
    "Sb": 0.76, "Te": 2.21, "I": 2.20, "Cs": 1.67, "Ba": 1.35,
    "La": 1.032, "Ce": 0.87, "Nd": 0.983, "Sm": 0.958, "Eu": 1.17,
    "Gd": 0.938, "Tb": 0.923, "Dy": 0.912, "Ho": 0.901, "Er": 0.890,
    "Tm": 0.880, "Yb": 0.868, "Lu": 0.861, "Hf": 0.71, "Ta": 0.64,
    "W": 0.60, "Re": 0.58, "Os": 0.545, "Ir": 0.625, "Pt": 0.625,
    "Au": 1.37, "Hg": 1.02, "Tl": 1.50, "Pb": 1.19, "Bi": 1.03,
}


def get_atom_features(symbol: str) -> list[float]:
    """
    Return a fixed-length feature vector for an element symbol.

    Features (9 total):
        0: atomic number (normalized by 94)
        1: Pauling electronegativity (normalized by 4)
        2: ionic radius in angstroms (normalized by 2.5)
        3: group number (normalized by 18)
        4: period number (normalized by 7)
        5: is metal (binary)
        6: is transition metal (binary)
        7: is alkali metal (binary)
        8: is alkaline earth metal (binary)
    """
    try:
        el = Element(symbol)
    except Exception:
        return [0.0] * 9

    atomic_num = el.number / 94.0
    en = ELECTRONEGATIVITY.get(symbol, 0.0) / 4.0
    ir = IONIC_RADIUS.get(symbol, 1.0) / 2.5
    group = (el.group or 1) / 18.0
    period = el.row / 7.0
    is_metal = float(el.is_metal)
    is_tm = float(el.is_transition_metal)
    is_alkali = float(el.is_alkali)
    is_alkaline = float(el.is_alkaline)

    return [atomic_num, en, ir, group, period, is_metal, is_tm, is_alkali, is_alkaline]


def gaussian_basis(distance: float, n_bins: int = 64, d_min: float = 0.5,
                   d_max: float = 5.0) -> list[float]:
    """
    Encode a scalar distance as a Gaussian-smeared basis expansion.
    Centers are linearly spaced from d_min to d_max.
    Width sigma = (d_max - d_min) / n_bins.
    """
    centers = np.linspace(d_min, d_max, n_bins)
    sigma = (d_max - d_min) / n_bins
    return np.exp(-((distance - centers) ** 2) / (2 * sigma ** 2)).tolist()


def structure_to_graph(structure: Structure, cutoff: float = 5.0,
                       n_gbf_bins: int = 64) -> Data:
    """
    Convert a pymatgen Structure to a PyG Data object.

    Node features: 9-dimensional atom feature vector per site.
    Edge features: 64-dimensional Gaussian basis expansion of bond distance.
    Returns a Data object with:
        x         (N, 9)   node features
        edge_index (2, E)  bond connectivity
        edge_attr  (E, 64) edge features
        num_nodes  int
    """
    nn_finder = CrystalNN()

    node_features = []
    edge_src, edge_dst = [], []
    edge_features = []

    for i, site in enumerate(structure):
        node_features.append(get_atom_features(site.specie.symbol))

    for i in range(len(structure)):
        try:
            neighbors = nn_finder.get_nn_info(structure, i)
        except Exception:
            # Fallback to distance-based neighbors
            neighbors = [
                {"site_index": j, "weight": 1.0}
                for j, site in enumerate(structure)
                if i != j and structure.get_distance(i, j) < cutoff
            ]

        for nb in neighbors:
            j = nb["site_index"]
            dist = structure.get_distance(i, j)
            if dist < cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                edge_features.append(gaussian_basis(dist, n_gbf_bins))

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(structure))


def get_chemistry_family(formula: str) -> str:
    """
    Classify a formula string into a broad chemistry family
    based on the anion species present.
    """
    formula_lower = formula.lower()
    if "po4" in formula_lower or "p" in formula_lower and "o" in formula_lower:
        return "phosphate"
    if "so4" in formula_lower or "s" in formula_lower and "o" in formula_lower:
        return "sulfate"
    if "sio" in formula_lower:
        return "silicate"
    if "s" in formula_lower and "o" not in formula_lower:
        return "sulfide"
    if "f" in formula_lower and "o" not in formula_lower:
        return "fluoride"
    if "o" in formula_lower:
        return "oxide"
    return "other"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
