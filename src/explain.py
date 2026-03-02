"""
Explainability helpers for graph voltage models using Captum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


AttributionMethod = Literal["ig", "gradient_shap"]


@dataclass
class GraphAttributionResult:
    """Container for single-graph attribution outputs."""

    prediction: float
    method: AttributionMethod
    node_attr: torch.Tensor
    edge_attr: torch.Tensor
    node_scores: torch.Tensor
    edge_scores: torch.Tensor
    atom_scores: torch.Tensor


def _make_forward_fn(model: nn.Module, edge_index: torch.Tensor,
                     num_nodes: int):
    """
    Build a Captum-compatible forward function from tensor inputs.
    """

    def forward_fn(x_batch: torch.Tensor,
                   edge_attr_batch: torch.Tensor) -> torch.Tensor:
        data_list = []
        for i in range(x_batch.size(0)):
            data_list.append(
                Data(
                    x=x_batch[i],
                    edge_index=edge_index,
                    edge_attr=edge_attr_batch[i],
                    num_nodes=num_nodes,
                )
            )
        batch = Batch.from_data_list(data_list).to(x_batch.device)
        return model(batch).view(-1)

    return forward_fn


def _aggregate_atom_scores(node_attr: torch.Tensor,
                           edge_attr: torch.Tensor,
                           edge_index: torch.Tensor,
                           normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert node/edge feature attributions to per-atom influence scores.
    """
    node_scores = node_attr.abs().sum(dim=-1)
    edge_scores = edge_attr.abs().sum(dim=-1)

    atom_scores = node_scores.clone()
    src, dst = edge_index
    atom_scores.index_add_(0, src, 0.5 * edge_scores)
    atom_scores.index_add_(0, dst, 0.5 * edge_scores)

    if normalize:
        denom = atom_scores.sum().clamp_min(1e-12)
        atom_scores = atom_scores / denom

    return node_scores, edge_scores, atom_scores


def explain_single_graph_prediction(
    model: nn.Module,
    graph: Data,
    method: AttributionMethod = "ig",
    n_steps: int = 64,
    n_samples: int = 64,
    stdevs: float = 0.01,
    normalize: bool = True,
    internal_batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> GraphAttributionResult:
    """
    Compute feature attribution for one graph prediction.

    Args:
        model: trained PyG model returning graph-level regression output
        graph: single graph with x, edge_index, edge_attr
        method: "ig" (Integrated Gradients) or "gradient_shap" (SHAP-like)
        n_steps: IG integration steps
        n_samples: GradientShap Monte Carlo samples
        stdevs: GradientShap noise scale
        normalize: normalize per-atom scores to sum to 1
        internal_batch_size: optional Captum chunking size for IG
        device: optional device override
    """
    try:
        from captum.attr import GradientShap, IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "captum is required for explainability. Install with `pip install captum`."
        ) from exc

    if graph.x is None or graph.edge_attr is None:
        raise ValueError("Graph must include `x` and `edge_attr` tensors.")

    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    x = graph.x.to(device).float()
    edge_attr = graph.edge_attr.to(device).float()
    edge_index = graph.edge_index.to(device)
    num_nodes = int(getattr(graph, "num_nodes", x.size(0)))

    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(-1)

    x_in = x.unsqueeze(0).requires_grad_(True)
    edge_in = edge_attr.unsqueeze(0).requires_grad_(True)

    forward_fn = _make_forward_fn(model=model, edge_index=edge_index,
                                  num_nodes=num_nodes)

    if method == "ig":
        explainer = IntegratedGradients(forward_fn)
        attr_x, attr_edge = explainer.attribute(
            inputs=(x_in, edge_in),
            baselines=(torch.zeros_like(x_in), torch.zeros_like(edge_in)),
            n_steps=n_steps,
            method="gausslegendre",
            internal_batch_size=internal_batch_size,
        )
    elif method == "gradient_shap":
        explainer = GradientShap(forward_fn)
        x_mean = x.mean(dim=0, keepdim=True).expand_as(x)
        edge_mean = edge_attr.mean(dim=0, keepdim=True).expand_as(edge_attr)
        x_baselines = torch.stack([torch.zeros_like(x), x_mean], dim=0)
        edge_baselines = torch.stack(
            [torch.zeros_like(edge_attr), edge_mean], dim=0
        )
        attr_x, attr_edge = explainer.attribute(
            inputs=(x_in, edge_in),
            baselines=(x_baselines, edge_baselines),
            n_samples=n_samples,
            stdevs=stdevs,
        )
    else:
        raise ValueError("method must be either 'ig' or 'gradient_shap'.")

    attr_x = attr_x.squeeze(0)
    attr_edge = attr_edge.squeeze(0)

    node_scores, edge_scores, atom_scores = _aggregate_atom_scores(
        node_attr=attr_x,
        edge_attr=attr_edge,
        edge_index=edge_index,
        normalize=normalize,
    )

    with torch.no_grad():
        prediction = float(forward_fn(x_in.detach(), edge_in.detach())[0].item())

    return GraphAttributionResult(
        prediction=prediction,
        method=method,
        node_attr=attr_x.detach().cpu(),
        edge_attr=attr_edge.detach().cpu(),
        node_scores=node_scores.detach().cpu(),
        edge_scores=edge_scores.detach().cpu(),
        atom_scores=atom_scores.detach().cpu(),
    )


def rank_atoms(atom_scores: torch.Tensor, top_k: int = 10,
               structure=None) -> list[dict]:
    """
    Return top-k atom indices sorted by descending influence.
    """
    scores = atom_scores.detach().cpu().float()
    top_k = max(1, min(top_k, scores.numel()))
    top_indices = torch.argsort(scores, descending=True)[:top_k]

    ranked = []
    for idx in top_indices.tolist():
        label = f"atom_{idx}"
        if structure is not None:
            try:
                label = f"{structure[idx].specie.symbol}{idx}"
            except Exception:
                pass
        ranked.append({
            "index": idx,
            "label": label,
            "score": float(scores[idx].item()),
        })
    return ranked


def plot_top_atom_importance(atom_scores: torch.Tensor, structure=None,
                             top_k: int = 15,
                             save_path: Optional[str] = None):
    """
    Plot a horizontal bar chart of top-k atom influence scores.
    """
    import matplotlib.pyplot as plt

    ranked = rank_atoms(atom_scores, top_k=top_k, structure=structure)
    labels = [r["label"] for r in ranked][::-1]
    scores = [r["score"] for r in ranked][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(labels) + 1.5)))
    ax.barh(labels, scores, color="#1f77b4", alpha=0.9)
    ax.set_xlabel("Atom influence score")
    ax.set_title("Top influential atoms for predicted voltage")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_atom_importance_3d(structure, atom_scores: torch.Tensor,
                            top_k_labels: int = 10,
                            save_path: Optional[str] = None):
    """
    Plot atom influence directly on 3D structure coordinates.
    """
    import matplotlib.pyplot as plt

    coords = np.asarray(structure.cart_coords)
    scores = atom_scores.detach().cpu().numpy().astype(float)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    sizes = 70.0 + 500.0 * norm_scores

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=norm_scores,
        cmap="inferno",
        s=sizes,
        alpha=0.95,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.72, label="Normalized influence")

    top_idx = np.argsort(-scores)[:max(1, min(top_k_labels, len(scores)))]
    for idx in top_idx:
        atom_label = f"{structure[idx].specie.symbol}{idx}"
        ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2], atom_label,
                fontsize=8)

    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.set_title("Atom-level influence map")
    plt.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
