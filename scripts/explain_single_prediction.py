#!/usr/bin/env python3
"""
Run Captum attribution on one crystal graph prediction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from pymatgen.core import Structure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import VoltageGraphDataset
from src.explain import (
    explain_single_graph_prediction,
    plot_atom_importance_3d,
    plot_top_atom_importance,
    rank_atoms,
)
from src.models import CGCNNModel, CrystalTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a single voltage prediction with Captum."
    )
    parser.add_argument(
        "--model",
        choices=["transformer", "cgcnn"],
        default="transformer",
        help="Model architecture to load.",
    )
    parser.add_argument(
        "--method",
        choices=["ig", "gradient_shap"],
        default="ig",
        help="Attribution method: Integrated Gradients or SHAP-like GradientShap.",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=0,
        help="Index of graph inside data/test_graphs.pkl.",
    )
    parser.add_argument(
        "--graphs-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_graphs.pkl",
        help="Path to processed test graphs pickle.",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits.pkl",
        help="Path to dataset splits pickle (for metadata and structure labels).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Defaults to models/<model>_best.pt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many most influential atoms to print and plot.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Integrated gradients integration steps.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="GradientShap sample count.",
    )
    parser.add_argument(
        "--stdevs",
        type=float,
        default=0.01,
        help="GradientShap noise scale.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for explainability outputs.",
    )
    parser.add_argument(
        "--skip-3d-plot",
        action="store_true",
        help="Disable 3D structure scatter output.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str, node_dim: int, edge_dim: int) -> torch.nn.Module:
    if model_name == "transformer":
        return CrystalTransformer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=128,
            n_conv=4,
            heads=4,
            dropout=0.1,
        )
    return CGCNNModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        n_conv=4,
        dropout=0.1,
    )


def load_matching_structure(graph, splits_path: Path, graph_index: int):
    if not splits_path.exists():
        return None, None
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    test_entries = splits.get("test", []) if isinstance(splits, dict) else []
    battery_id = getattr(graph, "battery_id", None)
    formula = getattr(graph, "formula", None)

    matched_entry = None
    if battery_id is not None:
        for entry in test_entries:
            if entry.get("battery_id") == battery_id:
                matched_entry = entry
                break

    if matched_entry is None:
        if 0 <= graph_index < len(test_entries):
            matched_entry = test_entries[graph_index]

    if matched_entry is None:
        return None, {
            "battery_id": battery_id,
            "formula": formula,
        }

    struct_dict = (
        matched_entry.get("structure")
        or matched_entry.get("charged_structure")
        or matched_entry.get("discharged_structure")
    )
    structure = Structure.from_dict(struct_dict) if struct_dict else None
    metadata = {
        "battery_id": matched_entry.get("battery_id", battery_id),
        "formula": matched_entry.get("formula", formula),
        "average_voltage": matched_entry.get("average_voltage"),
        "chemistry_family": matched_entry.get("chemistry_family"),
    }
    return structure, metadata


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = VoltageGraphDataset.from_processed(str(args.graphs_path))
    if args.graph_index < 0 or args.graph_index >= len(dataset):
        raise IndexError(
            f"--graph-index {args.graph_index} out of bounds for {len(dataset)} graphs."
        )
    graph = dataset[args.graph_index]

    node_dim = int(graph.x.shape[1])
    edge_dim = int(graph.edge_attr.shape[1]) if graph.edge_attr.ndim > 1 else 1
    model = build_model(args.model, node_dim=node_dim, edge_dim=edge_dim)

    if args.model_path is None:
        ckpt_name = "transformer_best.pt" if args.model == "transformer" else "cgcnn_best.pt"
        model_path = PROJECT_ROOT / "models" / ckpt_name
    else:
        model_path = args.model_path

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    structure, metadata = load_matching_structure(
        graph=graph, splits_path=args.splits_path, graph_index=args.graph_index
    )
    if metadata is None:
        metadata = {
            "battery_id": getattr(graph, "battery_id", None),
            "formula": getattr(graph, "formula", None),
        }

    result = explain_single_graph_prediction(
        model=model,
        graph=graph,
        method=args.method,
        n_steps=args.n_steps,
        n_samples=args.n_samples,
        stdevs=args.stdevs,
        normalize=True,
        device=device,
    )
    ranked = rank_atoms(result.atom_scores, top_k=args.top_k, structure=structure)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Graph index: {args.graph_index}")
    print(f"Battery ID: {metadata.get('battery_id')}")
    print(f"Formula: {metadata.get('formula')}")
    print(f"Predicted voltage: {result.prediction:.4f} V")
    print("\nTop influential atoms:")
    for i, item in enumerate(ranked, start=1):
        print(f"{i:2d}. {item['label']:>10s}  score={item['score']:.5f}")

    stem = f"explain_{args.model}_{args.method}_idx{args.graph_index:04d}"
    bar_path = args.results_dir / f"{stem}_top_atoms.png"
    fig = plot_top_atom_importance(
        result.atom_scores,
        structure=structure,
        top_k=args.top_k,
        save_path=str(bar_path),
    )
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass

    structure_plot_path = None
    if structure is not None and not args.skip_3d_plot:
        structure_plot_path = args.results_dir / f"{stem}_structure_3d.png"
        fig3d = plot_atom_importance_3d(
            structure=structure,
            atom_scores=result.atom_scores,
            top_k_labels=args.top_k,
            save_path=str(structure_plot_path),
        )
        try:
            import matplotlib.pyplot as plt
            plt.close(fig3d)
        except Exception:
            pass

    payload = {
        "model": args.model,
        "method": args.method,
        "graph_index": args.graph_index,
        "metadata": metadata,
        "prediction_voltage_V": result.prediction,
        "top_atoms": ranked,
        "atom_scores": result.atom_scores.tolist(),
        "node_scores": result.node_scores.tolist(),
        "edge_scores": result.edge_scores.tolist(),
        "bar_plot": str(bar_path),
        "structure_plot_3d": str(structure_plot_path) if structure_plot_path else None,
    }
    json_path = args.results_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved: {bar_path}")
    if structure_plot_path:
        print(f"Saved: {structure_plot_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
