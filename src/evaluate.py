"""
Evaluation metrics, publication-quality plots, and model comparison utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Publication style defaults
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette consistent across all figures
PALETTE = {
    "oxide": "#2196F3",       # blue
    "phosphate": "#4CAF50",   # green
    "sulfide": "#FF9800",     # orange
    "fluoride": "#9C27B0",    # purple
    "sulfate": "#F44336",     # red
    "silicate": "#009688",    # teal
    "other": "#607D8B",       # blue-grey
    "rf": "#E91E63",
    "cgcnn": "#2196F3",
    "m3gnet": "#4CAF50",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R-squared."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name:20s}  MAE: {metrics['MAE']:.4f} V  "
          f"RMSE: {metrics['RMSE']:.4f} V  R2: {metrics['R2']:.4f}")


# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------

def parity_plot(y_true: np.ndarray, y_pred: np.ndarray,
                labels: Optional[list[str]] = None,
                title: str = "Predicted vs Actual Voltage",
                model_name: str = "model",
                save_path: Optional[str] = None,
                figsize: tuple = (5.5, 5.5)) -> plt.Figure:
    """
    Parity plot (predicted vs actual voltage) optionally colored by chemistry family.
    Includes 1:1 reference line and displays MAE and R2 in the legend.
    """
    metrics = compute_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            color = PALETTE.get(lbl, "#607D8B")
            ax.scatter(y_true[mask], y_pred[mask], s=12, alpha=0.55,
                       label=lbl, color=color, linewidths=0)
        ax.legend(title="Chemistry", loc="upper left",
                  frameon=True, framealpha=0.8)
    else:
        color = PALETTE.get(model_name.lower(), "#2196F3")
        ax.scatter(y_true, y_pred, s=12, alpha=0.55,
                   color=color, linewidths=0)

    # 1:1 reference line
    vmin = min(y_true.min(), y_pred.min()) - 0.2
    vmax = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.0, alpha=0.6)

    # Annotation box
    ann_text = (f"MAE = {metrics['MAE']:.3f} V\n"
                f"RMSE = {metrics['RMSE']:.3f} V\n"
                f"R$^2$ = {metrics['R2']:.3f}")
    ax.text(0.97, 0.04, ann_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel("DFT Average Voltage (V vs Li/Li+)")
    ax.set_ylabel("Predicted Voltage (V vs Li/Li+)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved parity plot: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curves(history: dict, model_name: str = "CGCNN",
                     save_path: Optional[str] = None,
                     figsize: tuple = (7, 3.5)) -> plt.Figure:
    """
    Plot training and validation MAE loss curves with learning rate overlay.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    lrs = history.get("lr", [])
    best_epoch = history.get("best_epoch", np.argmin(val_loss))
    epochs = np.arange(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    color_train = PALETTE.get(model_name.lower(), "#2196F3")
    color_val = "#F44336"

    ax = axes[0]
    ax.plot(epochs, train_loss, color=color_train, label="Train MAE", linewidth=1.5)
    ax.plot(epochs, val_loss, color=color_val, label="Val MAE", linewidth=1.5)
    ax.axvline(best_epoch, color="black", linestyle=":", linewidth=1.0,
               alpha=0.7, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE Loss (V)")
    ax.set_title(f"{model_name}: Loss Curves")
    ax.legend(frameon=False)

    ax2 = axes[1]
    if lrs:
        ax2.semilogy(epochs, lrs, color="#607D8B", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
    else:
        ax2.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved loss curves: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Error analysis by chemistry family
# ---------------------------------------------------------------------------

def error_by_chemistry(y_true: np.ndarray, y_pred: np.ndarray,
                        labels: list[str],
                        save_path: Optional[str] = None,
                        figsize: tuple = (7, 4)) -> plt.Figure:
    """
    Bar chart of MAE per chemistry family, sorted descending.
    """
    families = sorted(set(labels))
    maes = []
    counts = []

    for fam in families:
        mask = np.array(labels) == fam
        if mask.sum() > 0:
            maes.append(mean_absolute_error(y_true[mask], y_pred[mask]))
            counts.append(mask.sum())
        else:
            maes.append(0.0)
            counts.append(0)

    order = np.argsort(maes)[::-1]
    families_sorted = [families[i] for i in order]
    maes_sorted = [maes[i] for i in order]
    counts_sorted = [counts[i] for i in order]
    colors = [PALETTE.get(f, "#607D8B") for f in families_sorted]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(families_sorted)), maes_sorted,
                   color=colors, alpha=0.85, edgecolor="none")

    for i, (mae, n) in enumerate(zip(maes_sorted, counts_sorted)):
        ax.text(mae + 0.005, i, f"n={n}", va="center", fontsize=9)

    ax.set_yticks(range(len(families_sorted)))
    ax.set_yticklabels(families_sorted)
    ax.set_xlabel("MAE (V vs Li/Li+)")
    ax.set_title("Prediction Error by Chemistry Family")
    ax.axvline(np.mean(maes), color="black", linestyle="--",
               linewidth=1.0, alpha=0.6, label="Overall MAE")
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved error by chemistry: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def model_comparison_chart(results: dict[str, dict],
                            save_path: Optional[str] = None,
                            figsize: tuple = (7, 4)) -> plt.Figure:
    """
    Grouped bar chart comparing MAE, RMSE across models.

    Args:
        results: dict mapping model_name -> metrics dict (keys: MAE, RMSE, R2)
    """
    models = list(results.keys())
    metrics_to_plot = ["MAE", "RMSE"]
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        offsets = x + (i - len(models) / 2 + 0.5) * width
        vals = [results[model][m] for m in metrics_to_plot]
        color = PALETTE.get(model.lower(), "#607D8B")
        ax.bar(offsets, vals, width=width * 0.9, label=model,
               color=color, alpha=0.85, edgecolor="none")
        for offset, val in zip(offsets, vals):
            ax.text(offset, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel("Error (V)")
    ax.set_title("Model Comparison: MAE and RMSE on Test Set")
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved comparison chart: {save_path}")
    return fig
