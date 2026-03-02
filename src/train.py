"""
Training loop, early stopping, and learning rate scheduling for CGCNN.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor validation loss and stop training when it stops improving.

    Args:
        patience:  number of epochs with no improvement before stopping
        min_delta: minimum change in val_loss to qualify as improvement
        mode:      "min" (stop when loss stops decreasing)
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0


# ---------------------------------------------------------------------------
# Core training functions
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Run one training epoch. Returns mean loss per sample."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> float:
    """Run one validation/test epoch. Returns mean loss per sample."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, targets) arrays for a full dataloader."""
    model.eval()
    preds, targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    return np.concatenate(preds), np.concatenate(targets)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_cgcnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 25,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    save_dir: Optional[str] = None,
    model_name: str = "cgcnn",
    verbose: bool = True,
) -> dict:
    """
    Full training loop for CGCNN with early stopping and ReduceLROnPlateau.

    Returns a history dict with keys: train_loss, val_loss, lr, best_epoch.
    """
    model = model.to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor,
                                   patience=lr_patience, verbose=verbose)
    early_stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_state = None

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < early_stopper.best_loss:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"train MAE: {train_loss:.4f} V | "
                  f"val MAE: {val_loss:.4f} V | "
                  f"lr: {current_lr:.2e} | "
                  f"time: {elapsed:.1f}s")

        if early_stopper(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}. "
                  f"Best val MAE: {early_stopper.best_loss:.4f} V "
                  f"(epoch {early_stopper.best_epoch})")
            break

    history["best_epoch"] = early_stopper.best_epoch

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / f"{model_name}_best.pt")
        with open(save_path / f"{model_name}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved model to {save_path / f'{model_name}_best.pt'}")

    return history


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def make_loaders(train_ds, val_ds, test_ds,
                 batch_size: int = 64,
                 num_workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from PyG datasets."""
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
