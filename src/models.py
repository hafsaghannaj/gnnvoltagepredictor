"""
Model definitions: CGCNN (PyTorch Geometric) and M3GNet fine-tuning wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool


# ---------------------------------------------------------------------------
# CGCNN
# ---------------------------------------------------------------------------

class CGCNNModel(nn.Module):
    """
    Crystal Graph Convolutional Neural Network for regression.

    Architecture:
        1. Linear embedding of node features to hidden_dim
        2. n_conv CGConv layers with batch normalization and ReLU
        3. Global mean pooling over atoms
        4. Two-layer MLP output head (Softplus activation)

    Reference:
        Xie, T. and Grossman, J.C. (2018) Crystal graph convolutional neural
        networks for an accurate and interpretable prediction of material properties.
        Physical Review Letters 120, 145301.
    """

    def __init__(self, node_dim: int = 9, edge_dim: int = 64,
                 hidden_dim: int = 128, n_conv: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_dim, aggr="mean")
            for _ in range(n_conv)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_conv)
        ])

        self.dropout = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_embedding(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # skip connection

        x = global_mean_pool(x, batch)
        out = self.output_head(x)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# M3GNet fine-tuning wrapper
# ---------------------------------------------------------------------------

class M3GNetVoltagePredictor(nn.Module):
    """
    Wraps a pretrained M3GNet backbone from matgl for voltage regression.

    The original output head is replaced with a new regression head trained
    on the battery voltage dataset. During fine-tuning, the backbone weights
    are frozen for the first warmup_epochs epochs; then all weights are
    unfrozen for joint fine-tuning at a lower learning rate.

    Usage:
        model = M3GNetVoltagePredictor.from_pretrained()
        model.freeze_backbone()   # call before epoch 1
        model.unfreeze_backbone() # call after warmup_epochs
    """

    def __init__(self, backbone, backbone_output_dim: int = 64,
                 hidden_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.regression_head = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def from_pretrained(cls, model_name: str = "M3GNet-MP-2021.2.8-PES",
                         hidden_dim: int = 64) -> "M3GNetVoltagePredictor":
        import matgl
        backbone = matgl.load_model(model_name)
        # Determine backbone output dim by inspecting the readout layer
        try:
            backbone_out = backbone.model.final_layer.out_features
        except AttributeError:
            backbone_out = 64  # safe default
        return cls(backbone, backbone_output_dim=backbone_out, hidden_dim=hidden_dim)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only train the regression head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for joint fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for joint fine-tuning.")

    def forward(self, graph, lattice, state_attr):
        features = self.backbone.model(graph, lattice, state_attr)
        return self.regression_head(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_cgcnn(node_dim: int = 9, edge_dim: int = 64,
                hidden_dim: int = 128, n_conv: int = 4,
                dropout: float = 0.1) -> CGCNNModel:
    """Convenience factory for CGCNN."""
    return CGCNNModel(node_dim=node_dim, edge_dim=edge_dim,
                      hidden_dim=hidden_dim, n_conv=n_conv, dropout=dropout)
