"""Compact 1D-CNN over the 90 CSI amplitude streams."""

from __future__ import annotations

import torch
from torch import nn

from csi_har.config import N_CHANNELS, N_CLASSES


def _block(c_in: int, c_out: int, kernel: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool1d(2))
    return nn.Sequential(*layers)


class HARCNN(nn.Module):
    """Input ``(batch, 90, time)`` -> logits ``(batch, n_classes)``.

    Global average pooling makes the network accept any sequence length.
    """

    def __init__(self, in_channels: int = N_CHANNELS, n_classes: int = N_CLASSES,
                 width: int = 64, dropout: float = 0.3):
        super().__init__()
        self.hparams = {"in_channels": in_channels, "n_classes": n_classes, "width": width, "dropout": dropout}
        self.features = nn.Sequential(
            _block(in_channels, width, 7, pool=True),
            _block(width, 2 * width, 5, pool=True),
            _block(2 * width, 2 * width, 3, pool=True),
            _block(2 * width, 2 * width, 3, pool=False),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
