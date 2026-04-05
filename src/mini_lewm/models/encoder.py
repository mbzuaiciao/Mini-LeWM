from __future__ import annotations

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(current_channels, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            current_channels = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))
