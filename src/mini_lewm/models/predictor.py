from __future__ import annotations

import torch
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [latent_dim + action_dim] + [hidden_dim] * num_layers + [latent_dim]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


class GRUPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, action], dim=-1)
        x = self.input_proj(x).unsqueeze(1)
        output, _ = self.gru(x)
        return self.output(output[:, -1])
