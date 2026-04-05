from __future__ import annotations

import torch
from torch import nn


class IdentityProjector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
