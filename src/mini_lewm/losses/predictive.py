from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_mse_loss(pred_next_latent: torch.Tensor, target_next_latent: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_next_latent, target_next_latent)
