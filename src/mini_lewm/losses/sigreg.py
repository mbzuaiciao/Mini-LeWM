from __future__ import annotations

import torch


def sigreg_loss(latents: torch.Tensor, num_projections: int = 64, eps: float = 1.0e-6) -> torch.Tensor:
    latents = latents - latents.mean(dim=0, keepdim=True)
    batch_size, latent_dim = latents.shape
    projections = torch.randn(latent_dim, num_projections, device=latents.device, dtype=latents.dtype)
    projections = projections / (projections.norm(dim=0, keepdim=True) + eps)
    projected = latents @ projections
    std = torch.sqrt(projected.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.relu(1.0 - std))
