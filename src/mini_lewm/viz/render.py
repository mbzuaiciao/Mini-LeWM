from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_rollout_grid(obs: torch.Tensor, next_obs: torch.Tensor, path: str | Path) -> None:
    num_items = min(len(obs), len(next_obs), 8)
    fig, axes = plt.subplots(2, num_items, figsize=(2 * num_items, 4))
    for idx in range(num_items):
        axes[0, idx].imshow(obs[idx].permute(1, 2, 0).numpy())
        axes[0, idx].axis("off")
        axes[1, idx].imshow(next_obs[idx].permute(1, 2, 0).numpy())
        axes[1, idx].axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
