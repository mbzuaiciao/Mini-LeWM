from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    best_val_pred_loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_pred_loss": best_val_pred_loss,
        },
        Path(path),
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(Path(path), map_location=device)
