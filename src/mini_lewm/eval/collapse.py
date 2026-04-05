from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_lewm.data.dataset import create_dataloaders
from mini_lewm.models import WorldModel
from mini_lewm.train.checkpoints import load_checkpoint
from mini_lewm.utils.device import resolve_device
from mini_lewm.viz.plots import plot_latent_std


def evaluate_collapse(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    config = checkpoint["config"]
    device = resolve_device(config["experiment"].get("device"))
    model = WorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataloader = create_dataloaders(config)["test"]
    latents = []
    with torch.no_grad():
        for batch in dataloader:
            obs = batch["obs"].to(device)
            latents.append(model.encode(obs).cpu())
    latents = torch.cat(latents, dim=0)
    latent_std = latents.std(dim=0)
    output_path = run_dir / "figures" / "latent_std.png"
    plot_latent_std(latent_std.numpy(), output_path)

    summary_path = run_dir / "collapse.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "latent_std_mean": float(latent_std.mean()),
                "latent_std_min": float(latent_std.min()),
                "latent_std_max": float(latent_std.max()),
            },
            handle,
            indent=2,
        )
    return output_path
