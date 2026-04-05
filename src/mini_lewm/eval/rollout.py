from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_lewm.data.dataset import create_dataloaders
from mini_lewm.models import WorldModel
from mini_lewm.train.checkpoints import load_checkpoint
from mini_lewm.utils.device import resolve_device
from mini_lewm.viz.render import save_rollout_grid


def evaluate_rollout(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    config = checkpoint["config"]
    device = resolve_device(config["experiment"].get("device"))
    model = WorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataloader = create_dataloaders(config)["test"]
    batch = next(iter(dataloader))
    obs = batch["obs"][:16].to(device)
    next_obs = batch["next_obs"][:16]
    action = batch["action"][:16].to(device)

    with torch.no_grad():
        pred = model(obs, action)
        pred_norm = pred["pred_next_latent"].norm(dim=-1).mean().item()

    output_path = run_dir / "figures" / "rollout_examples.png"
    save_rollout_grid(obs.cpu(), next_obs.cpu(), output_path)

    summary_path = run_dir / "figures" / "rollout_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({"pred_next_latent_norm_mean": pred_norm}, handle, indent=2)
    return output_path
