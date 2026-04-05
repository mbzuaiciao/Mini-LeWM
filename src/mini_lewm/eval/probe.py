from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from mini_lewm.data.dataset import create_dataloaders
from mini_lewm.models import WorldModel
from mini_lewm.train.checkpoints import load_checkpoint
from mini_lewm.utils.device import resolve_device


def evaluate_probe(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    config = checkpoint["config"]
    device = resolve_device(config["experiment"].get("device"))
    model = WorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataloader = create_dataloaders(config)["test"]
    latents = []
    states = []
    with torch.no_grad():
        for batch in dataloader:
            obs = batch["obs"].to(device)
            latent = model.encode(obs).cpu().numpy()
            latents.append(latent)
            states.append(batch["state"].numpy())

    x = np.concatenate(latents, axis=0)
    y = np.concatenate(states, axis=0)
    probe = LinearRegression().fit(x, y)
    score = float(probe.score(x, y))
    metrics = {"r2": score, "num_samples": int(x.shape[0])}

    output_path = run_dir / "probe" / "linear_probe_metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return output_path
