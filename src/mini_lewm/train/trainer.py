from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_lewm.data import create_dataloaders
from mini_lewm.models import WorldModel
from mini_lewm.train.checkpoints import save_checkpoint
from mini_lewm.train.loops import run_epoch
from mini_lewm.utils.config import save_yaml
from mini_lewm.utils.device import resolve_device
from mini_lewm.utils.logging import append_jsonl, setup_logger
from mini_lewm.utils.paths import create_run_dir, system_info, try_git_commit
from mini_lewm.utils.seed import set_seed
from mini_lewm.viz.plots import plot_training_curves


def train_model(config: dict) -> Path:
    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["experiment"].get("device"))
    run_dir = create_run_dir(config["paths"]["runs_root"], config["experiment"]["name"])
    logger = setup_logger(run_dir / "train.log")
    save_yaml(config, run_dir / "config_resolved.yaml")

    model = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    dataloaders = create_dataloaders(config)

    best_val_pred_loss = float("inf")
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            config["loss"]["lambda_sigreg"],
            config["loss"]["sigreg"],
        )
        val_metrics = run_epoch(
            model,
            dataloaders["val"],
            None,
            device,
            config["loss"]["lambda_sigreg"],
            config["loss"]["sigreg"],
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pred_loss": train_metrics["pred_loss"],
            "train_sigreg_loss": train_metrics["sigreg_loss"],
            "train_latent_std_mean": train_metrics["latent_std_mean"],
            "val_loss": val_metrics["loss"],
            "val_pred_loss": val_metrics["pred_loss"],
            "val_sigreg_loss": val_metrics["sigreg_loss"],
            "val_latent_std_mean": val_metrics["latent_std_mean"],
        }
        history.append(record)
        append_jsonl(run_dir / "metrics.jsonl", record)
        logger.info(
            "epoch=%s train_loss=%.4f train_pred=%.4f train_sigreg=%.4f val_pred=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["pred_loss"],
            train_metrics["sigreg_loss"],
            val_metrics["pred_loss"],
        )

        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            model,
            optimizer,
            epoch,
            config,
            best_val_pred_loss,
        )
        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                best_val_pred_loss,
            )
        if val_metrics["pred_loss"] < best_val_pred_loss:
            best_val_pred_loss = val_metrics["pred_loss"]
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                best_val_pred_loss,
            )

    plot_training_curves(history, run_dir / "figures" / "losses.png")
    summary = {
        "run_name": run_dir.name,
        "seed": config["experiment"]["seed"],
        "git_commit": try_git_commit(),
        "device": str(device),
        "status": "completed",
        "system": system_info(),
        "best_val_pred_loss": best_val_pred_loss,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return run_dir
