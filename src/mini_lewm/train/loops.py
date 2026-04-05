from __future__ import annotations

import torch

from mini_lewm.losses import predictive_mse_loss, sigreg_loss


def run_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_sigreg: float,
    sigreg_cfg: dict,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0, "latent_std_mean": 0.0}
    steps = 0

    for batch in dataloader:
        obs = batch["obs"].to(device)
        next_obs = batch["next_obs"].to(device)
        action = batch["action"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(obs, action)
            target_next_latent = model.encode(next_obs).detach()
            pred_loss = predictive_mse_loss(outputs["pred_next_latent"], target_next_latent)
            reg_loss = sigreg_loss(outputs["latent"], **sigreg_cfg)
            loss = pred_loss + lambda_sigreg * reg_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        latent_std_mean = outputs["latent"].std(dim=0).mean().item()
        totals["loss"] += loss.item()
        totals["pred_loss"] += pred_loss.item()
        totals["sigreg_loss"] += reg_loss.item()
        totals["latent_std_mean"] += latent_std_mean
        steps += 1

    if steps == 0:
        return totals
    return {key: value / steps for key, value in totals.items()}
