from __future__ import annotations

import torch
from torch import nn

from mini_lewm.models.encoder import ConvEncoder
from mini_lewm.models.predictor import GRUPredictor, MLPPredictor
from mini_lewm.models.projector import IdentityProjector


class WorldModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        env_cfg = config["env"]
        model_cfg = config["model"]
        latent_dim = model_cfg["latent_dim"]
        self.encoder = ConvEncoder(
            in_channels=env_cfg["channels"],
            hidden_dims=model_cfg["encoder"]["hidden_dims"],
            latent_dim=latent_dim,
        )
        predictor_type = model_cfg.get("type", "cnn_mlp")
        if predictor_type == "cnn_gru":
            self.predictor = GRUPredictor(
                latent_dim=latent_dim,
                action_dim=env_cfg["action_dim"],
                hidden_dim=model_cfg["predictor"]["hidden_dim"],
                num_layers=model_cfg["predictor"]["num_layers"],
            )
        else:
            self.predictor = MLPPredictor(
                latent_dim=latent_dim,
                action_dim=env_cfg["action_dim"],
                hidden_dim=model_cfg["predictor"]["hidden_dim"],
                num_layers=model_cfg["predictor"]["num_layers"],
            )
        self.projector = IdentityProjector()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(obs))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(obs)
        pred_next_latent = self.predictor(latent, action)
        return {"latent": latent, "pred_next_latent": pred_next_latent}
