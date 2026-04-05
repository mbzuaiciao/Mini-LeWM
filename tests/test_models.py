from __future__ import annotations

import torch

from mini_lewm.models import ConvEncoder, MLPPredictor, WorldModel


def test_encoder_output_shape() -> None:
    encoder = ConvEncoder(in_channels=3, hidden_dims=[16, 32], latent_dim=32)
    x = torch.randn(2, 3, 64, 64)
    z = encoder(x)
    assert z.shape == (2, 32)


def test_predictor_output_shape() -> None:
    predictor = MLPPredictor(latent_dim=32, action_dim=2, hidden_dim=64, num_layers=2)
    pred = predictor(torch.randn(2, 32), torch.randn(2, 2))
    assert pred.shape == (2, 32)


def test_world_model_shapes() -> None:
    config = {
        "env": {"channels": 3, "action_dim": 2},
        "model": {
            "type": "cnn_mlp",
            "latent_dim": 16,
            "encoder": {"hidden_dims": [16, 32]},
            "predictor": {"hidden_dim": 32, "num_layers": 2},
        },
    }
    model = WorldModel(config)
    outputs = model(torch.randn(4, 3, 64, 64), torch.randn(4, 2))
    assert outputs["latent"].shape == (4, 16)
    assert outputs["pred_next_latent"].shape == (4, 16)
