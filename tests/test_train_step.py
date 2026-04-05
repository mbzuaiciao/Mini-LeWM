from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mini_lewm.models import WorldModel
from mini_lewm.train.loops import run_epoch


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.randn(3, 64, 64),
            "next_obs": torch.randn(3, 64, 64),
            "action": torch.randn(2),
        }


def test_one_optimizer_step_runs() -> None:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = DataLoader(DummyDataset(), batch_size=2)
    metrics = run_epoch(
        model,
        dataloader,
        optimizer,
        device=torch.device("cpu"),
        lambda_sigreg=0.1,
        sigreg_cfg={"num_projections": 8, "eps": 1.0e-6},
    )
    assert "loss" in metrics
