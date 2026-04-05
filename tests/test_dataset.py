from __future__ import annotations

import torch

from mini_lewm.data.dataset import PointWorldTransitionDataset


def test_sample_returns_expected_keys_and_shapes(tmp_path) -> None:
    path = tmp_path / "dataset.pt"
    torch.save(
        {
            "observations": torch.zeros(4, 3, 64, 64),
            "next_observations": torch.zeros(4, 3, 64, 64),
            "actions": torch.zeros(4, 2),
            "states": torch.zeros(4, 2),
            "next_states": torch.zeros(4, 2),
        },
        path,
    )
    dataset = PointWorldTransitionDataset(str(path))
    sample = dataset[0]
    assert set(sample) == {"obs", "next_obs", "action", "state", "next_state"}
    assert sample["obs"].shape == (3, 64, 64)
    assert sample["action"].shape == (2,)
