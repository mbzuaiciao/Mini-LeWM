from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from mini_lewm.data.storage import load_tensor_dataset


class PointWorldTransitionDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.payload = load_tensor_dataset(path)

    def __len__(self) -> int:
        return self.payload["observations"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs": self.payload["observations"][index].float(),
            "next_obs": self.payload["next_observations"][index].float(),
            "action": self.payload["actions"][index].float(),
            "state": self.payload["states"][index].float(),
            "next_state": self.payload["next_states"][index].float(),
        }


def _split_lengths(total: int, train_split: float, val_split: float, test_split: float) -> list[int]:
    if not math.isclose(train_split + val_split + test_split, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Dataset splits must sum to 1.0")
    train_len = int(total * train_split)
    val_len = int(total * val_split)
    test_len = total - train_len - val_len
    return [train_len, val_len, test_len]


def create_dataloaders(config: dict) -> dict[str, DataLoader]:
    dataset = PointWorldTransitionDataset(config["data"]["path"])
    lengths = _split_lengths(
        len(dataset),
        config["data"]["train_split"],
        config["data"]["val_split"],
        config["data"]["test_split"],
    )
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    splits = torch.utils.data.random_split(dataset, lengths, generator=generator)
    names = ["train", "val", "test"]
    dataloaders = {}
    for name, split in zip(names, splits, strict=True):
        shuffle = name == "train"
        dataloaders[name] = DataLoader(
            split,
            batch_size=config["train"]["batch_size"],
            shuffle=shuffle,
            num_workers=config["train"]["num_workers"],
        )
    return dataloaders


def subset_from_indices(dataset: Dataset, indices: list[int]) -> Subset:
    return Subset(dataset, indices)
