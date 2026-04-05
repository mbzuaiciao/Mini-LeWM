from __future__ import annotations

from pathlib import Path

import torch


def load_tensor_dataset(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu")
