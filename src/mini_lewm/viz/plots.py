from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: list[dict], path: str | Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="train_loss")
    plt.plot(epochs, [entry["val_loss"] for entry in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_latent_std(latent_std: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(latent_std)), latent_std)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Std")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
