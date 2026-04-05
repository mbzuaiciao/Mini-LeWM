from __future__ import annotations

import argparse
from pathlib import Path

from mini_lewm.cli import add_config_args
from mini_lewm.data.storage import load_tensor_dataset
from mini_lewm.train import train_model
from mini_lewm.utils.config import load_config


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description="Train Mini-LeWM"))
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    dataset_path = Path(config["data"]["path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run scripts/generate_data.py first."
        )
    load_tensor_dataset(dataset_path)
    run_dir = train_model(config)
    print(run_dir)


if __name__ == "__main__":
    main()
