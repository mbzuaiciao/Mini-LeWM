from __future__ import annotations

import argparse

from mini_lewm.cli import add_config_args
from mini_lewm.data import generate_dataset
from mini_lewm.utils.config import load_config


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description="Generate PointWorld data"))
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    paths = generate_dataset(config)
    print(f"dataset={paths['dataset_path']}")
    print(f"manifest={paths['manifest_path']}")


if __name__ == "__main__":
    main()
