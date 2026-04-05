from __future__ import annotations

import argparse

from mini_lewm.eval import evaluate_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rollout visuals")
    parser.add_argument("--run_dir", required=True, help="Path to a training run directory")
    args = parser.parse_args()
    output_path = evaluate_rollout(args.run_dir)
    print(output_path)


if __name__ == "__main__":
    main()
