from __future__ import annotations

import argparse


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key, e.g. loss.lambda_sigreg=0.0",
    )
    return parser
