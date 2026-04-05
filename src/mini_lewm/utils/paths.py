from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(runs_root: str | Path, experiment_name: str) -> Path:
    run_dir = Path(runs_root) / f"{timestamp()}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ["checkpoints", "figures", "probe"]:
        (run_dir / child).mkdir(exist_ok=True)
    return run_dir


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def try_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def system_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
