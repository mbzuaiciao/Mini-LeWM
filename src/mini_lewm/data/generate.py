from __future__ import annotations

from pathlib import Path

import torch

from mini_lewm.env import PointWorldEnv
from mini_lewm.utils.config import save_yaml


def generate_dataset(config: dict) -> dict[str, Path]:
    env = PointWorldEnv(**config["env"])
    data_cfg = config["data"]
    seed = config["experiment"]["seed"]
    env.reset(seed=seed)

    observations = []
    next_observations = []
    actions = []
    states = []
    next_states = []
    episode_ids = []

    for episode_idx in range(data_cfg["num_trajectories"]):
        obs = env.reset(seed=seed + episode_idx)
        for _ in range(data_cfg["trajectory_length"]):
            action = env.sample_action()
            state = env.position.copy()
            next_obs, next_state = env.step(action)
            observations.append(torch.from_numpy(obs))
            next_observations.append(torch.from_numpy(next_obs))
            actions.append(torch.from_numpy(action))
            states.append(torch.from_numpy(state))
            next_states.append(torch.from_numpy(next_state))
            episode_ids.append(episode_idx)
            obs = next_obs

    payload = {
        "observations": torch.stack(observations),
        "next_observations": torch.stack(next_observations),
        "actions": torch.stack(actions),
        "states": torch.stack(states),
        "next_states": torch.stack(next_states),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
        "metadata": {
            "name": data_cfg["name"],
            "num_trajectories": data_cfg["num_trajectories"],
            "trajectory_length": data_cfg["trajectory_length"],
            "seed": seed,
        },
    }

    dataset_path = Path(data_cfg["path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dataset_path)

    manifest_path = Path(config["paths"]["data_root"]) / "manifests" / f"{data_cfg['name']}.yaml"
    save_yaml(
        {
            "name": data_cfg["name"],
            "num_trajectories": data_cfg["num_trajectories"],
            "trajectory_length": data_cfg["trajectory_length"],
            "image_size": config["env"]["image_size"],
            "dot_radius": config["env"]["dot_radius"],
            "step_scale": config["env"]["step_scale"],
            "seed": seed,
        },
        manifest_path,
    )
    return {"dataset_path": dataset_path, "manifest_path": manifest_path}
