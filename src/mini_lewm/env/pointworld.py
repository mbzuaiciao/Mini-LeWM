from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointWorldConfig:
    image_size: int = 64
    channels: int = 3
    dot_radius: int = 2
    step_scale: float = 2.5
    action_dim: int = 2
    use_wall: bool = False


class PointWorldEnv:
    def __init__(self, **kwargs) -> None:
        self.config = PointWorldConfig(**kwargs)
        self.rng = np.random.default_rng()
        self.position = np.zeros(2, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        margin = self.config.dot_radius + 1
        self.position = self.rng.uniform(
            low=margin,
            high=self.config.image_size - margin,
            size=2,
        ).astype(np.float32)
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        proposed = self.position + action[:2] * self.config.step_scale
        margin = self.config.dot_radius + 1
        proposed = np.clip(proposed, margin, self.config.image_size - margin)
        if self.config.use_wall:
            wall_x = self.config.image_size / 2.0
            crossed = (self.position[0] < wall_x <= proposed[0]) or (
                self.position[0] > wall_x >= proposed[0]
            )
            if crossed and abs(self.position[1] - self.config.image_size / 2.0) > 8:
                proposed[0] = self.position[0]
        self.position = proposed
        return self.render(), self.position.copy()

    def sample_action(self) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(self.config.action_dim,)).astype(np.float32)

    def render(self) -> np.ndarray:
        image = np.zeros(
            (self.config.channels, self.config.image_size, self.config.image_size), dtype=np.float32
        )
        yy, xx = np.ogrid[: self.config.image_size, : self.config.image_size]
        dist_sq = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2
        mask = dist_sq <= self.config.dot_radius**2
        image[0, mask] = 1.0
        image[1, mask] = 0.3
        image[2, mask] = 0.1
        if self.config.use_wall:
            wall_x = self.config.image_size // 2
            image[:, :, wall_x] = 0.5
            gap_center = self.config.image_size // 2
            image[:, gap_center - 8 : gap_center + 8, wall_x] = 0.0
        return image
