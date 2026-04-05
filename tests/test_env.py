from __future__ import annotations

import numpy as np

from mini_lewm.env import PointWorldEnv


def test_reset_returns_valid_state() -> None:
    env = PointWorldEnv()
    obs = env.reset(seed=123)
    assert obs.shape == (3, 64, 64)
    assert env.position.min() >= 0
    assert env.position.max() <= 64


def test_step_updates_position() -> None:
    env = PointWorldEnv()
    env.reset(seed=1)
    start = env.position.copy()
    _, next_state = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert next_state[0] >= start[0]


def test_render_shape() -> None:
    env = PointWorldEnv()
    env.reset(seed=0)
    render = env.render()
    assert render.shape == (3, 64, 64)
