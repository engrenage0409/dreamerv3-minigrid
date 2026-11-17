from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
import numpy as np
from minigrid.wrappers import ImgObsWrapper


def make_env(env_id: str = "MiniGrid-DoorKey-8x8-v0", render_mode: str | None = None):
    """Create a partially observable MiniGrid environment.

    ImgObsWrapper exposes the egocentric 7x7x3 symbolic image only; direction and
    mission text are deliberately not leaked to the agent in this first baseline.
    """
    env = gym.make(env_id, render_mode=render_mode)
    return ImgObsWrapper(env)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert MiniGrid's uint8 HWC observation into float32 CHW in [0, 1]."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    return image.astype(np.float32).transpose(2, 0, 1) / 255.0
