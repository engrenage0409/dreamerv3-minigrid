from __future__ import annotations

import torch

from envs import make_env, preprocess_image
from dreamer import WorldModel


def main():
    env = make_env()
    try:
        # ImgObsWrapper returns the image array directly (not {"image": ...}).
        obs, _ = env.reset(seed=123)
        image = preprocess_image(obs)
        assert image.shape == (3, 7, 7), image.shape

        model = WorldModel(action_dim=env.action_space.n)
        images = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        actions = torch.zeros(1, 1, env.action_space.n)

        with torch.no_grad():
            out = model(images, actions)

        assert out["image_logits"].shape == images.shape
        assert out["reward"].shape == (1, 1)
        assert out["continue_logits"].shape == (1, 1)

        print("Observation:", tuple(obs.shape), obs.dtype)
        print("Action space:", env.action_space)
        print("World model image output:", tuple(out["image_logits"].shape))
        print("Smoke test passed: MiniGrid -> encoder -> RSSM -> prediction heads.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
