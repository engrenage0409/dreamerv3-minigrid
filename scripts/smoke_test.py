from __future__ import annotations

import torch
from envs import make_env, preprocess_image
from dreamer import WorldModel


def main():
    env = make_env()
    obs, _ = env.reset(seed=123)
    image = preprocess_image(obs["image"])
    assert image.shape == (3, 7, 7)
    model = WorldModel(action_dim=env.action_space.n)
    images = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    actions = torch.zeros(1, 1, env.action_space.n)
    with torch.no_grad():
        out = model(images, actions)
    assert out["image_logits"].shape == images.shape
    assert out["reward"].shape == (1, 1)
    env.close()
    print("Smoke test passed: MiniGrid observation -> encoder -> RSSM -> decoders.")


if __name__ == "__main__":
    main()
