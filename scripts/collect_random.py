from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from envs import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="runs/random_dataset.npz")
    args = parser.parse_args()

    env = make_env(args.env_id)
    images, actions, rewards, dones, firsts = [], [], [], [], []
    rng = np.random.default_rng(args.seed)

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            first = True

            while not done:
                # ImgObsWrapper returns the image array directly.
                images.append(np.asarray(obs, dtype=np.uint8))
                action = int(rng.integers(env.action_space.n))
                actions.append(action)

                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards.append(float(reward))
                dones.append(done)
                firsts.append(first)
                first = False
    finally:
        env.close()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.asarray(images, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        is_first=np.asarray(firsts, dtype=np.bool_),
    )
    print(f"Saved {len(actions)} transitions from {args.episodes} episodes to {path}")
    print(f"Image array shape: {np.asarray(images).shape}")


if __name__ == "__main__":
    main()
