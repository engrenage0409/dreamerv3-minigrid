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
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        first = True
        while not done:
            images.append(obs["image"].astype(np.uint8))
            action = int(rng.integers(env.action_space.n))
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(float(reward))
            dones.append(done)
            firsts.append(first)
            first = False
    env.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.asarray(images), actions=np.asarray(actions),
                        rewards=np.asarray(rewards, np.float32),
                        dones=np.asarray(dones, np.bool_),
                        is_first=np.asarray(firsts, np.bool_))
    print(f"Saved {len(actions)} transitions to {path}")


if __name__ == "__main__":
    main()
