# DreamerV3 MiniGrid

A compact research project for visual navigation in partially observable MiniGrid environments using a Dreamer-style discrete latent world model.

> **Current status: Phase 1 scaffold.** Environment integration, random episode collection, encoder, discrete RSSM, prediction heads, world-model losses and smoke tests are present. The actor-critic imagination learner and trained policy are not implemented yet. No performance results are claimed.

## Goal

Learn from egocentric image observations and discrete actions. The intended full agent will learn a recurrent latent dynamics model and improve its policy using imagined rollouts in latent space.

## Milestones

1. **Environment and model foundation (current)** — MiniGrid wrapper, image preprocessing, random data collection, convolutional encoder, discrete RSSM, image/reward/continuation heads, balanced KL loss and smoke tests.
2. **World-model training** — episode sequence dataset, batching, optimizer, checkpointing and reconstruction/prediction diagnostics.
3. **Behavior learning** — discrete actor-critic trained on imagined trajectories with lambda returns.
4. **Experiments** — DoorKey curriculum, multiple seeds, success rate, episode length, learning curves and ablations.

## Layout

```text
configs/          Experiment configuration
dreamer/          Encoder, RSSM and world model
envs/             MiniGrid environment and preprocessing
scripts/          Data collection and smoke-test entry points
tests/            Unit tests
results/          Generated metrics and plots (not checkpoints)
```

## Install

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Phase 1 checks

```bash
python scripts/smoke_test.py
pytest -q
python scripts/collect_random.py --episodes 100 --output runs/random_dataset.npz
```

The initial environment is `MiniGrid-DoorKey-8x8-v0`. MiniGrid provides a 7×7×3 egocentric symbolic image and a discrete action space; the wrapper exposes only the image to the model in this baseline.

## Evaluation protocol

The final policy will be evaluated on held-out environment seeds. Report success rate, mean episode return and episode length across multiple seeds; do not use training return as the final score.

## Reproducibility and scope

Record config, random seed, code revision, environment ID and evaluation episode count for each run. Checkpoints, replay buffers, caches and raw training logs are excluded from Git.

This is an independent educational/research implementation, not an official DreamerV3 release or a claim of exact reproduction. Upstream components and licenses will be documented before any third-party implementation is incorporated.
