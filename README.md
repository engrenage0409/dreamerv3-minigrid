# DreamerV3 MiniGrid

A compact research project for visual navigation in partially observable MiniGrid environments using a DreamerV3-style latent world model.

> Status: project scaffold in progress. No training results are claimed until experiments are run and evaluated.

## Goal

Train an agent from egocentric image observations and discrete actions. The agent learns a recurrent latent dynamics model and improves its policy using imagined rollouts in latent space.

## Planned milestones

1. **Environment and data pipeline** — MiniGrid wrappers, image observations, episode collection, deterministic evaluation.
2. **World model** — convolutional encoder/decoder, recurrent state-space model (RSSM), reward and continuation prediction, balanced KL objective.
3. **Behavior learning** — discrete actor-critic trained on imagined trajectories with lambda returns.
4. **Experiments** — DoorKey and harder navigation tasks, multiple seeds, success rate, episode length, learning curves and ablations.

## Repository layout

```text
configs/          Experiment configurations
dreamer/          Model, distributions, losses and agent
envs/             MiniGrid environment and wrappers
scripts/          Training, evaluation and visualization entry points
tests/            Unit and smoke tests
results/          Generated metrics and plots (not checkpoints)
```

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py
```

## Training

The first milestone will expose a reproducible command similar to:

```bash
python -m scripts.train --config configs/doorkey.yaml
python -m scripts.evaluate --checkpoint runs/doorkey/best.pt --episodes 100
```

The exact training command and expected runtime will be documented after the end-to-end smoke test passes.

## Metrics

- **Success rate**: fraction of evaluation episodes that reach the goal.
- **Mean episode return** and **mean episode length**.
- **Seed-level mean ± standard deviation** for final comparisons.

## Reproducibility

Every experiment should record its config, random seed, code revision, environment ID and evaluation episode count. Checkpoints, replay buffers, caches and raw training logs are intentionally excluded from Git.

## Attribution

This repository is an independent educational/research implementation. It is not an official DreamerV3 release and does not claim exact reproduction of every detail in the original paper. References and implementation provenance will be recorded as upstream components are selected.

## License

A project license will be added after the provenance and licensing of any reused upstream code have been reviewed.
