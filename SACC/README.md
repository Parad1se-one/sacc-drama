# SACC

SACC is a DRAMA-specific structured actor-critic coordination method with one local progress tracker per agent.

## Included in this submission

- standalone `sacc` package
- DRAMA environment wrapper with per-agent progress features
- per-agent reward-machine-inspired progress trackers
- centralized per-agent critics with sequential policy updates
- checkpoint evaluation utilities

## Public benchmark scope

- `easy`
- `hard`

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick start

```bash
python -m sacc.experiments.train_drama           --difficulty easy           --episodes 200           --eval-interval 20           --eval-episodes 10           --seed 42

python -m sacc.experiments.train_drama           --difficulty hard           --episodes 400           --eval-interval 20           --eval-episodes 10           --seed 42
```

Greedy evaluation:

```bash
python -m sacc.utils.eval           --checkpoint results/drama/.../checkpoints/best_by_completion.pt           --episodes 20           --json-out results/drama/eval.json
```
