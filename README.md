# SACC + DRAMA

A compact public release for staged multi-robot construction learning:

- `DRAMA/` provides the benchmark environment and baseline implementations.
- `SACC/` provides the structured actor-critic coordination method built for this benchmark.

This repository is organized as a code-first release for training, evaluation, and reproduction. Heavy local experiment artifacts are intentionally excluded.

## Highlights

- Two public benchmark splits: `easy` and `hard`
- Unified DRAMA benchmark package under `DRAMA/DRAMA`
- Multiple MARL baselines under `DRAMA/marl/algos`
- Standalone SACC implementation under `SACC/sacc`
- Generic train and evaluation entrypoints for both benchmark baselines and SACC

## Repository Layout

```text
.
├── DRAMA/
│   ├── DRAMA/              # benchmark registry and environment package
│   ├── envs/               # grid environment
│   ├── marl/algos/         # baseline algorithms
│   ├── reward_machines/    # reward-machine utilities used by DRAMA methods
│   ├── train.py            # RM-enhanced training entrypoint
│   ├── train_baseline.py   # plain baseline training entrypoint
│   └── utils/              # evaluation and helper utilities
└── SACC/
    ├── sacc/               # SACC package
    ├── scripts/            # monitoring utilities
    └── tests/              # smoke/unit tests
```

## Benchmark Splits

| Split | Task Cells | Regions | Obstacles | Intended Use |
| --- | ---: | ---: | ---: | --- |
| `easy` | 2 | 2 | 0 | sanity checks, rapid iteration |
| `hard` | 10 | 4 | 1 | main benchmark setting |

## Installation

Install DRAMA:

```bash
cd DRAMA
python -m pip install -r requirements.txt
```

Install SACC:

```bash
cd ../SACC
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quick Start

Train a DRAMA baseline:

```bash
cd DRAMA
python train_baseline.py --algo qmix --difficulty easy --episodes 3000 --seed 42
python train_baseline.py --algo ippo --difficulty hard --episodes 1500 --seed 42
```

Train SACC:

```bash
cd ../SACC
python -m sacc.experiments.train_drama --difficulty easy --episodes 3000 --seed 42
python -m sacc.experiments.train_drama --difficulty hard --episodes 1500 --seed 42
```

Evaluate a SACC checkpoint:

```bash
python -m sacc.utils.eval \
  --checkpoint results/drama/.../checkpoints/best_by_completion.pt \
  --episodes 20 \
  --json-out results/drama/eval.json
```

## Included in This Release

- benchmark definitions
- environment code
- baseline training code
- SACC implementation
- evaluation and monitoring utilities

## Intentionally Excluded

- training logs
- TensorBoard event files
- checkpoints
- historical result directories
- paper-only assets
