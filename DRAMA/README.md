# DRAMA

DRAMA is the benchmark and baseline package used by the SACC submission.

## Scope in this submission

This package exposes two public benchmark splits:

- `easy`: 2 task cells, 2 regions, 0 obstacles
- `hard`: 10 task cells, 4 regions, 1 obstacle

## Included components

- standalone benchmark package under `DRAMA/`
- grid environment under `envs/`
- baseline algorithms under `marl/algos/`
- generic training entrypoints:
  - `train.py` for RM-enhanced methods
  - `train_baseline.py` for plain baselines
- evaluation helpers under `utils/`

## Quick start

```bash
python -m pip install -r requirements.txt

python train.py --algo rmix --difficulty easy --seed 42
python train.py --algo rmix --difficulty hard --seed 42

python train_baseline.py --algo qmix --difficulty easy --seed 42
python train_baseline.py --algo ippo --difficulty hard --seed 42
```
