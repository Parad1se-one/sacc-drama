#!/usr/bin/env python3
"""Deterministic statistics whose independent unit is one training seed."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from statistics import mean

import numpy as np


BOOTSTRAP_SEED = 20260914
BOOTSTRAP_RESAMPLES = 100_000


def exact_two_sided_sign_flip_p(differences: Sequence[float]) -> float | None:
    """Exact paired randomization p-value for the absolute mean difference."""

    values = [float(value) for value in differences]
    if not values:
        return None
    observed = abs(mean(values))
    tolerance = 1e-15
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean(sign * value for sign, value in zip(signs, values)))
        total += 1
        if permuted + tolerance >= observed:
            extreme += 1
    return extreme / total


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
) -> list[float] | None:
    """Percentile bootstrap CI formed by resampling independent seeds."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    # Chunking bounds peak memory while preserving the single deterministic RNG stream.
    chunk_size = 10_000
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        estimates[start:stop] = [statistic(sample) for sample in array[indices]]
    alpha = 1.0 - confidence_level
    return [
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    ]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        estimates[start:stop] = np.mean(array[indices], axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the conventional 25%-trimmed mean used by rliable's IQM."""

    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        raise ValueError("IQM requires at least one value")
    trim = int(array.size * 0.25)
    retained = array[trim : array.size - trim]
    if retained.size == 0:
        raise ValueError("IQM trimming removed every value")
    return float(np.mean(retained))


def bootstrap_iqm_ci(
    values: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    """Percentile CI for the seed-level IQM, resampling seeds with replacement."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    trim = int(array.size * 0.25)
    if array.size - 2 * trim <= 0:
        raise ValueError("IQM trimming removed every value")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        samples = np.sort(array[indices], axis=1)
        estimates[start:stop] = np.mean(samples[:, trim : array.size - trim], axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-adjust valid p-values while preserving the input order."""

    adjusted: list[float | None] = [None] * len(p_values)
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(p_values) if value is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
