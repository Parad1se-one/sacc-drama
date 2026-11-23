#!/usr/bin/env python3
"""Plot audited seed-level ablation learning curves on the selection starts.

Input is a frozen, plot-facing extraction of the 100 run records and 2,500
evaluation JSON files. The 20 evaluation starts are the same within each
split; they are not blind final-test episodes. No display smoothing is used.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STYLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STYLE_DIR))
from paper_plot_style import (  # noqa: E402
    BLUE,
    GOLD,
    ORANGE,
    POINT,
    RED,
    assert_artists_inside,
    configure_style,
    save_figure,
    style_axis,
)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "tenseed_ablation_selection_curves.json"
SEED_SUMMARY = ROOT / "tenseed_ablation_seed_summary.csv"
OUTPUT = ROOT / "tenseed_ablation_selection_curves"
VARIANTS = (
    ("full", "Full", BLUE, "-"),
    ("no_actor_progress", "No actor", ORANGE, "--"),
    ("no_critic_progress", "No critic", RED, "-."),
    ("no_structured_reward", "No reward", GOLD, ":"),
    ("all_off", "All off", POINT, (0, (5, 2))),
)
SEEDS = (42, 123, 305, 456, 789, 2024, 207668, 459207, 649032, 904400)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260920


def read_curves() -> dict:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["metric"] == "mean_completion_ratio"
    assert data["statistical_unit"] == "training_seed"
    assert len(data["runs"]) == 100
    assert set(data["episode_grids"]) == {"easy", "hard"}
    assert set(data["source_reports"]) == {"easy", "hard"}
    for split, expected_len in (("easy", 20), ("hard", 30)):
        grid = data["episode_grids"][split]
        assert len(grid) == expected_len
        expected_grid = list(range(50, 1001, 50)) if split == "easy" else list(range(100, 3001, 100))
        assert grid == expected_grid
        for variant, *_ in VARIANTS:
            runs = [r for r in data["runs"] if r["split"] == split and r["variant"] == variant]
            assert tuple(sorted(r["seed"] for r in runs)) == SEEDS
            assert all(len(r["completion"]) == expected_len for r in runs)
            assert all(0 <= value <= 1 for r in runs for value in r["completion"])
            assert all(len(r["eval_files_sha256"]) == 64 for r in runs)
    with SEED_SUMMARY.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 100
    summary = {
        (r["split"], r["variant"], int(r["train_seed"])): r
        for r in summary_rows
    }
    assert len(summary) == 100
    for run in data["runs"]:
        split = run["split"]
        row = summary[(split, run["variant"], run["seed"])]
        assert row["run_id"] == run["run_id"]
        assert row["selection_start_set_id"] == data["selection_start_set_ids"][split]
        x = np.asarray([0] + data["episode_grids"][split], dtype=float)
        y = np.asarray([0] + run["completion"], dtype=float)
        auc = np.trapz(y, x) / x[-1]
        assert abs(auc - float(row["completion_auc"])) < 1e-8
    return data


def seed_matrix(data: dict, split: str, variant: str) -> np.ndarray:
    rows = {r["seed"]: r["completion"] for r in data["runs"] if r["split"] == split and r["variant"] == variant}
    return np.array([rows[seed] for seed in SEEDS], dtype=float)


def draw() -> None:
    configure_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = read_curves()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.20, top=0.80, wspace=0.14)
    legend_ax = fig.add_axes((0.08, 0.84, 0.90, 0.12))
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.9, label=label)
        for _, label, color, linestyle in VARIANTS
    ]
    legend = legend_ax.legend(
        handles=handles,
        ncol=5,
        loc="center",
        fontsize=7.1,
        handlelength=2.4,
        columnspacing=1.3,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    for ax, split in zip(axes, ("easy", "hard")):
        style_axis(ax)
        x = np.asarray(data["episode_grids"][split], dtype=float)
        # Reuse paired seed-resample indices across all methods in this split.
        sample_indices = rng.integers(0, len(SEEDS), size=(BOOTSTRAP_DRAWS, len(SEEDS)))
        for variant, label, color, linestyle in VARIANTS:
            values = seed_matrix(data, split, variant)
            mean = values.mean(axis=0)
            draw_means = values[sample_indices].mean(axis=1)
            low, high = np.quantile(draw_means, (0.025, 0.975), axis=0)
            ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.85)
            records.append(
                {
                    "split": split,
                    "variant": variant,
                    "seeds": len(SEEDS),
                    "first_evaluated_episode": int(x[0]),
                    "last_evaluated_episode": int(x[-1]),
                    "first_mean_completion": float(mean[0]),
                    "last_mean_completion": float(mean[-1]),
                    "peak_mean_completion": float(mean.max()),
                }
            )
        ax.set_xlim(0, x[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.25, 0.50, 0.75, 1.0))
        ax.set_title(f"{split.capitalize()} · {int(x[-1]):,} training episodes")
        ax.set_xlabel("Training episode")
    axes[0].set_ylabel("Selection completion")
    assert_artists_inside(fig, legend_ax, [legend])
    png, pdf = save_figure(fig, OUTPUT)
    # The shared helper renders the PDF; fix its creation metadata so repeated
    # runs also produce a stable PDF byte stream for artifact hashing.
    fixed_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"CreationDate": fixed_date, "ModDate": fixed_date},
    )
    print(json.dumps({"png": str(png), "pdf": str(pdf), "summary": records}, indent=2))


if __name__ == "__main__":
    draw()
