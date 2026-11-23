#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_COLORS = {
    "rmix": "#D84A4A",
    "qmix": "#2E5FA8",
    "ippo": "#E0903A",
    "vdn": "#4EA86A",
    "qtran": "#8A63C7",
    "mappo": "#9B7C63",
    "happo": "#8E98A6",
}

DEFAULT_MARKERS = {
    "rmix": "o",
    "qmix": "s",
    "ippo": "^",
    "vdn": "D",
    "qtran": "P",
    "mappo": "X",
    "happo": "v",
}

DEFAULT_LINESTYLES = {
    "rmix": "solid",
    "qmix": "solid",
    "ippo": (0, (5, 2)),
    "vdn": "solid",
    "qtran": "solid",
    "mappo": (0, (5, 2)),
    "happo": (0, (5, 2)),
}


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFDFF",
            "axes.edgecolor": "#D7DCE5",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#C7CEDA",
            "grid.linestyle": "--",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "xtick.color": "#30343A",
            "ytick.color": "#30343A",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def tensorboard_smooth(values: np.ndarray, weight: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    last = float(values[0])
    smoothed[0] = last
    for index in range(1, len(values)):
        last = last * weight + (1.0 - weight) * float(values[index])
        smoothed[index] = last
    return smoothed


def default_color(method: dict) -> str:
    return method.get("color") or DEFAULT_COLORS.get(method.get("key", ""), "#5B6574")


def default_linestyle(method: dict):
    return method.get("linestyle") or DEFAULT_LINESTYLES.get(method.get("key", ""), "solid")


def default_marker(method: dict) -> str:
    return method.get("marker") or DEFAULT_MARKERS.get(method.get("key", ""), "o")


def method_tier(method: dict) -> str:
    if method.get("primary"):
        return "primary"
    if method.get("secondary"):
        return "secondary"
    return "baseline"


def line_alpha(tier: str) -> float:
    return {"primary": 1.0, "secondary": 0.98}.get(tier, 0.82)


def line_width(tier: str) -> float:
    return {"primary": 2.9, "secondary": 2.1}.get(tier, 1.55)


def raw_alpha(tier: str) -> float:
    return 0.11 if tier in {"primary", "secondary"} else 0.0


def zorder(tier: str) -> int:
    return {"primary": 6, "secondary": 5}.get(tier, 3)


def normalize_eval_points(eval_points: dict | None) -> dict[str, np.ndarray] | None:
    if not eval_points:
        return None
    return {
        key: np.asarray(value, dtype=float)
        for key, value in eval_points.items()
        if value is not None
    }


def normalize_method(method: dict, success_threshold: float) -> dict:
    completion_raw = np.asarray(method["completion"], dtype=float)
    steps_raw = np.asarray(method["steps"], dtype=float)
    episodes = np.asarray(method["episodes"], dtype=float)
    success_raw = np.asarray(
        method.get("success", (completion_raw >= success_threshold).astype(float)),
        dtype=float,
    )
    completion_smooth = method.get("completion_smooth")
    success_smooth = method.get("success_smooth")
    steps_smooth = method.get("steps_smooth")

    return {
        "key": method.get("key", method.get("label", "method").lower()),
        "label": method.get("label", method.get("key", "Method").upper()),
        "color": default_color(method),
        "linestyle": default_linestyle(method),
        "marker": default_marker(method),
        "tier": method_tier(method),
        "episodes": episodes,
        "completion_raw": completion_raw,
        "completion_smooth": np.asarray(completion_smooth, dtype=float)
        if completion_smooth is not None
        else tensorboard_smooth(completion_raw, weight=0.96),
        "success_raw": success_raw,
        "success_smooth": np.asarray(success_smooth, dtype=float)
        if success_smooth is not None
        else tensorboard_smooth(success_raw, weight=0.96),
        "steps_raw": steps_raw,
        "steps_smooth": np.asarray(steps_smooth, dtype=float)
        if steps_smooth is not None
        else tensorboard_smooth(steps_raw, weight=0.97),
        "eval_points": normalize_eval_points(method.get("eval_points")),
    }


def style_axis(ax: plt.Axes, *, show_left_spine: bool = True) -> None:
    ax.spines["left"].set_visible(show_left_spine)
    ax.spines["bottom"].set_color("#D7DCE5")
    ax.tick_params(axis="both", labelsize=8.5)


def add_annotation(ax: plt.Axes, annotation: dict, methods_by_key: dict[str, dict]) -> None:
    color = annotation.get("color")
    method_key = annotation.get("method")
    if color is None and method_key in methods_by_key:
        color = methods_by_key[method_key]["color"]
    color = color or "#6E7684"

    arrowprops = {
        "arrowstyle": annotation.get("arrowstyle", "->"),
        "lw": annotation.get("linewidth", 0.95),
        "color": annotation.get("arrow_color", color),
    }
    if annotation.get("connectionstyle"):
        arrowprops["connectionstyle"] = annotation["connectionstyle"]

    bbox = None
    if annotation.get("bbox"):
        bbox = {
            "boxstyle": "round,pad=0.26",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
        }
        if isinstance(annotation["bbox"], dict):
            bbox.update(annotation["bbox"])

    ax.annotate(
        annotation["text"],
        xy=tuple(annotation["xy"]),
        xytext=tuple(annotation["xytext"]),
        arrowprops=arrowprops,
        bbox=bbox,
        color=color,
        fontsize=annotation.get("fontsize", 8.2),
        ha=annotation.get("ha", "left"),
        va=annotation.get("va", "center"),
        zorder=annotation.get("zorder", 7),
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=320)
    plt.close(fig)


def build_learning_curves(spec: dict) -> plt.Figure:
    configure_style()
    success_threshold = float(spec.get("success_threshold", 0.999))
    methods = [normalize_method(method, success_threshold) for method in spec["methods"]]
    methods_by_key = {method["key"]: method for method in methods}

    fig, axes = plt.subplots(1, 3, figsize=tuple(spec.get("figsize", [14.1, 4.0])), sharex=True)
    completion_ax, success_ax, steps_ax = axes

    for method in methods:
        tier = method["tier"]
        color = method["color"]
        linestyle = method["linestyle"]
        episodes = method["episodes"]

        if raw_alpha(tier) > 0.0:
            completion_ax.plot(
                episodes,
                method["completion_raw"],
                color=color,
                linewidth=0.8,
                alpha=raw_alpha(tier),
                zorder=1,
            )
            steps_ax.plot(
                episodes,
                method["steps_raw"],
                color=color,
                linewidth=0.8,
                alpha=raw_alpha(tier),
                zorder=1,
            )

        completion_ax.plot(
            episodes,
            method["completion_smooth"],
            color=color,
            linestyle=linestyle,
            linewidth=line_width(tier),
            alpha=line_alpha(tier),
            zorder=zorder(tier),
        )
        success_ax.plot(
            episodes,
            method["success_smooth"],
            color=color,
            linestyle=linestyle,
            linewidth=line_width(tier),
            alpha=line_alpha(tier),
            zorder=zorder(tier),
        )
        steps_ax.plot(
            episodes,
            method["steps_smooth"],
            color=color,
            linestyle=linestyle,
            linewidth=line_width(tier),
            alpha=line_alpha(tier),
            zorder=zorder(tier),
        )

        eval_points = method["eval_points"]
        if eval_points:
            marker_size = 24 if tier == "primary" else 17 if tier == "secondary" else 14
            edge_alpha = 0.95 if tier == "primary" else 0.80 if tier == "secondary" else 0.72

            for ax, metric in [
                (completion_ax, "completion"),
                (success_ax, "success"),
                (steps_ax, "steps"),
            ]:
                if metric in eval_points and "episodes" in eval_points:
                    ax.scatter(
                        eval_points["episodes"],
                        eval_points[metric],
                        s=marker_size,
                        facecolors="white",
                        edgecolors=color,
                        linewidth=0.9 if tier == "primary" else 0.8,
                        alpha=edge_alpha,
                        zorder=zorder(tier) + 1,
                    )

        if spec.get("shade_primary_success", True) and tier == "primary":
            success_ax.fill_between(
                method["episodes"],
                0,
                method["success_smooth"],
                color=color,
                alpha=0.06,
                zorder=1,
            )

    xlim = tuple(spec.get("xlim", [1, max(method["episodes"][-1] for method in methods)]))
    x_ticks = spec.get("x_ticks")
    if not x_ticks:
        x_ticks = list(np.linspace(xlim[0], xlim[1], 6, dtype=int))[1:]

    for ax, title, ylabel in [
        (completion_ax, "Completion during training", "Smoothed completion ratio"),
        (success_ax, "Success during training", "Smoothed success rate"),
    ]:
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel(spec.get("x_label", "Training episodes"), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*spec.get("ratio_ylim", [-0.05, 1.05]))
        ax.set_xticks(x_ticks)
        ax.set_yticks(spec.get("ratio_yticks", [0.0, 0.25, 0.5, 0.75, 1.0]))
        style_axis(ax, show_left_spine=True)

    steps_ax.set_title("Episode steps during training", fontsize=11, pad=10)
    steps_ax.set_xlabel(spec.get("x_label", "Training episodes"), fontsize=10)
    steps_ax.set_ylabel("Smoothed episode steps", fontsize=10)
    steps_ax.set_xlim(*xlim)
    steps_ax.set_ylim(*spec.get("steps_ylim", [0, 310]))
    steps_ax.set_xticks(x_ticks)
    steps_ax.set_yticks(spec.get("steps_yticks", [0, 75, 150, 225, 300]))
    style_axis(steps_ax, show_left_spine=True)

    annotations = spec.get("annotations", {})
    for panel_name, ax in [
        ("completion", completion_ax),
        ("success", success_ax),
        ("steps", steps_ax),
    ]:
        for annotation in annotations.get(panel_name, []):
            add_annotation(ax, annotation, methods_by_key)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=method["color"],
            lw=line_width(method["tier"]),
            linestyle=method["linestyle"],
            marker=method["marker"],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=method["label"],
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=int(spec.get("legend_ncol", min(4, max(1, len(methods))))),
        bbox_to_anchor=tuple(spec.get("legend_anchor", [0.5, 0.02])),
        columnspacing=1.5,
        handletextpad=0.5,
        fontsize=8.7,
    )
    fig.suptitle(spec.get("title", "Learning Dynamics"), fontsize=13, y=float(spec.get("title_y", 0.98)))
    fig.subplots_adjust(
        top=float(spec.get("top", 0.79)),
        bottom=float(spec.get("bottom", 0.23)),
        left=float(spec.get("left", 0.06)),
        right=float(spec.get("right", 0.995)),
        wspace=float(spec.get("wspace", 0.22)),
    )
    return fig


def make_demo_spec() -> dict:
    episodes = np.arange(1, 3001, dtype=float)

    rmix_completion = np.clip(1 / (1 + np.exp(-(episodes - 1500) / 120)), 0, 1)
    rmix_completion[:150] += 0.55 * np.exp(-((episodes[:150] - 80) ** 2) / (2 * 45**2))
    rmix_completion = np.clip(rmix_completion, 0, 1)
    rmix_steps = 300 - 284 * (1 / (1 + np.exp(-(episodes - 1700) / 55)))

    qmix_completion = 0.42 + 0.42 * (episodes / episodes.max()) + 0.08 * np.sin(episodes / 140)
    qmix_completion = np.clip(qmix_completion, 0, 0.88)
    qmix_steps = 300 - 18 * (episodes / episodes.max()) + 8 * np.sin(episodes / 160)

    mappo_completion = np.clip(0.58 + 0.22 * np.sin(episodes / 260) + 0.10 * np.sin(episodes / 70), 0.15, 0.90)
    mappo_steps = np.clip(300 - 110 * np.abs(np.sin(episodes / 180)) - 0.03 * episodes, 170, 300)

    checkpoint_episodes = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=float)

    return {
        "title": "Learning Dynamics on DemoTask",
        "success_threshold": 0.999,
        "xlim": [1, 3000],
        "x_ticks": [500, 1000, 1500, 2000, 2500, 3000],
        "steps_ylim": [0, 310],
        "methods": [
            {
                "key": "rmix",
                "label": "RMix",
                "primary": True,
                "episodes": episodes.tolist(),
                "completion": rmix_completion.tolist(),
                "steps": rmix_steps.tolist(),
                "eval_points": {
                    "episodes": checkpoint_episodes.tolist(),
                    "completion": [0.0, 0.0, 0.95, 1.0, 1.0, 1.0],
                    "success": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    "steps": [300, 300, 185, 18, 16, 16],
                },
            },
            {
                "key": "qmix",
                "label": "QMIX",
                "secondary": True,
                "episodes": episodes.tolist(),
                "completion": qmix_completion.tolist(),
                "steps": qmix_steps.tolist(),
                "eval_points": {
                    "episodes": checkpoint_episodes.tolist(),
                    "completion": [0.50, 0.00, 0.50, 0.25, 0.375, 0.875],
                    "steps": [300, 300, 300, 300, 300, 300],
                },
            },
            {
                "key": "mappo",
                "label": "MAPPO",
                "episodes": episodes.tolist(),
                "completion": mappo_completion.tolist(),
                "steps": mappo_steps.tolist(),
            },
        ],
        "annotations": {
            "completion": [
                {
                    "text": "RMix reaches 1.00",
                    "xy": [3000, 1.0],
                    "xytext": [2260, 1.02],
                    "method": "rmix",
                    "fontsize": 8.5,
                    "va": "bottom",
                },
                {
                    "text": "QMIX peaks at 0.875",
                    "xy": [3000, 0.875],
                    "xytext": [2140, 0.79],
                    "method": "qmix",
                    "fontsize": 8.2,
                },
            ],
            "success": [
                {
                    "text": "Baseline success spikes\nare transient",
                    "xy": [900, 0.18],
                    "xytext": [255, 0.57],
                    "color": "#6E7684",
                    "arrow_color": "#7D8694",
                    "connectionstyle": "arc3,rad=-0.12",
                    "bbox": True,
                    "fontsize": 8.1,
                },
                {
                    "text": "RMix reaches full success",
                    "xy": [2000, 1.0],
                    "xytext": [1480, 0.72],
                    "method": "rmix",
                    "fontsize": 8.8,
                }
            ],
            "steps": [
                {
                    "text": "RMix converges to ~16 steps",
                    "xy": [3000, 16],
                    "xytext": [1960, 70],
                    "method": "rmix",
                    "fontsize": 8.4,
                },
                {
                    "text": "Baselines stay near\nthe 300-step cap",
                    "xy": [2860, 299],
                    "xytext": [520, 165],
                    "color": "#6E7684",
                    "arrow_color": "#7D8694",
                    "connectionstyle": "arc3,rad=0.18",
                    "bbox": True,
                    "fontsize": 8.0,
                }
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a TensorBoard-style three-panel learning-curve figure."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="JSON spec describing methods, traces, eval points, and annotations.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("learning_curves_demo"),
        help="Output stem used for both PNG and PDF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.spec:
        with args.spec.open() as handle:
            spec = json.load(handle)
    else:
        spec = make_demo_spec()

    fig = build_learning_curves(spec)
    save_figure(fig, args.output_stem)
    print(f"Saved {args.output_stem.with_suffix('.png')}")
    print(f"Saved {args.output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
