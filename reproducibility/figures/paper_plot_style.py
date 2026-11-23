#!/usr/bin/env python3
"""Shared Matplotlib style and layout checks for ML-paper figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/paper-figure-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2F5D8C"
ORANGE = "#D2673A"
RED = "#C4514B"
GOLD = "#C58A2E"
POINT = "#7C8796"
INK = "#243040"
GRID = "#CFD6E0"
PANEL = "#FBFCFE"

SINGLE_COLUMN = (3.45, 2.35)
DOUBLE_COLUMN_TWO_PANEL = (7.0, 2.50)
DOUBLE_COLUMN_THREE_PANEL = (7.0, 2.28)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.2,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#AEB8C5",
            "axes.labelcolor": INK,
            "axes.labelsize": 8.2,
            "axes.titlecolor": INK,
            "axes.titlesize": 8.8,
            "axes.titleweight": "semibold",
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(
        True,
        color=GRID,
        linewidth=0.55,
        linestyle=(0, (3, 2)),
        alpha=0.75,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AEB8C5")
    ax.spines["bottom"].set_color("#AEB8C5")
    ax.tick_params(length=3, width=0.7)


def assert_artists_inside(
    fig: plt.Figure,
    ax: plt.Axes,
    artists: Iterable,
    *,
    tolerance: float = 1.0,
) -> None:
    """Raise when a legend or annotation escapes its intended axes."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    for artist in artists:
        artist_box = artist.get_window_extent(renderer)
        if (
            artist_box.x0 < axes_box.x0 - tolerance
            or artist_box.x1 > axes_box.x1 + tolerance
            or artist_box.y0 < axes_box.y0 - tolerance
            or artist_box.y1 > axes_box.y1 + tolerance
        ):
            raise RuntimeError(
                "Artist exceeds axes bounds: "
                f"artist={artist_box.bounds}, axes={axes_box.bounds}"
            )


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    *,
    png_dpi: int = 360,
) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
