#!/usr/bin/env python3
"""Revised two-panel Completion figure using the submission's server skill.

Usage: python plot_completion_curves.py
The frozen snapshot includes all 60 official training runs, not test episodes.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'figure_sources'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import assert_artists_inside, save_figure
from remote_rl_learning_curves import (
    DEFAULT_COLORS, DEFAULT_LINESTYLES, DEFAULT_MARKERS,
    configure_style, tensorboard_smooth, style_axis,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SNAPSHOT = ROOT / 'figure_sources/completion_training_snapshot_20260927.json'
METHODS = ['sacc', 'qmix', 'ippo', 'vdn', 'mappo', 'happo']
SEEDS = [42, 43, 44, 45, 46]
WEIGHT = 0.96
OUT = ROOT / 'completion_easy_hard_5seed'


def load_data():
    snapshot = json.loads(SNAPSHOT.read_text())
    assert snapshot['seeds'] == SEEDS
    renderer = ROOT / 'figure_sources/remote_rl_learning_curves.py'
    assert hashlib.sha256(renderer.read_bytes()).hexdigest() == snapshot['style_sha256']
    groups = {}
    for wire in snapshot['groups']:
        assert wire['encoding'] == 'zlib+base64'
        raw = zlib.decompress(base64.b64decode(wire['data']))
        assert hashlib.sha256(raw).hexdigest() == wire['sha256']
        group = json.loads(raw)
        assert group['method'] == wire['method'] and group['difficulty'] == wire['difficulty']
        assert [r['seed'] for r in group['runs']] == SEEDS
        budget = 3000 if group['difficulty'] == 'easy' else 1200
        assert group['episodes'] == budget
        assert all(r['observations'] == budget and r['first_episode'] == 1 and r['last_episode'] == budget for r in group['runs'])
        y = np.array([r['completion_stage_counts'] for r in group['runs']], dtype=float) / group['completion_denominator']
        assert y.shape == (5, budget) and np.isfinite(y).all()
        assert ((0 <= y) & (y <= 1)).all()
        group['raw'] = y
        group['smoothed'] = np.stack([tensorboard_smooth(row, WEIGHT) for row in y])
        group['mean'] = group['smoothed'].mean(axis=0)
        group['sd'] = group['smoothed'].std(axis=0, ddof=1)
        # EMA is linear; aggregation must not change the displayed mean.
        assert np.allclose(group['mean'], tensorboard_smooth(y.mean(axis=0), WEIGHT))
        groups[(group['difficulty'], group['method'])] = group
    assert set(groups) == {(d, m) for d in ['easy', 'hard'] for m in METHODS}
    assert sum(g['raw'].size for g in groups.values()) == 126000
    return snapshot, groups


def export_data(snapshot, groups):
    with (ROOT / 'figure_sources/completion_training_observations.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['difficulty', 'method', 'train_seed', 'episode', 'completion', 'ema_completion'])
        for (difficulty, method), group in groups.items():
            for i, seed in enumerate(SEEDS):
                for j, value in enumerate(group['raw'][i]):
                    writer.writerow([difficulty, method, seed, j + 1, f'{value:.8f}', f'{group["smoothed"][i, j]:.10f}'])
    with (ROOT / 'figure_sources/completion_training_mean_sd.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['difficulty', 'method', 'episode', 'n_train_seeds', 'mean_ema_completion', 'sample_sd_ema_completion'])
        for (difficulty, method), group in groups.items():
            for j, mean in enumerate(group['mean']):
                writer.writerow([difficulty, method, j + 1, 5, f'{mean:.10f}', f'{group["sd"][j]:.10f}'])
    audit = {
        'snapshot_sha256': hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        'official_training_runs': 60, 'raw_episode_observations': 126000,
        'seed_unit': 'independent training run', 'seeds': SEEDS,
        'display_smoothing': {'kind': 'EMA', 'previous_weight': WEIGHT, 'initial_value': 'first observed episode per seed'},
        'center': 'mean across smoothed training seeds',
        'band': 'sample SD (ddof=1) across smoothed training seeds; clipped to feasible [0,1] range',
        'raw_lines': 'faint, unsmoothed mean across the same five training seeds',
        'not_a_confidence_interval': True, 'no_interpolation': True,
        'omitted_method': snapshot['omitted_method'],
        'style_source': snapshot['style_source'], 'style_sha256': snapshot['style_sha256'],
        'last_100_training_episode_mean_by_seed': {
            f'{d}/{m}': g['raw'][:, -100:].mean(axis=1).tolist() for (d, m), g in groups.items()
        },
    }
    (ROOT / 'figure_sources/completion_plot_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main():
    assert np.allclose(tensorboard_smooth(np.array([0., 1., 1.]), WEIGHT), [0., .04, .0784])
    snapshot, groups = load_data()
    audit = export_data(snapshot, groups)
    configure_style()  # Unmodified server skill: same palette, typeface, panel and grid style.
    plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.8,
                         'axes.titlesize': 10, 'legend.fontsize': 8.5})
    fig = plt.figure(figsize=(7.1, 3.15))
    left = fig.add_axes([.082, .265, .404, .625])
    right = fig.add_axes([.564, .265, .404, .625])
    legend_ax = fig.add_axes([.08, .015, .90, .115])
    legend_ax.axis('off')
    for ax, difficulty, title in [(left, 'easy', '(a) DRAMA-Easy'), (right, 'hard', '(b) DRAMA-Hard')]:
        for method in METHODS[::-1]:
            group = groups[(difficulty, method)]
            key = 'rmix' if method == 'sacc' else method
            color, line = DEFAULT_COLORS[key], DEFAULT_LINESTYLES[key]
            x = np.arange(1, group['episodes'] + 1)
            mean, sd = group['mean'], group['sd']
            ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd),
                            color=color, alpha=.095, linewidth=0, zorder=1)
            ax.plot(x, group['raw'].mean(axis=0), color=color, alpha=.085, linewidth=.45, zorder=2)
            ax.plot(x, mean, color=color, linestyle=line,
                    linewidth=2.1 if method == 'sacc' else 1.4 if method == 'qmix' else 1.15,
                    alpha=1 if method == 'sacc' else .95, zorder=6 if method == 'sacc' else 4,
                    marker=DEFAULT_MARKERS[key], markevery=500 if difficulty == 'easy' else 200,
                    markersize=2.7, markeredgewidth=.35)
        ax.set_title(title, pad=7)
        ax.set_xlabel('Episode', labelpad=3)
        ax.set_xlim(0, 3000 if difficulty == 'easy' else 1200)
        ax.set_xticks(np.arange(0, 3001, 500) if difficulty == 'easy' else np.arange(0, 1201, 200))
        ax.set_ylim(-.02, 1.05)
        ax.set_yticks(np.arange(0, 1.01, .2))
        style_axis(ax)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.8)
    left.set_ylabel('Training completion', labelpad=4)
    handles = []
    for method in METHODS:
        key = 'rmix' if method == 'sacc' else method
        handles.append(Line2D([], [], label=method.upper(), color=DEFAULT_COLORS[key],
                              linestyle=DEFAULT_LINESTYLES[key], marker=DEFAULT_MARKERS[key],
                              markersize=3.5, linewidth=2.1 if method == 'sacc' else 1.4))
    legend = legend_ax.legend(handles=handles, loc='center', ncol=6,
                             handlelength=2.2, handletextpad=.45, columnspacing=1.55)
    assert_artists_inside(fig, legend_ax, [legend])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in [left, right]:
        for artist in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (artist.get_text(), box.bounds, fig.bbox.bounds)
    png, pdf = save_figure(fig, OUT, png_dpi=360)
    print(json.dumps({'png': str(png), 'pdf': str(pdf), 'training_runs': audit['official_training_runs'],
                      'observations': audit['raw_episode_observations'], 'layout_checks': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
