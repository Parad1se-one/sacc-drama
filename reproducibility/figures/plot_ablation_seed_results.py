#!/usr/bin/env python3
"""Ten-seed ablation AUC points, using the existing frozen selection records.

The data and protocol are unchanged. This script verifies all 100 seed AUCs
against the original evaluation curves and every active table mean/SD.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from plot_tenseed_ablation_curves import (
    INPUT, SEED_SUMMARY, SEEDS, VARIANTS, read_curves,
)
from paper_plot_style import assert_artists_inside, configure_style, save_figure, style_axis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ablation_seed_results'
LABELS = ['Full', 'Actor\nprogress\noff', 'Critic\nprogress\noff', 'Structured\nreward\noff', 'All off']
TABLE_LABELS = ['Full', 'No actor progress', 'No critic progress', 'No structured reward', 'All off']
METRICS = ['completion_auc', 'selected_completion', 'selected_success', 'selected_steps']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(values, digits):
    # Decimal means preserve displayed half-up rounding of the existing prose.
    mean = sum(Decimal(str(v)) for v in values) / Decimal(len(values))
    sd = np.std(np.asarray(values, dtype=float), ddof=1)
    precision = Decimal('1').scaleb(-digits)
    m = mean.quantize(precision, rounding=ROUND_HALF_UP)
    s = Decimal(str(sd)).quantize(precision, rounding=ROUND_HALF_UP)
    return f'${m}\\;({s})$'


def main():
    data = read_curves()  # Independently integrates all 100 complete curves.
    rows = list(csv.DictReader(SEED_SUMMARY.open()))
    groups = {}
    for split in ['easy', 'hard']:
        for variant, *_ in VARIANTS:
            selected = sorted([r for r in rows if r['split'] == split and r['variant'] == variant],
                              key=lambda r: int(r['train_seed']))
            assert tuple(int(r['train_seed']) for r in selected) == SEEDS
            for r in selected:
                assert r['selection_start_set_id'] == data['selection_start_set_ids'][split]
                assert all(0 <= float(r[k]) <= 1 for k in METRICS[:-1])
                assert 0 < float(r['selected_steps']) <= (300 if split == 'easy' else 500)
                curve = next(c for c in data['runs'] if c['run_id'] == r['run_id'])
                index = data['episode_grids'][split].index(int(r['selected_checkpoint_episode']))
                assert abs(float(r['selected_completion']) - curve['completion'][index]) < 1e-8
            groups[split, variant] = selected

    active_tex = (ROOT / 'ablation_table_reference.tex').read_text().split('\\end{document}')[0]
    table_checks = []
    stats = {}
    for split in ['easy', 'hard']:
        for (variant, *_), label in zip(VARIANTS, TABLE_LABELS):
            group = groups[split, variant]
            cells = [fmt([r[k] for r in group], 1 if k == 'selected_steps' else 3) for k in METRICS]
            expected = label + ' & ' + ' & '.join(cells)
            assert expected in active_tex, expected
            table_checks.append(expected)
            stats[f'{split}/{variant}'] = {
                k: {'mean': float(np.mean([float(r[k]) for r in group])),
                    'sample_sd': float(np.std([float(r[k]) for r in group], ddof=1)),
                    'seed_values': [float(r[k]) for r in group]} for k in METRICS
            }
    signs = np.asarray(list(itertools.product([-1, 1], repeat=10)))
    tests = {}
    for split in ['easy', 'hard']:
        full = np.array([float(r['completion_auc']) for r in groups[split, 'full']])
        raw_p = []
        for variant, *_ in VARIANTS[1:]:
            vals = np.array([float(r['completion_auc']) for r in groups[split, variant]])
            diff = full - vals
            p = float(np.mean(np.abs((signs * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-12))
            raw_p.append(p)
            tests[f'{split}/{variant}'] = {'paired_auc_difference': float(diff.mean()), 'exact_p': p}
        order = np.argsort(raw_p)
        adjusted = np.minimum(1, np.maximum.accumulate(np.array(raw_p)[order] * np.arange(4, 0, -1)))
        for j, i in enumerate(order):
            tests[f'{split}/{VARIANTS[i + 1][0]}']['holm_p'] = float(adjusted[j])

    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    fig.subplots_adjust(left=.082, right=.985, bottom=.28, top=.88, wspace=.14)
    offsets = np.linspace(-.12, .12, 10)
    for ax, split, letter in zip(axes, ['easy', 'hard'], ['a', 'b']):
        for j, (variant, label, color, linestyle) in enumerate(VARIANTS):
            vals = np.array([float(r['completion_auc']) for r in groups[split, variant]])
            # Every seed is a visible dot; fixed seed order determines jitter.
            ax.scatter(j + offsets, vals, color=color, edgecolor='white', linewidth=.4,
                       s=17, alpha=.75, zorder=3)
            ax.errorbar(j + .22, vals.mean(), yerr=vals.std(ddof=1), fmt='D',
                        color=color, markeredgecolor='#243040', markeredgewidth=.6,
                        markersize=4.2, capsize=3.0, elinewidth=1.4, zorder=5)
        ax.set_title(f'({letter}) {split.capitalize()}: ' + ('0-1000' if split == 'easy' else '0-3000') + ' episodes')
        ax.set_xlim(-.42, 4.5)
        ax.set_xticks(range(5), LABELS)
        ax.set_ylim(0, .7)
        ax.set_yticks([0, .2, .4, .6])
        ax.tick_params(axis='x', labelsize=7.0)
        style_axis(ax)
        ax.xaxis.grid(False)
    axes[0].set_ylabel('Selection Completion AUC')
    legend_ax = fig.add_axes([.12, .01, .8, .085])
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=[
        Line2D([], [], marker='o', color='#7C8796', linestyle='none', markersize=4, label='Individual training seed (n = 10)'),
        Line2D([], [], marker='D', color='#243040', markersize=4, linewidth=1.4, label=r'Mean $\pm$ sample SD'),
    ], ncol=2, loc='center', fontsize=7.3)
    assert_artists_inside(fig, legend_ax, [legend])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        assert all(a.x1 + 2 < b.x0 for a, b in zip(tick_boxes, tick_boxes[1:]))
        for artist in [ax.title, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1
            assert box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1
    png, pdf = save_figure(fig, OUT, png_dpi=360)
    audit = {
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [INPUT, SEED_SUMMARY]},
        'independent_training_runs': 100, 'variants': [v[0] for v in VARIANTS], 'seeds': SEEDS,
        'periodic_evaluations_checked': sum(len(r['completion']) for r in data['runs']),
        'auc_rule': 'Trapezoidal integral / horizon; frozen (0,0) boundary; all observations retained',
        'selection_set_ids': data['selection_start_set_ids'],
        'table_rows_verified': table_checks, 'statistics': stats, 'paired_tests': tests,
        'interpretation': 'Fixed-training-start selection-set ablation, not held-out test outcomes',
        'seed_filtering': False, 'new_smoothing': False, 'new_experiments': False,
        'visual_encoding': 'All seeds as circles; mean diamonds and sample-SD error bars (not confidence intervals)',
        'output_hashes': {p.name: sha(p) for p in [png, pdf]},
        'layout_checks': 'passed',
    }
    audit_path = ROOT / 'figure_sources/ablation_seed_results_audit_20260927.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'png': str(png), 'pdf': str(pdf), 'table_rows_checked': len(table_checks),
                      'runs_checked': 100, 'paired_tests': tests}, indent=2))


if __name__ == '__main__':
    main()
