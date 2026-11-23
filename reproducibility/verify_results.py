#!/usr/bin/env python3
"""Verify the included artifact and statistics without training or checkpoints."""
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zlib

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8), (a, b)


def verify_integrity():
    manifest = read('MANIFEST_SHA256.json')
    observed = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                if p.is_file() and p.relative_to(ROOT).as_posix() != 'MANIFEST_SHA256.json'
                and p.relative_to(ROOT).parts[0] != '.git'
                and '__pycache__' not in p.relative_to(ROOT).parts
                and '.pytest_cache' not in p.relative_to(ROOT).parts}
    missing = sorted(set(manifest) - observed)
    added = sorted(observed - set(manifest))
    if missing or added:
        details = ['Artifact file set differs from MANIFEST_SHA256.json.']
        for label, paths in [('Missing', missing), ('Added', added)]:
            if paths:
                details.append(f'{label} ({len(paths)}): ' + ', '.join(paths[:5]))
        details.append('Use a clean extraction. Keep virtual environments and generated outputs '
                       'outside the verification copy. No package installation is needed for '
                       'python verify_results.py; do not regenerate the supplied manifest.')
        raise ValueError('\n'.join(details))
    for name, digest in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Checksum mismatch: {name}. Use an unmodified extraction.')
    return manifest


def main():
    if not __debug__:
        raise ValueError('Run without -O or PYTHONOPTIMIZE: numerical checks require assertions.')
    manifest = verify_integrity()
    provenance = {p['path']: p for p in read('SOURCE_PROVENANCE.json')['files']}
    index = read('evaluation_index.json')
    assert len(index) == 94
    identities = set()
    groups = {}
    shared_starts = {}
    metric_map = {'mean_completion_ratio': 'completion_ratio', 'success_rate': 'success',
                  'mean_steps': 'steps', 'mean_environment_return': 'environment_return'}
    for item in index:
        ident = (item['workspace'], item['difficulty'], item['method'], item['mode'], item['train_seed'])
        assert ident not in identities, ident
        identities.add(ident)
        raw = read(item['result'])
        record = read(item['record'])
        if 'result_sha256' in record:
            assert record['result_sha256'] == provenance[item['result']]['source_sha256']
        assert len(item['checkpoint_sha256']) == 64
        starts = read(item['start_set'])
        eps = raw['episodes']
        assert len(eps) == starts['count'] == 100
        assert [e['start_positions'] for e in eps] == starts['starts'], ident
        key = (item['workspace'], item['difficulty'])
        if key in shared_starts:
            assert starts['starts'] == shared_starts[key]
        shared_starts[key] = starts['starts']
        count = {'easy': 8, 'hard': 40, 'large': 96, 'cross5': 84}[item['difficulty']]
        for e in eps:
            assert 0 <= e['completion_ratio'] <= 1
            assert e['success'] in {0, 1, False, True}
            assert 0 < e['steps'] <= item['max_steps']
            assert math.isfinite(e['environment_return'])
            assert e['environment_return'] <= e['completion_ratio'] * count + 1e-7
            assert not e['success'] or e['completion_ratio'] == 1
        metrics = {name: statistics.mean(float(e[field]) for e in eps) for name, field in metric_map.items()}
        if 'summary' in raw:
            for name, value in metrics.items():
                close(value, raw['summary'][name])
        else:
            close(metrics['mean_completion_ratio'], raw['mean_completion_ratio'])
            close(metrics['success_rate'], raw['mean_success_rate'])
            close(metrics['mean_environment_return'], raw['mean_environment_return'])
        groups.setdefault(ident[:4], {})[item['train_seed']] = metrics

    def compare(key, stored, sd_key):
        rows = groups[key]
        assert sorted(rows) == ([42, 43, 44, 45, 46] if key[0] == 'main_eval' else [42, 123, 456])
        for name in metric_map:
            points = [row[name] for seed, row in sorted(rows.items())]
            close(statistics.mean(points), stored[name]['mean'])
            close(statistics.stdev(points), stored[name][sd_key])

    for difficulty in ['easy', 'hard']:
        summary = read(f'workspaces/main_eval/experiments/mahrm_direct_20260924/summaries/table6_mahrm_t035_{difficulty}_holdout_v1.json')
        for method in summary['methods']:
            mode = 'greedy' if method in {'qmix', 'vdn', 'mahrm'} else 'tempered'
            compare(('main_eval', difficulty, method, mode), summary['method_statistics'][method], 'sample_sd')
    for alias, name, difficulty in [('large', 'reviewer3_large_random_tempered_sensitivity_v1', 'large'),
        ('cross', 'workflow_cross5_randomtrain_v2_final_eval_integrated', 'cross5')]:
        summary = read(f'workspaces/{alias}/experiments/summaries/{name}.json')
        for row in summary['aggregates']:
            method = {'sacc': 'full', 'happo': 'all_off'}[row['method']] if alias == 'large' else row['method']
            compare((alias, difficulty, method, row.get('execution_mode', 'tempered')), row['metrics'], 'sample_std')

    curves = read('figures/tenseed_ablation_selection_curves.json')
    with (ROOT/'figures/tenseed_ablation_seed_summary.csv').open() as f:
        rows = {r['run_id']: r for r in csv.DictReader(f)}
    assert len(rows) == len(curves['runs']) == 100
    for run in curves['runs']:
        grid = [0] + curves['episode_grids'][run['split']]
        vals = [0] + run['completion']
        auc = sum((grid[i]-grid[i-1]) * (vals[i]+vals[i-1])/2 for i in range(1,len(grid))) / grid[-1]
        close(auc, rows[run['run_id']]['completion_auc'])
    snapshot = read('figures/figure_sources/completion_training_snapshot_20260927.json')
    total = 0
    for group in snapshot['groups']:
        payload = zlib.decompress(base64.b64decode(group['data']))
        assert hashlib.sha256(payload).hexdigest() == group['sha256']
        data = json.loads(payload)
        budget = 3000 if data['difficulty'] == 'easy' else 1200
        assert data['episodes'] == budget
        assert [r['seed'] for r in data['runs']] == [42,43,44,45,46]
        for r in data['runs']:
            assert len(r['completion_stage_counts']) == r['observations'] == budget
            assert r['first_episode'] == 1 and r['last_episode'] == budget
            total += budget
    assert total == 126000
    print(json.dumps({'status': 'passed', 'payload_files': len(manifest), 'evaluations': 94,
        'raw_episodes': 9400, 'seed_metric_groups': len(groups), 'ablation_seed_curves': 100,
        'full_training_observations': total, 'checkpoints_required': False}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, ValueError, KeyError, OSError) as error:
        print(f'Verification failed: {error or "a numerical consistency check failed"}', file=sys.stderr)
        sys.exit(1)
