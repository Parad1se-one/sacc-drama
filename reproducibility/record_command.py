#!/usr/bin/env python3
"""Print a relocated direct command from an archived run record. Never executes it."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('record', type=Path)
args = p.parse_args()
path = (ROOT / args.record).resolve()
relative = path.relative_to(ROOT)
assert relative.parts[0] == 'workspaces'
workspace = ROOT / 'workspaces' / relative.parts[1]
x = json.loads(path.read_text())
command = x.get('command')
if not isinstance(command, list):
    raise SystemExit('This record has no direct command. Use the MAHRM CLI and its frozen hyperparameters/protocol files.')
command = [str(v).replace('{ARTIFACT_ROOT}', str(ROOT)) for v in command]
command[0] = sys.executable
assert not any('{ARCHIVED_ROOT}' in v for v in command), 'An unbundled historical dependency is referenced.'
cwd = x.get('cwd', str(workspace)).replace('{ARTIFACT_ROOT}', str(ROOT))
imports = ':'.join(str(workspace / 'src/sacc-drama' / d) for d in ['DRAMA', 'SACC'])
print('# Review in a separate working copy; training/evaluation will create outputs.')
print('cd ' + shlex.quote(cwd))
print('PYTHONPATH=' + shlex.quote(imports) + ' ' + shlex.join(command))
