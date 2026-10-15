"""Freeze E13 (chained to the E12 freeze) and submit sanity/main through gpurun only."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def freeze():
    target = E / 'protocol-freeze.json'; assert not target.exists()
    d = json.loads((E / 'review/disposition.json').read_text()); assert d['release'] and not d['unresolved_blocking']
    for p, h in d['post_fix_files_sha256'].items():
        assert sha(p) == h, 'file changed after the release review: ' + p
    cpu = json.loads((E / 'cache/cpu-check.json').read_text()); assert cpu['passed']
    for p, h in cpu['files_sha256'].items():
        assert sha(p) == h, 'CPU check stale: ' + p
    dry = json.loads((E / 'cpu-dryrun/dryrun-report.json').read_text()); assert dry['passed'] and not dry['cuda_initialized']
    for p, h in dry['code_sha256'].items():
        assert sha(p) == h, 'CPU dry run is stale for ' + p
    parent = E.parent / 'e12/protocol-freeze.json'
    files = dict(json.loads(parent.read_text())['files_sha256']); files[str(parent)] = sha(parent)
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json')) + list((E / 'analysis-e0').glob('*.py'))
    paths += [E / f for f in ['EXPERIMENT_PLAN.md', 'preservation-before.json', 'cache/cpu-check.json', 'cache/e0-local-error.json',
                              'cache/e0-schedule-oracle.json', 'cache/e0-linear-schedule.txt', 'cache/e0-schedule-effect.txt',
                              'review/disposition.json', 'review/code-final.json', 'tools/cpu_dryrun.py', 'cpu-dryrun/dryrun-report.json',
                              'cpu-dryrun/cpu-sanity-acceptance.json']]
    paths += [Path('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008/serial/diagnostics-plain.pt')]
    c = json.loads((E / 'config/main.json').read_text())
    for k in ['e0_runner', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_predictions', 'calibration_export', 'calibration_export_manifest']:
        paths.append(Path(c[k]))
    parent_cfg = json.loads(Path(c['parent_config']).read_text())
    paths += [Path(parent_cfg['dataset_manifest']), Path(parent_cfg['source']['checkpoint']), Path(parent_cfg['freeze_manifest'])]
    paths += [Path(parent_cfg['source_snapshot']['serial']) / 'Models' / f for f in ['layer.py', 'ResNet.py', '__init__.py']]
    for p in paths:
        files[str(p)] = sha(p)
    for p, h in files.items():
        assert sha(p) == h, p
    target.write_text(json.dumps({'schema': 'ftbc.e13.serial-deferral-screen.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'protected_evaluation': False, 'files_sha256': files}, indent=2) + '\n')
    w = Path(c['workspace']); staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w)}
    staged[str(target.relative_to(w))] = sha(target)
    (E / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': staged}, indent=2) + '\n')
    print('Frozen', len(files), 'entries;', len(staged), 'workspace files in the Slurm source manifest')


def submit(which):
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    if which == 'main':
        assert json.loads((E / 'sanity-acceptance.json').read_text())['passed']
    receipt = E / f'slurm/{which}-submit.json'; assert not receipt.exists(), 'Reconcile the previous receipt; do not duplicate'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    c = json.loads((E / 'config' / f'{which}.json').read_text())
    cmd = ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-e13-{which}-v1', '--gpu', 'a100', '--gpus', '1',
           '--cpus', '24', '--mem', '128G', '--time', {'sanity': '00:30:00', 'main': '03:00:00'}[which]]
    for u in c['allowed_uuids']:
        cmd += ['--allowed-uuid', u]
    cmd += ['--chdir', c['workspace'], '--output', str(E / f'slurm/{which}-%j.out'), '--source-manifest', str(E / 'slurm-source-manifest.json'), '--',
            '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e13_runner.py'), '--config', str(E / 'config' / f'{which}.json')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'sanity', 'main']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit(a.action)
