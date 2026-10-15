"""Freeze E14 (chained to the E13 freeze) and submit sanity/main/extra through gpurun only."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parents[1]
TIME = {'sanity': '00:30:00', 'main': '03:00:00', 'extra': '03:00:00'}
ATTEMPT = 'v2'   # v1 (freeze superseded, see superseded-freeze-v1/) was held by the TaskProlog before any payload ran


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def freeze():
    target = E / 'protocol-freeze.json'; assert not target.exists()
    d = json.loads((E / 'review/disposition.json').read_text()); assert d['release'] and not d['unresolved_blocking']
    final = json.loads((E / 'review/code-final.json').read_text()); verdict = final['response'].strip().splitlines()[-1]
    assert final['status'] == 'completed' and final['model'] == 'claude-opus-5-5' and 'release' in verdict and 'not release' not in verdict, verdict
    for p, h in d['post_fix_files_sha256'].items():
        assert sha(p) == h, 'file changed after the release review: ' + p
    for name, key in [('cache/cpu-check.json', 'files_sha256'), ('cache/step-solver-check.json', 'files_sha256'), ('cpu-dryrun/dryrun-report.json', 'code_sha256')]:
        rep = json.loads((E / name).read_text()); assert rep['passed'] and not rep['cuda_initialized'], name
        for p, h in rep[key].items():
            assert sha(p) == h, f'{name} is stale for {p}'
    parent = E.parent / 'e13/protocol-freeze.json'
    files = dict(json.loads(parent.read_text())['files_sha256']); files[str(parent)] = sha(parent)
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json'))
    paths += [E / f for f in ['EXPERIMENT_PLAN.md', 'preservation-before.json', 'cache/cpu-check.json', 'cache/step-solver-check.json',
                              'input/fresh-development.json', 'input/preparation.json', 'input/exclusion-audit.json', 'review/disposition.json', 'review/code-final.json',
                              'record_costs.py', 'verify_preservation.py',
                              'tools/cpu_dryrun.py', 'cpu-dryrun/dryrun-report.json', 'cpu-dryrun/cpu-sanity-acceptance.json']]
    c = json.loads((E / 'config/main.json').read_text())
    for k in ['e0_runner', 'e13_runner', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_predictions', 'calibration_export', 'calibration_export_manifest', 'fresh_manifest']:
        paths.append(Path(c[k]))
    paths.append(Path(c['e13_runner']).parent / 'serial_solver.py')
    paths += [Path(c['e13_output']) / f for f in ['predictions.pt', 'bias-M0.pt', 'bias-F0-n3.pt', 'teacher-targets.pt', 'hardware.json', 'result.json']]
    parent_cfg = json.loads(Path(c['parent_config']).read_text())
    paths += [Path(parent_cfg['dataset_manifest']), Path(parent_cfg['source']['checkpoint']), Path(parent_cfg['freeze_manifest'])]
    paths += [Path(parent_cfg['source_snapshot']['serial']) / 'Models' / f for f in ['layer.py', 'ResNet.py', '__init__.py']]
    prep = json.loads((E / 'input/preparation.json').read_text()); paths += [Path(p) for p in prep['exclusion_sources']]
    for p in paths:
        files[str(p)] = sha(p)
    for p, h in files.items():
        assert sha(p) == h, p
    target.write_text(json.dumps({'schema': 'ftbc.e14.per-step-ftbc-screen.v2', 'created_utc': datetime.now(timezone.utc).isoformat(), 'attempt': ATTEMPT,
                                  'protected_evaluation': False, 'files_sha256': files}, indent=2) + '\n')
    # gpurun stages only files whose resolved path stays in the workspace; symlinked entries (e.g. ssd-runs -> /data5) are verified by the runner from the freeze
    w = Path(c['workspace']); staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)}
    staged[str(target.relative_to(w))] = sha(target)
    (E / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': staged}, indent=2) + '\n')
    print('Frozen', len(files), 'entries;', len(staged), 'workspace files in the Slurm source manifest')


def submit(which):
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    if which != 'sanity':
        assert json.loads((E / 'sanity-acceptance.json').read_text())['passed']
    receipt = E / f'slurm/{which}-submit.json'; assert not receipt.exists(), 'Reconcile the previous receipt; do not duplicate'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    c = json.loads((E / 'config' / f'{which}.json').read_text())
    cmd = ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-e14-{which}-{ATTEMPT}', '--gpu', 'a100', '--gpus', '1',
           '--cpus', '24', '--mem', '128G', '--time', TIME[which]]
    for u in c['allowed_uuids']:
        cmd += ['--allowed-uuid', u]
    cmd += ['--chdir', c['workspace'], '--output', str(E / f'slurm/{which}-%j.out'), '--source-manifest', str(E / 'slurm-source-manifest.json'), '--',
            '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e14_runner.py'), '--config', str(E / 'config' / f'{which}.json')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'sanity', 'main', 'extra']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit(a.action)
