"""Freeze E12 (chained to the E11 freeze) and submit sanity/main through gpurun only."""
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
    parent = E.parent / 'e11/protocol-freeze.json'
    files = dict(json.loads(parent.read_text())['files_sha256']); files[str(parent)] = sha(parent)
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json'))
    paths += [E / f for f in ['EXPERIMENT_PLAN.md', 'RESEARCH_CONTRACT.md', 'preservation-before.json', 'cache/cpu-check.json', 'cache/math-check.json',
                              'review/disposition.json', 'review/code-final.json', 'review/concept-result.json', 'refine-logs/PROPOSAL.md',
                              'summarize.py', 'record_costs.py', 'verify_preservation.py', 'tools/review_client.py']]
    paths += [E.parent / 'e7/protocol-freeze.json']
    for name in ['sanity', 'main']:
        c = json.loads((E / 'config' / f'{name}.json').read_text())
        paths += [Path(c['parent_config']), Path(c['parent_runner']), Path(c['prior_freeze']), Path(c['kl_bias'])]
        paths += [Path(v) for b in ['p0', 'p1'] for v in c[b].values()]
        paths += [Path(c['e7_reference']['output']) / v for v in c['e7_reference']['files'].values()]
    for p in paths:
        files[str(p)] = sha(p)
    for p, h in files.items():
        assert sha(p) == h, p
    target.write_text(json.dumps({'schema': 'ftbc.e12.combination-control.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'protected_evaluation': False, 'files_sha256': files}, indent=2) + '\n')
    w = Path(json.loads((E / 'config/main.json').read_text())['workspace'])
    staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w)}
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
    cmd = ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-e12-{which}-v1', '--gpu', 'a100', '--gpus', '1',
           '--cpus', '24', '--mem', '128G', '--time', {'sanity': '00:30:00', 'main': '01:30:00'}[which],
           '--allowed-uuid', c['allowed_uuid'], '--chdir', c['workspace'], '--output', str(E / f'slurm/{which}-%j.out'),
           '--source-manifest', str(E / 'slurm-source-manifest.json'), '--',
           '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e12_runner.py'), '--config', str(E / 'config' / f'{which}.json')]
    r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'sanity', 'main']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit(a.action)
