"""Freeze E29 (chained to the E23 freeze, which chains E22 -> E20 -> E19/E18 for the reused R34 and VGG-16 code) and submit its single gpurun
job: the R34 and VGG-16 chains run one after the other in one allocation (e29_job.py; EXPERIMENT_PLAN.md section 7). Adapted from E23."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parents[1]; C1 = E.parent
TIME = '04:00:00'                                      # section 7 (initial review N6): estimate about 2.2 h with the paired arm; rounded up with margin
ATTEMPT = 'v1'
H100 = 'GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'
DRYRUNS = ['r34', 'r34-oom', 'r34-libnoev', 'r34-runtime', 'vgg']   # section 6 item 8 (initial review B4)
ACCEPTED_DRYRUNS = ['r34', 'r34-oom', 'vgg']          # the two failure injections leave no acceptance


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def collect(with_review=True):
    """Every freeze entry. with_review=False leaves out the review files and the CPU-check report (the dry freeze of the CPU check)."""
    files = {}
    def put(p, h):                                     # never overwrite an entry; one already present must agree (E26 initial review B1)
        assert files.setdefault(str(p), h) == h, f'{p}: hash differs from its parent freeze'
    for parent in [C1 / 'e23/protocol-freeze.json']:
        for q, h in json.loads(parent.read_text())['files_sha256'].items():
            put(q, h)
        put(parent, sha(parent))
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json'))
    names = ['EXPERIMENT_PLAN.md', 'preservation-before.json', 'record_costs.py', 'verify_preservation.py', 'tools/cpu_dryrun.py', 'tools/unit_checks.py']
    names += [f'cpu-dryrun/{t}/dryrun-report.json' for t in DRYRUNS] + [f'cpu-dryrun/{t}/cpu-{t}-sanity-acceptance.json' for t in ACCEPTED_DRYRUNS]
    names += ['cache/cpu-check.json', 'review/disposition.json', 'review/code-final.json'] if with_review else []
    paths += [E / f for f in names]
    for which in ['main', 'sanity']:
        for model in ['r34', 'vgg']:
            c = json.loads((E / f'config/{model}-{which}.json').read_text())
            paths += [Path(p) for p in c['previous_fresh_manifests'] + c.get('previous_calibration_manifests', [])]
            for k in ['e0_runner', 'e13_runner', 'e14_runner', 'e15_runner', 'e16_runner', 'e17_runner', 'e18_runner', 'e23_r34_runner', 'ce_solver', 'parent_config',
                      'plain_deployment', 'plain_proxy', 'e0_artifacts', 'e5_roles', 'calibration_export', 'calibration_export_manifest', 'fresh_manifest',
                      'calibration_manifest', 'step_solver', 'serial_solver', 'source_checkpoint', 'dataset_manifest', 'e23_release_record']:   # follow-up 1 N2
                if k in c:
                    paths.append(Path(c[k]))
    for p in paths:
        put(p, sha(p))
    for p, h in files.items():
        assert sha(p) == h, p
    return files


def freeze(dry_dir=None):
    """dry_dir: write both manifests there instead and skip the review gate (used by the CPU check)."""
    if dry_dir is None:
        target = E / 'protocol-freeze.json'; assert not target.exists()
        d = json.loads((E / 'review/disposition.json').read_text()); assert d['release'] and not d['unresolved_blocking']
        final = json.loads((E / 'review/code-final.json').read_text()); verdict = final['response'].strip().splitlines()[-1]
        assert final['status'] == 'completed' and final['model'] == 'claude-opus-5-5' and 'release' in verdict and 'not release' not in verdict, verdict
        for p, h in d['post_fix_files_sha256'].items():
            assert sha(p) == h, 'file changed after the release review: ' + p
        rep = json.loads((E / 'cache/cpu-check.json').read_text()); assert rep['passed'] and not rep['cuda_initialized']
        for p, h in rep['files_sha256'].items():
            assert sha(p) == h, f'cpu-check is stale for {p}'
        for t in DRYRUNS:
            rep = json.loads((E / f'cpu-dryrun/{t}/dryrun-report.json').read_text()); assert rep['passed'] and not rep['cuda_initialized'] and rep['tag'] == t, t
            for p, h in rep['code_sha256'].items():
                assert sha(p) == h, f'{t} dry run is stale for {p}'
    else:
        target = Path(dry_dir) / 'protocol-freeze.json'; target.parent.mkdir(parents=True, exist_ok=True)
    files = collect(with_review=dry_dir is None)
    target.write_text(json.dumps({'schema': 'ftbc.e29.chunked-release-default-equivalence.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'attempt': ATTEMPT,
                                  'protected_evaluation': False, 'dry': dry_dir is not None, 'files_sha256': files}, indent=2) + '\n')
    w = Path('/data3/snn_prjs/ftbc_neural_networks')
    staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)}
    staged[str((E / 'protocol-freeze.json').relative_to(w))] = sha(target)
    (target.parent / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': staged}, indent=2) + '\n')
    print('Frozen' if dry_dir is None else 'Dry-frozen', len(files), 'entries;', len(staged), 'workspace files in the Slurm source manifest')
    return files, staged


def command():
    return ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-e29-job-{ATTEMPT}', '--gpu', 'h100', '--gpus', '1', '--cpus', '16', '--mem', '128G',
            '--time', TIME, '--allowed-uuid', H100, '--chdir', '/data3/snn_prjs/ftbc_neural_networks', '--output', str(E / 'slurm/job-%j.out'),
            '--source-manifest', str(E / 'slurm-source-manifest.json'), '--', '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e29_job.py')]


def submit():
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    for model in ['r34', 'vgg']:
        assert not (E / f'{model}-sanity-acceptance.json').exists() and not (E / f'{model}-main-acceptance.json').exists(), model
    receipt = E / 'slurm/job-submit.json'; assert not receipt.exists(), 'Reconcile the previous receipt; do not duplicate'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(); r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'job']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit()
