"""Freeze E25 (chained to the E18 freeze, which covers the reused E13/E14 solver code, the E18 Run class, the released FTBC export and the serial
source snapshot) and submit its four gpurun jobs, one CIFAR source model and one child chain each (e25_job.py; EXPERIMENT_PLAN.md section 8).
Usage: e25_freeze_submit.py freeze | job <tag>."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parents[1]; C1 = E.parent
TIMES = {'c10-vgg16': '06:00:00', 'c100-vgg16': '06:00:00', 'c10-resnet20': '06:00:00', 'c100-resnet20': '06:00:00'}   # section 8; submit in this order
ATTEMPT = 'v1'
GPU_MEM = 16                                                  # section 8: a 16 GiB shard (gpurun --gpu-mem; the user's 2026-10-10 GPU-sharing rule)
A100 = ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']
H100 = ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211']
GPU = {t: (('a100', 'h100'), A100 + H100) for t in TIMES}
TYPE = {'a100': A100, 'h100': H100}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def collect(with_review=True):
    """Every freeze entry. with_review=False leaves out the review files and the CPU-check report (the dry freeze of the CPU check)."""
    files = {}
    for parent in [C1 / 'e18/protocol-freeze.json']:
        files.update(json.loads(parent.read_text())['files_sha256']); files[str(parent)] = sha(parent)
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json'))
    names = ['EXPERIMENT_PLAN.md', 'preservation-before.json', 'input/cifar10-rows.json', 'input/cifar100-rows.json', 'input/preparation.json',
             'record_costs.py', 'verify_preservation.py', 'tools/cpu_dryrun.py', 'tools/make_configs.py']
    names += [f'cpu-dryrun/{t}/{f}' for t in TIMES for f in ['dryrun-report.json', f'cpu-{t}-sanity-acceptance.json']]
    names += ['cache/cpu-check.json', 'review/disposition.json', 'review/code-final.json'] if with_review else []
    paths += [E / f for f in names]
    paths += [C1 / 'e14/source/e14_accept.py', C1 / 'e16/source/e16_accept.py', C1 / 'source/run_probe.py',
              Path('/data3/snn_prjs/ftbc_neural_networks/code/ftbc-eccv/QCFS/ANN_SNN_QCFS/Preprocess/augment.py')]
    for which in ['main', 'sanity']:
        for tag in TIMES:
            c = json.loads((E / f'config/{tag}-{which}.json').read_text())
            for k in ['e13_runner', 'e18_runner', 'cifar_e0', 'step_solver', 'serial_solver', 'calibration_export', 'calibration_export_manifest',
                      'source_checkpoint', 'preparation', 'rows_manifest']:
                paths.append(Path(c[k]))
    prep = json.loads((E / 'input/preparation.json').read_text()); paths += [Path(p) for p in prep['exclusion_sources']]
    for p in paths:
        files[str(p)] = sha(p)
    for p, h in files.items():
        assert sha(p) == h, p
    return files


def manuscript_current():
    """Initial review N-g: verify_preservation.py asserts that HEAD, the index entries and the staged changes of manuscript/ equal
    preservation-before.json (follow-up 1 R1), so a snapshot that is already stale at freeze would fail the integrity check. Read-only git commands."""
    M = Path('/data3/snn_prjs/ftbc_neural_networks/manuscript'); before = json.loads((E / 'preservation-before.json').read_text())
    git = lambda *a: subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)
    cur = {'HEAD': git('rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git('ls-files', '--stage').encode()).hexdigest(), 'staged': git('diff', '--cached', '--name-status')}
    return all(cur[k] == before[k] for k in cur)


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
        for model in TIMES:
            rep = json.loads((E / f'cpu-dryrun/{model}/dryrun-report.json').read_text()); assert rep['passed'] and not rep['cuda_initialized'], model
            for p, h in rep['code_sha256'].items():
                assert sha(p) == h, f'{model} dry run is stale for {p}'
        assert manuscript_current(), 'preservation-before.json is stale: HEAD, index entries or staged changes moved; re-snapshot before freezing'
    else:
        target = Path(dry_dir) / 'protocol-freeze.json'; target.parent.mkdir(parents=True, exist_ok=True)
    files = collect(with_review=dry_dir is None)
    target.write_text(json.dumps({'schema': 'ftbc.e25.cifar-development-screen.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'attempt': ATTEMPT,
                                  'protected_evaluation': False, 'dry': dry_dir is not None, 'files_sha256': files}, indent=2) + '\n')
    w = Path('/data3/snn_prjs/ftbc_neural_networks')
    staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)}
    staged[str((E / 'protocol-freeze.json').relative_to(w))] = sha(target)
    (target.parent / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': staged}, indent=2) + '\n')
    print('Frozen' if dry_dir is None else 'Dry-frozen', len(files), 'entries;', len(staged), 'workspace files in the Slurm source manifest')
    return files, staged


def command(tag, kind):
    """kind: the card type chosen at submission among the allowed ones, by which type is expected to free first (the time limits are
    computed for the slower A100 unless only H100 is allowed); only that type's UUIDs are passed."""
    kinds, allowed = GPU[tag]; assert kind in kinds, (tag, kind)
    uuids = [x for u in allowed if u in TYPE[kind] for x in ('--allowed-uuid', u)]
    return ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'c1pre-e25-{tag}-{ATTEMPT}', '--gpu', kind, '--gpu-mem', str(GPU_MEM), '--cpus', '16', '--mem', '128G',
            '--time', TIMES[tag], *uuids, '--chdir', '/data3/snn_prjs/ftbc_neural_networks', '--output', str(E / f'slurm/job-{tag}-%j.out'),
            '--source-manifest', str(E / 'slurm-source-manifest.json'), '--', '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e25_job.py'), tag]


def submit(tag, kind):
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    assert not (E / f'{tag}-sanity-acceptance.json').exists() and not (E / f'{tag}-main-acceptance.json').exists(), tag
    receipt = E / f'slurm/job-submit-{tag}.json'; assert not receipt.exists(), 'Reconcile the previous receipt; do not duplicate'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(tag, kind); r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'kind': kind, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'job']); p.add_argument('tag', nargs='?', choices=list(TIMES))
    p.add_argument('kind', nargs='?', choices=['a100', 'h100']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit(a.tag, a.kind)
