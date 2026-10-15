"""Freeze the locked evaluation v2 implementation and submit its ten gpurun jobs (plan sections 2 and 5; adapted from E25/E29).
Usage: l1_freeze_submit.py freeze | job <tag> <a100|h100>. ImageNet jobs take a whole card: R34 T=4 and T=8 the H100 only, the others the H100 or
either A100 (initial review B3: the configs allow exactly those cards, and the submission checks it); CIFAR jobs take a 16 GiB shard of
either card type. The time limit depends on the card type (TIMES).
The freeze is chained to the E29 freeze (E23 -> E22 -> ... for the ImageNet chains; E29's release_chunked module is loaded but the release-default
arm is not run, plan candidate 4) and to the E25 freeze (the CIFAR
construction on the E18 Run); entries present in both must agree. The plan itself is not in the freeze: the final plan states the SHA-256 of
protocol-freeze.json, and the author approves the plan's SHA-256 with "开启评价" (approval.json). A job is submitted only if approval.json names
the current plan hash, the plan states the current freeze hash, and every frozen file is unchanged."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
from datetime import datetime, timezone
E = Path(__file__).resolve().parents[1]; RC = E.parent; C1 = RC / 'c1-pre'
PLAN = RC / 'innovation/official-evaluation-plan-v2.md'
ATTEMPT = 'v1'
IMAGENET = [f'{m}-t{T}' for m in ['r34', 'vgg'] for T in (2, 4, 8)]
CIFAR = ['c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20']
TIMES = {'r34-t2': {'h100': '08:00:00', 'a100': '16:00:00'}, 'r34-t4': {'h100': '12:00:00'}, 'r34-t8': {'h100': '18:00:00'},
         'vgg-t2': {'h100': '05:00:00', 'a100': '08:00:00'}, 'vgg-t4': {'h100': '06:00:00', 'a100': '12:00:00'}, 'vgg-t8': {'h100': '08:00:00', 'a100': '14:00:00'},
         **{t: {'h100': '04:00:00', 'a100': '04:00:00'} for t in CIFAR}}
# section 5: whole cards for ImageNet with at least 1.7x the H100 estimates; on an A100, which other jobs and external processes may share, at
# least 5x the H100 estimate plus 1 h (follow-up 1); R34 T=4 and T=8 run on the H100 only; 16 GiB shards for CIFAR on either card type
CPUS = {**{t: 16 for t in IMAGENET}, **{t: 8 for t in CIFAR}}        # CIFAR loads in the main process (0 workers), follow-up 1
MIN_FREE_GIB = 40                                                    # an A100 whole-card job is submitted only if one A100 has this much free (recorded)
GPU_MEM = 16                                                         # CIFAR: a 16 GiB shard (the user's 2026-10-10 GPU-sharing rule), as E25
A100 = ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']
H100 = ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211']
TYPE = {'a100': A100, 'h100': H100}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def collect(with_review=True):
    files = {}
    def put(p, h):
        assert files.setdefault(str(p), h) == h, f'{p}: hash differs between parent freezes'
    for parent in [C1 / 'e29/protocol-freeze.json', C1 / 'e25/protocol-freeze.json']:
        for q, h in json.loads(parent.read_text())['files_sha256'].items():
            put(q, h)
        put(parent, sha(parent))
    paths = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json')) + list((E / 'input').glob('*.json'))
    names = ['README.md', 'preservation-before.json', 'record_costs.py', 'verify_preservation.py', 'tools/cpu_dryrun.py']
    names += [f'cpu-dryrun/{t}/dryrun-report.json' for t in ['r34-t8', 'vgg-t8', 'c10-vgg16', 'c10-vgg16-test', 'c100-resnet20']]
    names += ['cache/cpu-check.json', 'review/disposition.json', 'review/code-final.json'] if with_review else []
    paths += [E / f for f in names]
    for tag in IMAGENET + CIFAR:
        for phase in ['sanity', 'main']:
            c = json.loads((E / f'config/{tag}-{phase}.json').read_text())
            for k in ['e0_runner', 'e13_runner', 'e14_runner', 'e15_runner', 'e16_runner', 'e17_runner', 'e18_runner', 'e23_r34_runner', 'e25_runner', 'cifar_e0',
                      'cifar_test', 'release_chunked', 'ce_solver', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_artifacts', 'e5_roles',
                      'calibration_export', 'calibration_export_manifest', 'fresh_manifest', 'calibration_manifest', 'rows_manifest', 'preparation',
                      'step_solver', 'serial_solver', 'source_checkpoint', 'dataset_manifest', 'evaluation_manifest', 'standin_manifest']:
                # 'test_file' (the CIFAR test pickle) is not hashed here: it is not read before the approval; the runner checks it against
                # the inventory hash recorded in input/preparation.json
                if k in c and c[k]:
                    paths.append(Path(c[k]))
            paths += [Path(p) for p in c.get('previous_fresh_manifests', []) + c.get('previous_calibration_manifests', [])]
    for p in paths:
        put(p, sha(p))
    for p, h in files.items():
        assert sha(p) == h, p
    return files


def freeze(dry_dir=None):
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
    else:
        target = Path(dry_dir) / 'protocol-freeze.json'; target.parent.mkdir(parents=True, exist_ok=True)
    files = collect(with_review=dry_dir is None)
    target.write_text(json.dumps({'schema': 'ftbc.locked-evaluation-v2.v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'attempt': ATTEMPT,
                                  'protected_evaluation': True, 'dry': dry_dir is not None, 'files_sha256': files}, indent=2) + '\n')
    w = Path('/data3/snn_prjs/ftbc_neural_networks')
    staged = {str(Path(p).relative_to(w)): h for p, h in files.items() if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w)}
    staged[str((E / 'protocol-freeze.json').relative_to(w))] = sha(target)
    (target.parent / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': staged}, indent=2) + '\n')
    print('Frozen' if dry_dir is None else 'Dry-frozen', len(files), 'entries;', len(staged), 'workspace files in the Slurm source manifest')
    return files, staged


def approved():
    """The author's approval names the plan hash; the plan names the freeze hash."""
    a = json.loads((E / 'approval.json').read_text()); plan_sha = sha(PLAN)
    assert a['plan_sha256'] == plan_sha and a['author_reply'].strip() == '开启评价', 'no author approval for the current plan version'
    m = re.search(r'protocol-freeze\.json` 的 SHA-256：`([0-9a-f]{64})`', PLAN.read_text()); assert m, 'the plan does not state the freeze hash'
    assert m.group(1) == sha(E / 'protocol-freeze.json'), 'the plan names a different freeze'
    return a


def command(tag, kind):
    assert kind in TYPE and kind in TIMES[tag], f'{tag} does not run on {kind} (section 5)'
    allowed = json.loads((E / f'config/{tag}-main.json').read_text())['allowed_uuids']
    assert set(TYPE[kind]) <= set(allowed), f'the {tag} configs do not allow the {kind} cards'
    uuids = [x for u in TYPE[kind] for x in ('--allowed-uuid', u)]
    share = ['--gpu-mem', str(GPU_MEM)] if tag in CIFAR else ['--gpus', '1']
    return ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', f'locked-eval-v2-{tag}-{ATTEMPT}', '--gpu', kind, *share, '--cpus', str(CPUS[tag]), '--mem', '128G',
            '--time', TIMES[tag][kind], *uuids, '--chdir', '/data3/snn_prjs/ftbc_neural_networks', '--output', str(E / f'slurm/job-{tag}-%j.out'),
            '--source-manifest', str(E / 'slurm-source-manifest.json'), '--', '/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/l1_job.py'), tag]


def submit(tag, kind):
    approved()
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    assert not (E / f'{tag}-sanity-acceptance.json').exists() and not (E / f'{tag}-main-acceptance.json').exists(), tag
    receipt = E / f'slurm/job-submit-{tag}.json'; assert not receipt.exists(), 'Reconcile the previous receipt; do not duplicate'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    cmd = command(tag, kind)
    q = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    free = {l.split(',')[0].strip(): (int(l.split(',')[2]) - int(l.split(',')[1])) / 1024 for l in q.strip().splitlines()}
    if tag in IMAGENET and kind == 'a100':                            # follow-up 1: other jobs and external processes may hold A100 memory
        assert any(free.get(u, 0) >= MIN_FREE_GIB for u in A100), f'no A100 has {MIN_FREE_GIB} GiB free now: {free}'
    r = subprocess.run(cmd, capture_output=True, text=True)
    receipt.write_text(json.dumps({'command': cmd, 'kind': kind, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr,
                                   'free_gib_at_submission': free, 'submitted_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    print(r.stdout, r.stderr); assert r.returncode == 0, 'check squeue for the run id before anything else (a timeout may still have submitted the job)'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'job']); p.add_argument('tag', nargs='?', choices=IMAGENET + CIFAR)
    p.add_argument('kind', nargs='?', choices=['a100', 'h100']); a = p.parse_args()
    freeze() if a.action == 'freeze' else submit(a.tag, a.kind)
