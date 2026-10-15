"""CPU-only logic dry run of one E25 job (runner + acceptance) at the job's sanity sizes and the real 32x32 CIFAR images (EXPERIMENT_PLAN.md
section 7; adapted from the E24 dry run). Never initializes CUDA. Usage: cpu_dryrun.py <scratch_dir> <job tag>. Patches only the E13 GPU guard
(through the E25 runner's loader) and the CUDA calls; the frozen-file check runs against a dry freeze in the scratch directory."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve(); TAG = sys.argv[2]
E25 = Path(__file__).resolve().parents[1]
import torch
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.max_memory_reserved = lambda *a, **k: 0
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.cuda.manual_seed_all = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
CAP_CALLS = []                                         # the runner's per-process cap (initial review N-e) is recorded, not applied, on CPU
torch.cuda.set_per_process_memory_fraction = lambda fraction, device=None: CAP_CALLS.append([fraction, device])
assert TAG in ('c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20')
torch.set_num_threads(int(os.environ.get('E25_DRYRUN_THREADS', '12')))
torch.set_num_threads = lambda n: None                 # num_workers=0 everywhere: keep the chosen thread count
sys.path.insert(0, str(E25 / 'source'))
runner = __import__('e25_runner'); accept = __import__('e25_accept')
original_load = runner.load


def fake_guard(allowed):
    return {'slurm_job_id': '999999', 'pid': os.getpid(), 'cgroup': '0::/cpu-dryrun/slurm-mock\n', 'gpu_uuid': allowed[0],
            'name': 'CPU dry run (no CUDA)', 'total_memory': 84974239744, 'torch': torch.__version__, 'cuda_runtime': None, 'cudnn': None}


def patched_load(name, path):
    module = original_load(name, path)
    if name == 'e25_e13':
        module.gpu_guard = fake_guard
    return module


runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'tag': TAG, 'python': sys.version, 'torch': torch.__version__}
try:
    base = json.loads((E25 / f'config/{TAG}-sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    chain_files = [Path(base[k]) for k in ['e18_runner', 'e13_runner', 'step_solver', 'serial_solver', 'calibration_export', 'cifar_e0']]
    # The protocol text is left out of the dry freeze, so a wording change does not invalidate a running dry run; the real freeze, which the
    # real acceptance checks, includes it.
    files = [E25 / 'source' / f for f in ['e25_runner.py', 'e25_accept.py']] + chain_files
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E25 / 'preservation-before.json', fake / 'preservation-before.json')
    accept.E = fake
    c = dict(base, output=str(SCRATCH / f'{TAG}-sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{TAG}-sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / f'slurm/{TAG}-sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(runner.run(c)))
    with open(SCRATCH / f'accept-{TAG}-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        accept.check(TAG, 'sanity')
    acc = json.loads((fake / f'{TAG}-sanity-acceptance.json').read_text())
    report.update(passed=acc['passed'], runner_seconds=time.time() - t0, Ts=c['Ts'], arms=len(acc['metrics']), claims=acc['claims'],
                  selection={k: v['selected'] for k, v in acc['selection'].items()}, equivalence=acc['equivalence'],
                  decisions={k: v['reading_e25'] for k, v in acc['decisions'].items()}, timing_seconds=acc['timing_seconds'], memory_fraction_calls=CAP_CALLS)
    assert len(CAP_CALLS) == 1 and CAP_CALLS[0] == [c['gpu_mem_cap_gib'] * 2 ** 30 / fake_guard(c['allowed_uuids'])['total_memory'], 0], CAP_CALLS
    shutil.copy(fake / f'{TAG}-sanity-acceptance.json', SCRATCH / f'cpu-{TAG}-sanity-acceptance.json')
    code = [E25 / 'source' / f for f in ['e25_runner.py', 'e25_accept.py', 'cifar_e0.py']] + [E25 / 'config' / f for f in [f'{TAG}-sanity.json', f'{TAG}-main.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU logic dry run at sanity sizes; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['tag', 'passed', 'cuda_initialized', 'runner_seconds', 'selection', 'decisions']}))
    if 'traceback' in report: print(report['traceback'])
