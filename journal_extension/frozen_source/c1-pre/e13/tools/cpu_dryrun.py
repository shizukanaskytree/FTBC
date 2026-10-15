"""CPU-only logic dry run of the E13 runner + acceptance (sanity sizes, images downsampled to 32x32). Never initializes CUDA.
Exercises every code path before the GPU sanity; numbers are meaningless (CPU arithmetic, tiny images)."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E13 = Path(__file__).resolve().parents[1]
import torch, torch.nn.functional as F
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.max_memory_reserved = lambda *a, **k: 0
torch.cuda.set_device = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
sys.path.insert(0, str(E13 / 'source'))
import e13_runner, e13_accept
original_load = e13_runner.load


def fake_guard(allowed):
    return {'slurm_job_id': '999999', 'pid': os.getpid(), 'cgroup': '0::/cpu-dryrun/slurm-mock\n', 'gpu_uuid': allowed[0],
            'name': 'CPU dry run (no CUDA)', 'total_memory': 84974239744, 'torch': torch.__version__, 'cuda_runtime': None, 'cudnn': None}


def load(name, path):
    module = original_load(name, path)
    if hasattr(module, 'loader') and hasattr(module, 'evaluate'):
        real = module.loader
        def small(rows, config, batch_size, shuffle=False):
            for images, labels, ids in real(rows, config, batch_size, shuffle):
                yield F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False), labels, ids
        module.loader = small
    return module


e13_runner.gpu_guard = fake_guard; e13_runner.load = load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time()}
try:
    c = json.loads((E13 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E13 / 'source' / f for f in ['serial_solver.py', 'e13_runner.py', 'e13_accept.py']] + [E13 / 'EXPERIMENT_PLAN.md']
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}))
    c['output'] = str(SCRATCH / 'sanity-cpu'); c['freeze_manifest'] = str(fake / 'protocol-freeze.json')
    (fake / 'config/sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / 'slurm/sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(e13_runner.run(c)))
    report['runner_seconds'] = time.time() - t0
    shutil.copy(E13 / 'preservation-before.json', fake / 'preservation-before.json')
    e13_accept.E = fake
    with open(SCRATCH / 'accept-console.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        e13_accept.check('sanity')
    acc = json.loads((fake / 'sanity-acceptance.json').read_text())
    code = [E13 / 'source' / f for f in ['serial_solver.py', 'e13_runner.py', 'e13_accept.py']] + [E13 / 'config/sanity.json', E13 / 'config/main.json']
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report.update(passed=acc['passed'], arms=sorted(acc['metrics']), decisions={k: {kk: vv for kk, vv in v.items() if kk != 'pairs'} for k, v in acc['decisions'].items()},
                  mechanism_predictions=acc['mechanism_predictions'], note='CPU 32x32 logic dry run; numbers meaningless')
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized', 'runner_seconds']}))
    if 'traceback' in report: print(report['traceback'])
