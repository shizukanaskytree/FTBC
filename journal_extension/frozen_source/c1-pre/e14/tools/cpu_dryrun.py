"""CPU-only logic dry run of the E14 runner + acceptance (sanity config: T=4 main stage and T=2 extra stage; images downsampled
to 32x32). Never initializes CUDA. Exercises every code path before the GPU sanity; numbers are meaningless (CPU, tiny images)."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E14 = Path(__file__).resolve().parents[1]
import torch, torch.nn.functional as F
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.max_memory_reserved = lambda *a, **k: 0
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
sys.path.insert(0, str(E14 / 'source'))
import e14_runner, e14_accept
original_load = e14_runner.load


def fake_guard(allowed):
    return {'slurm_job_id': '999999', 'pid': os.getpid(), 'cgroup': '0::/cpu-dryrun/slurm-mock\n', 'gpu_uuid': allowed[0],
            'name': 'CPU dry run (no CUDA)', 'total_memory': 84974239744, 'torch': torch.__version__, 'cuda_runtime': None, 'cudnn': None}


def small_images(load):
    def wrapped(name, path):
        module = load(name, path)
        if hasattr(module, 'loader') and hasattr(module, 'evaluate'):
            real = module.loader
            def small(rows, config, batch_size, shuffle=False):
                for images, labels, ids in real(rows, config, batch_size, shuffle):
                    yield F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False), labels, ids
            module.loader = small
        return module
    return wrapped


def load14(name, path):
    module = original_load(name, path)
    if name == 'e14_e13':
        module.gpu_guard = fake_guard; module.load = small_images(module.load)
    return module


e14_runner.load = load14
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time()}
try:
    c = json.loads((E14 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E14 / 'source' / f for f in ['step_solver.py', 'e14_runner.py', 'e14_accept.py']] + [E14 / 'EXPERIMENT_PLAN.md', Path(c['e13_runner'])]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(c['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    c['output'] = str(SCRATCH / 'sanity-cpu'); c['freeze_manifest'] = str(fake / 'protocol-freeze.json')
    (fake / 'config/sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / 'slurm/sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(e14_runner.run(c)))
    report['runner_seconds'] = time.time() - t0
    shutil.copy(E14 / 'preservation-before.json', fake / 'preservation-before.json')
    e14_accept.E = fake
    with open(SCRATCH / 'accept-console.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        e14_accept.check('sanity')
    acc = json.loads((fake / 'sanity-acceptance.json').read_text())
    code = [E14 / 'source' / f for f in ['step_solver.py', 'e14_runner.py', 'e14_accept.py']] + [E14 / 'config' / f for f in ['sanity.json', 'main.json', 'extra.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report.update(passed=acc['passed'], arms=sorted(acc['metrics']), decisions={k: ({kk: vv for kk, vv in v.items() if kk != 'pair'} if isinstance(v, dict) else v)
                                                                             for k, v in acc.get('decisions', {}).items()},
                  mechanism_predictions=acc.get('mechanism_predictions'), solver_summary=acc['solver_summary'], note='CPU 32x32 logic dry run; numbers meaningless')
    shutil.copy(fake / 'sanity-acceptance.json', SCRATCH / 'cpu-sanity-acceptance.json')
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized', 'runner_seconds']}))
    if 'traceback' in report: print(report['traceback'])
