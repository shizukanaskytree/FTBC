"""CPU-only logic dry run of the E19 runner + acceptance at sanity sizes with images downsampled to 32x32 (EXPERIMENT_PLAN.md section 7).
Never initializes CUDA. Also checks P0 prefix consistency (T=4 equals the first four steps of T=8 bitwise) and that the released update is
restored after every variant call."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E19 = Path(__file__).resolve().parents[1]
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
sys.path.insert(0, str(E19 / 'source'))
import e19_runner, e19_accept
original_load = e19_runner.load


def fake_guard(allowed):
    return {'slurm_job_id': '999999', 'pid': os.getpid(), 'cgroup': '0::/cpu-dryrun/slurm-mock\n', 'gpu_uuid': allowed[0],
            'name': 'CPU dry run (no CUDA)', 'total_memory': 84974239744, 'torch': torch.__version__, 'cuda_runtime': None, 'cudnn': None}


def small(images):
    return F.interpolate(images, size=(32, 32), mode='bilinear', align_corners=False)


def small_images(load):
    def wrapped(name, path):
        module = load(name, path)
        if hasattr(module, 'loader') and hasattr(module, 'evaluate'):
            real = module.loader
            def loader(rows, config, batch_size, shuffle=False):
                for images, labels, ids in real(rows, config, batch_size, shuffle):
                    yield small(images), labels, ids
            module.loader = loader
        return module
    return wrapped


def chain(module, names):
    """Patch each runner's own loader along E19 -> E17 -> E16 -> E15 -> E14 -> E13 so E13's globals get the fakes."""
    inner = module.load
    def load_next(n, p):
        m = inner(n, p)
        if n == names[0]:
            if len(names) == 1:
                m.gpu_guard = fake_guard; m.load = small_images(m.load)
            else:
                chain(m, names[1:])
        return m
    module.load = load_next


def patched_load(name, path):
    module = original_load(name, path)
    if name == 'e19_e17':
        chain(module, ['e17_e16', 'e16_e15', 'e15_e14', 'e14_e13'])
    return module


e19_runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'configs': {}}
try:
    base = json.loads((E19 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E19 / 'source' / f for f in ['e19_runner.py', 'e19_accept.py', 'ftbc_variants.py']] + [E19 / 'EXPERIMENT_PLAN.md'] + [Path(base[k]) for k in ['e17_runner', 'e16_runner', 'e15_runner', 'e14_runner', 'e13_runner']]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E19 / 'preservation-before.json', fake / 'preservation-before.json')
    e19_accept.E = fake
    name = 'sanity'
    c = dict(base, output=str(SCRATCH / f'{name}-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{name}.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / f'slurm/{name}-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(e19_runner.run(c)))
    with open(SCRATCH / f'accept-{name}.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        e19_accept.check(name)
    acc = json.loads((fake / f'{name}-acceptance.json').read_text())
    report['configs'][name] = {'passed': acc['passed'], 'runner_seconds': time.time() - t0, 'arms': sorted(acc['metrics']), 'claims': acc['claims'],
                               'mechanism': acc['mechanism'], 'ftbc': acc['ftbc']}
    shutil.copy(fake / f'{name}-acceptance.json', SCRATCH / f'cpu-{name}-acceptance.json')
    report['prefix_consistency'] = acc['prefix_consistency_gpu']
    assert all(acc['prefix_consistency_gpu'].values()), acc['prefix_consistency_gpu']
    code = [E19 / 'source' / f for f in ['e19_runner.py', 'e19_accept.py', 'ftbc_variants.py']] + [E19 / 'config' / f for f in ['sanity.json', 'main.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['passed'] = acc['passed']
    report['note'] = 'CPU 32x32 logic dry run at sanity sizes; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized', 'prefix_consistency']}))
    if 'traceback' in report: print(report['traceback'])
