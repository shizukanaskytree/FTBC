"""CPU-only logic dry run of one E22 child (runner + acceptance) at reduced sizes (EXPERIMENT_PLAN.md section 7; adapted from the E20 dry run). Never initializes CUDA.
Usage: cpu_dryrun.py <scratch_dir> {r34,vgg}. R34: sanity sizes with images downsampled to 32x32 (as the E19 dry run); VGG-16: smaller sizes at
224x224, because the VGG-16 ImageNet classifier needs 7x7 features (as the E18 dry run). Patches only the GPU guard, CUDA calls and (R34) the
image size, along each child's runner chain. The TPP arms run on CPU with the CPU generator (the seeds are set the same way)."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve(); MODEL = sys.argv[2]; assert MODEL in ('r34', 'vgg')
E22 = Path(__file__).resolve().parents[1]
import torch, torch.nn.functional as F
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
if MODEL == 'vgg':
    torch.set_num_threads(int(os.environ.get('E22_DRYRUN_THREADS', '12')))
    torch.set_num_threads = lambda n: None             # VGG dry run uses num_workers=0 (no DataLoader fork): keep the chosen thread count
sys.path.insert(0, str(E22 / 'source'))
runner = __import__(f'e22_{MODEL}_runner'); accept = __import__(f'e22_{MODEL}_accept')
original_load = runner.load


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
    """R34: patch each runner's own loader along E17 -> E16 -> E15 -> E14 -> E13 so E13's globals get the fakes (as the E19 dry run)."""
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
    if MODEL == 'r34' and name == 'e22_e17':
        chain(module, ['e17_e16', 'e16_e15', 'e15_e14', 'e14_e13'])
    if MODEL == 'vgg' and name == 'e22_e18_runner':                 # the E18 runner loads E13 for its GPU guard
        inner = module.load
        def load13(n, p):
            m = inner(n, p)
            if n == 'e18_e13':
                m.gpu_guard = fake_guard
            return m
        module.load = load13
    return module


runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'model': MODEL, 'python': sys.version, 'torch': torch.__version__}
try:
    base = json.loads((E22 / f'config/{MODEL}-sanity.json').read_text())
    if MODEL == 'vgg':
        base = dict(base, draw_n=8, small_n=6, ftbc_batch=2, eccv_batches=2, eccv_n=4, fresh_count=8, dev_count=4, solver_batch=4, num_workers=0)
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    chain_files = [Path(base[k]) for k in (['e17_runner', 'e16_runner', 'e15_runner', 'e14_runner', 'e13_runner'] if MODEL == 'r34' else ['e18_runner', 'e13_runner', 'step_solver', 'serial_solver'])]
    files = [E22 / 'source' / f for f in [f'e22_{MODEL}_runner.py', f'e22_{MODEL}_accept.py', 'e22_r34_accept.py', 'tpp_neuron.py']] + [E22 / 'EXPERIMENT_PLAN.md'] + chain_files
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E22 / 'preservation-before.json', fake / 'preservation-before.json')
    accept.E = fake
    c = dict(base, output=str(SCRATCH / f'{MODEL}-sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / f'slurm/{MODEL}-sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(runner.run(c)))
    with open(SCRATCH / f'accept-{MODEL}-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        accept.check('sanity')
    acc = json.loads((fake / f'{MODEL}-sanity-acceptance.json').read_text())
    report.update(passed=acc['passed'], runner_seconds=time.time() - t0, draw_n=c['draw_n'], seeds=c['seeds'], arms=sorted(acc['metrics']), claims=acc['claims'],
                  prefix_consistency=acc['prefix_consistency_gpu'], decisions={k: v['reading_e22'] for k, v in acc['decisions'].items()}, seed_spread=acc['seed_spread'])
    shutil.copy(fake / f'{MODEL}-sanity-acceptance.json', SCRATCH / f'cpu-{MODEL}-sanity-acceptance.json')
    code = [E22 / 'source' / f for f in [f'e22_{MODEL}_runner.py', f'e22_{MODEL}_accept.py', 'e22_r34_accept.py', 'tpp_neuron.py']] + [E22 / 'config' / f for f in [f'{MODEL}-sanity.json', f'{MODEL}-main.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU logic dry run at reduced sizes; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['model', 'passed', 'cuda_initialized', 'runner_seconds', 'decisions']}))
    if 'traceback' in report: print(report['traceback'])
