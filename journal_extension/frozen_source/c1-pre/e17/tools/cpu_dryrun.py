"""CPU-only logic dry run of the E17 runner + acceptance at sanity sizes with images downsampled to 32x32 (EXPERIMENT_PLAN.md section 7).
Never initializes CUDA. Also checks prefix consistency (P0 and CE at T=2 equal the first two steps of T=4 bitwise; CE-A need not) and
reports how closely the CE-A last step holds the cumulative channel mean, separately for in-range and out-of-range anchor targets."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E17 = Path(__file__).resolve().parents[1]
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
sys.path.insert(0, str(E17 / 'source'))
import e17_runner, e17_accept
original_load = e17_runner.load


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
    """Patch each runner's own loader along E17 -> E16 -> E15 -> E14 -> E13 so E13's globals get the fakes."""
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
    if name == 'e17_e16':
        chain(module, ['e16_e15', 'e15_e14', 'e14_e13'])
    return module


e17_runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'configs': {}}
try:
    base = json.loads((E17 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E17 / 'source' / f for f in ['e17_runner.py', 'e17_accept.py', 'ce_solver.py']] + [E17 / 'EXPERIMENT_PLAN.md', Path(base['e16_runner']), Path(base['e15_runner']), Path(base['e14_runner']), Path(base['e13_runner'])]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E17 / 'preservation-before.json', fake / 'preservation-before.json')
    e17_accept.E = fake
    name = 'sanity'
    c = dict(base, output=str(SCRATCH / f'{name}-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{name}.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / f'slurm/{name}-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(e17_runner.run(c)))
    with open(SCRATCH / f'accept-{name}.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        e17_accept.check(name)
    acc = json.loads((fake / f'{name}-acceptance.json').read_text())
    report['configs'][name] = {'passed': acc['passed'], 'runner_seconds': time.time() - t0, 'arms': sorted(acc['metrics']), 'adoption': acc['adoption']}
    shutil.copy(fake / f'{name}-acceptance.json', SCRATCH / f'cpu-{name}-acceptance.json')
    o = SCRATCH / f'{name}-cpu'; prefix = {}
    for kind in ['P0', 'CE', 'CE-A']:
        for d in ['C1', 'C2', 'C3']:
            b2 = torch.load(o / f'bias-{kind}-T2-{d}.pt', weights_only=False)['runtime_biases']; b4 = torch.load(o / f'bias-{kind}-T4-{d}.pt', weights_only=False)['runtime_biases']
            prefix[f'{kind}-{d}'] = all(torch.equal(b2[n][t], b4[n][t]) for n in b2 for t in range(2))
    report['prefix_identical_T2_vs_first_two_steps_of_T4'] = prefix
    assert all(v for k, v in prefix.items() if not k.startswith('CE-A')), prefix        # P0 and CE are prefix-consistent; CE-A anchors its own last step
    s = torch.load(o / 'solver.pt', weights_only=False); dev = {'in_range_max_quanta': 0.0, 'in_range_channels': 0, 'out_of_range_channels': 0, 'out_of_range_max_over_theta': 0.0}
    for arm, rec in s.items():
        if rec.get('objective') == 'CE-A':
            for n, v in rec['layers'].items():
                T = len(v['steps']); last = v['steps'][-1]; rate = sum(st['mean_after'] for st in v['steps']) / T
                gap = (rate - v['target_mean']).abs(); outr = last['above_range'] | last['below_range']
                quantum = v['theta'] / (T * v['observations_per_channel_per_step'])            # one count of the anchored step, in rate units
                if (~outr).any():
                    dev['in_range_max_quanta'] = max(dev['in_range_max_quanta'], float((gap[~outr] / quantum).max()))
                if outr.any():
                    dev['out_of_range_max_over_theta'] = max(dev['out_of_range_max_over_theta'], float((gap[outr] / v['theta']).max()))
                dev['in_range_channels'] += int((~outr).sum()); dev['out_of_range_channels'] += int(outr.sum())
    report['CE_A_anchor_deviation'] = dev           # reported, not asserted: tied margins can make some in-range targets miss by more than one count (as for P0)
    report['near_ties'] = {a: v['near_tie_mismatches'] for a, v in acc['solver_summary'].items() if 'near_tie_mismatches' in v}
    code = [E17 / 'source' / f for f in ['e17_runner.py', 'e17_accept.py', 'ce_solver.py']] + [E17 / 'config' / f for f in ['sanity.json', 'main.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['passed'] = acc['passed']
    report['note'] = 'CPU 32x32 logic dry run at sanity sizes; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized', 'CE_A_anchor_deviation', 'prefix_identical_T2_vs_first_two_steps_of_T4']}))
    if 'traceback' in report: print(report['traceback'])
