"""CPU-only logic dry run of the E15 runner + acceptance for every job configuration (EXPERIMENT_PLAN.md section 7 item 10):
stages [main, release, extra] (sanity), [main], [release], [extra], plus [release] with an injected out-of-memory error in the
middle of the R2 release recipe (after the released GetLayerInputOutput hook was registered), all at sanity sizes with images
downsampled to 32x32. Never initializes CUDA. Also checks that ftbc_batches reproduces the frozen E13 ftbc_calibrate bitwise."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E15 = Path(__file__).resolve().parents[1]
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
sys.path.insert(0, str(E15 / 'source'))
import e15_runner, e15_accept
original_load = e15_runner.load


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


def patched_load(name, path):
    module = original_load(name, path)
    if name == 'e15_e14':                       # E14 module: patch its own loader of E13 so E13's globals get the fakes
        inner = module.load
        def load13(n, p):
            m = inner(n, p)
            if n == 'e14_e13':
                m.gpu_guard = fake_guard; m.load = small_images(m.load)
            return m
        module.load = load13
    return module


e15_runner.load = patched_load
FAULT = {'seed': None}
real_build = e15_runner.build
def build(c):
    out = real_build(c); Run = out[-1]
    orig = Run.release_batches
    def release_batches(self, rows, seed):          # same recipe, images downsampled after augmentation
        batches, order = orig(self, rows, seed)
        return [(small(x), y) for x, y in batches], order
    Run.release_batches = release_batches
    orig_ftbc = Run.ftbc_batches
    def ftbc_batches(self, arm, T, batches, alpha, images, recipe, extra=None):
        if extra and extra.get('seed') == FAULT['seed']:   # inject an OOM on the 5th saved activation of the released helper
            cls = self.inherited.ActivationSaverHookEachTimeStep; real_call = cls.__call__; n = {'k': 0}
            def failing(hook, module, inp, outp):
                n['k'] += 1
                if n['k'] == 5: raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected by the CPU dry run)')
                return real_call(hook, module, inp, outp)
            cls.__call__ = failing
            try:
                return orig_ftbc(self, arm, T, batches, alpha, images, recipe, extra)
            finally:
                cls.__call__ = real_call
        return orig_ftbc(self, arm, T, batches, alpha, images, recipe, extra)
    Run.ftbc_batches = ftbc_batches
    return out
e15_runner.build = build
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'configs': {}}
try:
    base = json.loads((E15 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E15 / 'source' / f for f in ['e15_runner.py', 'e15_accept.py']] + [E15 / 'EXPERIMENT_PLAN.md', Path(base['e14_runner']), Path(base['e13_runner'])]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E15 / 'preservation-before.json', fake / 'preservation-before.json')
    e15_accept.E = fake
    for name, stages in [('sanity', ['main', 'release', 'extra']), ('main', ['main']), ('release', ['release']), ('extra', ['extra']), ('release-fault', ['release'])]:
        FAULT['seed'] = base['release_sets']['R2'][2] if name == 'release-fault' else None
        c = dict(base, stages=stages, output=str(SCRATCH / f'{name}-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
        (fake / f'config/{name}.json').write_text(json.dumps(c, indent=2))
        t0 = time.time()
        with open(fake / f'slurm/{name}-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
            print(json.dumps(e15_runner.run(c)))
        with open(SCRATCH / f'accept-{name}.txt', 'w') as fh, contextlib.redirect_stdout(fh):
            e15_accept.check(name)
        acc = json.loads((fake / f'{name}-acceptance.json').read_text())
        report['configs'][name] = {'stages': stages, 'passed': acc['passed'], 'runner_seconds': time.time() - t0, 'arms': sorted(acc['metrics'])}
        if 'release' in stages:
            res = json.loads((SCRATCH / f'{name}-cpu/result.json').read_text())
            report['configs'][name]['release'] = {k: {kk: vv for kk, vv in v.items() if kk != 'traceback'} for k, v in res['release'].items()}
            report['configs'][name]['Q3'] = {k: v for k, v in acc['decisions']['Q3'].items() if k != 'pairs'}
        shutil.copy(fake / f'{name}-acceptance.json', SCRATCH / f'cpu-{name}-acceptance.json')
    # ftbc_batches equals the frozen E13 ftbc_calibrate on the same registered batches (bitwise biases).
    c = dict(base, stages=['main'], output=str(SCRATCH / 'equiv-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    E13, E14, ss, ps, Run = e15_runner.build(c); r = Run(c); T = 4
    rows = r.rows['ftbc'][:3 * c['ftbc_batch']]
    a = r.ftbc_batches('eq-new', T, r.registered_batches(rows), c['ftbc_alpha'], [int(x['index']) for x in rows], 'registered')
    b = r.ftbc_calibrate('eq-old', T, 3, r.zero_biases(T))
    report['ftbc_batches_equals_e13_ftbc_calibrate'] = all(torch.equal(x, y) for n in a for x, y in zip(a[n], b[n]))
    assert report['ftbc_batches_equals_e13_ftbc_calibrate']
    code = [E15 / 'source' / f for f in ['e15_runner.py', 'e15_accept.py']] + [E15 / 'config' / f for f in ['sanity.json', 'main.json', 'release.json', 'extra.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    flt = report['configs']['release-fault']
    assert flt['release']['R2']['status'] == 'out_of_memory' and flt['release']['R2']['leftover_hooks_dropped'] >= 1 and not flt['Q3']['pairs_pass']
    assert flt['release']['R1']['status'] == flt['release']['R3']['status'] == 'completed' and 'F0-release-R2' not in flt['arms']
    report['fault_injection'] = {'R2_status': flt['release']['R2']['status'], 'leftover_hooks_dropped': flt['release']['R2']['leftover_hooks_dropped'], 'Q3_pairs_pass': flt['Q3']['pairs_pass']}
    report['passed'] = all(v['passed'] for v in report['configs'].values()) and len(report['configs']) == 5
    report['note'] = 'CPU 32x32 logic dry run at sanity sizes for each stage combination; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized']} | {'configs': {k: v['passed'] for k, v in report['configs'].items()}}))
    if 'traceback' in report: print(report['traceback'])
