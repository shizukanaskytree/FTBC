"""CPU-only logic dry run of one E23 child (runner + acceptance) at reduced sizes (EXPERIMENT_PLAN.md section 6; adapted from the E22 dry run). Never initializes CUDA.
Usage: cpu_dryrun.py <scratch_dir> {r34,vgg,r34-score-assert,r34-update-error}. R34: sanity sizes with images downsampled to 32x32 (as the E19
dry run); VGG-16: smaller sizes at 224x224, because the VGG-16 ImageNet classifier needs 7x7 features (as the E18 dry run). Patches only the GPU
guard, CUDA calls and (R34) the image size, along each child's runner chain. The timing and memory calls run against CPU stand-ins (fixed
allocator readings), so the costs are meaningless; the code path that records them is what is tested. The R34 release-default arm loads its
own 224x224 augmented images (not resized).
Fault injection (initial review B3, follow-up 1 B5), R34 only. 'r34': inside the SNN forward of the third interface of the released update,
while the released saver hook is registered, F0rel-T4-r1 and F0rel-T8-r1 raise torch.cuda.OutOfMemoryError and F0rel-T8-r2 raises a
RuntimeError with the pre-registered signature CUDNN_STATUS_INTERNAL_ERROR while the device reports 0.5 GiB free; the run must continue,
exactly those arms must be recorded as out_of_memory at the update stage (basis exception, exception and library) with the hook dropped, and
the acceptance must pass (the other F0rel arms complete). 'r34-score-assert': the strict-reload scoring of F0rel-T2-r1 raises
AssertionError; 'r34-update-error': the released update of F0rel-T2-r1 raises a RuntimeError carrying the same signature but without allocator
evidence (40 GiB free, num_ooms unchanged); in both the runner must fail with that error.
Sampler logs (B4): after a run that completes, the gpuapps and gpumem logs are synthesized from the arm windows: the runner's own process every
second, with one gap of more than GAP s inside one arm and a foreign process inside another; the acceptance's per-arm readings must be unknown,
possibly_affected and no_other_process_seen respectively."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve(); TAG = sys.argv[2]; assert TAG in ('r34', 'vgg', 'r34-score-assert', 'r34-update-error')
MODEL = TAG.split('-')[0]; MODE = {'r34': 'oom', 'vgg': 'none'}.get(TAG, TAG[len('r34-'):])
INJECT = {'oom': ['F0rel-T4-r1', 'F0rel-T8-r1', 'F0rel-T8-r2'], 'score-assert': ['F0rel-T2-r1'], 'update-error': ['F0rel-T2-r1']}.get(MODE, [])
E23 = Path(__file__).resolve().parents[1]
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'     # as e23_job.py sets it; the acceptance asserts the recorded value (N6)
import torch, torch.nn.functional as F
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 3 << 30     # fixed stand-ins with 0 < allocated <= reserved <= card and baseline <= peak (N10)
torch.cuda.max_memory_reserved = lambda *a, **k: 4 << 30
torch.cuda.memory_allocated = lambda *a, **k: 1 << 30
torch.cuda.memory_reserved = lambda *a, **k: 2 << 30
torch.cuda.mem_get_info = lambda *a, **k: ((512 << 20) if MODE == 'oom' else (40 << 30), 80 << 30)   # read only for a library signature (B5)
LIB_ARMS = ['F0rel-T8-r2'] if MODE == 'oom' else []
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.cuda.manual_seed_all = lambda *a, **k: None
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
if MODEL == 'vgg':
    torch.set_num_threads(int(os.environ.get('E23_DRYRUN_THREADS', '12')))
    torch.set_num_threads = lambda n: None             # VGG dry run uses num_workers=0 (no DataLoader fork): keep the chosen thread count
sys.path.insert(0, str(E23 / 'source'))
runner = __import__(f'e23_{MODEL}_runner'); accept = __import__(f'e23_{MODEL}_accept')
accept_module = sys.modules['e23_r34_accept'] if MODEL == 'r34' else accept.RA
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
    if MODEL == 'r34' and name == 'e23_e17':
        chain(module, ['e17_e16', 'e16_e15', 'e15_e14', 'e14_e13'])
    if MODEL == 'vgg' and name == 'e23_e18_runner':                 # the E18 runner loads E13 for its GPU guard
        inner = module.load
        def load13(n, p):
            m = inner(n, p)
            if n == 'e18_e13':
                m.gpu_guard = fake_guard
            return m
        module.load = load13
    return module


runner.load = patched_load


def injected_build(real_build):
    """R34 fault injection (B3): a subclass of the runner's Run that raises inside the released update or in scoring for the INJECT arms."""
    def build(c):
        out = real_build(c); Run = out[-1]
        class Injected(Run):
            def ftbc_batches_ann(self, arm, T, batches, *a, **k):
                if arm not in INJECT or MODE == 'score-assert':
                    return super().ftbc_batches_ann(arm, T, batches, *a, **k)
                real_step, calls = self.inherited.bias_corr_step_by_step, [0]
                def step(ann, module_ann, snn, module_snn, *aa, **kk):
                    calls[0] += 1
                    if calls[0] < 3:
                        return real_step(ann, module_ann, snn, module_snn, *aa, **kk)
                    def fail(*_):                      # raised inside the SNN forward, after the released code registered its saver hook
                        if MODE == 'oom' and arm not in LIB_ARMS:
                            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 1.00 GiB (injected by the E23 CPU dry run)')
                        if MODE == 'oom':
                            raise RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR (injected by the E23 CPU dry run; 0.5 GiB free)')
                        raise RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR (injected by the E23 CPU dry run; no allocator evidence)')
                    h = module_snn.register_forward_pre_hook(fail)
                    try:
                        return real_step(ann, module_ann, snn, module_snn, *aa, **kk)
                    finally:
                        h.remove()
                self.inherited.bias_corr_step_by_step = step
                try:
                    return super().ftbc_batches_ann(arm, T, batches, *a, **k)
                finally:
                    self.inherited.bias_corr_step_by_step = real_step
            def score(self, arm, T, biases, detail):
                if MODE == 'score-assert' and arm in INJECT:
                    raise AssertionError(f'{arm}: strict reload differs (injected by the E23 CPU dry run)')
                return super().score(arm, T, biases, detail)
        return (*out[:-1], Injected)
    return build


if INJECT:
    runner.build = injected_build(runner.build)


def stamp(t):
    return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(t)) + f'.{int((t % 1) * 1000):03d}'


def synthesize_samplers(slurm, job, pid, cost, gap_arm, foreign_arm, uuid):
    """B4: one runner row per second over the run, none inside the middle half of gap_arm, one foreign PID at the middle of foreign_arm; gpumem
    every 0.5 s with the seven fields of e23_job.py."""
    win = accept_module.windows(cost); s0 = min(a for a, _ in win.values()) - 5; e0 = max(b for _, b in win.values()) + 5
    gs, ge = win[gap_arm]; gs, ge = gs + 0.25 * (ge - gs), gs + 0.75 * (ge - gs); assert ge - gs > accept_module.GAP + 1, (gap_arm, ge - gs)
    fs, fe = win[foreign_arm]; apps, dev, t = [], [], s0
    while t <= e0:
        if not gs < t < ge:
            apps.append(f'{stamp(t)}, {uuid}, {pid}, 2274')
        dev.append(f'{stamp(t)}, {uuid}, 2297, 81559, 1980, 205.31, 58'); dev.append(f'{stamp(t + 0.5)}, {uuid}, 2297, 81559, 1980, 205.31, 58'); t += 1.0
    apps.append(f'{stamp((fs + fe) / 2)}, {uuid}, 4242424, 512'); apps.sort()
    (slurm / f'gpuapps-{job}.csv').write_text('\n'.join(apps) + '\n' + '2026/10/11 0'); (slurm / f'gpumem-{job}.csv').write_text('\n'.join(dev) + '\n')
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'tag': TAG, 'model': MODEL, 'mode': MODE, 'injected_arms': INJECT, 'python': sys.version, 'torch': torch.__version__}
try:
    base = json.loads((E23 / f'config/{MODEL}-sanity.json').read_text())
    if MODEL == 'vgg':
        base = dict(base, draw_n=8, small_n=6, ftbc_batch=2, eccv_batches=2, eccv_n=4, fresh_count=8, dev_count=4, solver_batch=4, num_workers=0, sizes=[2, 6])
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    chain_files = [Path(base[k]) for k in (['e17_runner', 'e16_runner', 'e15_runner', 'e14_runner', 'e13_runner'] if MODEL == 'r34' else ['e18_runner', 'e13_runner', 'step_solver', 'serial_solver'])]
    # The protocol text is left out of the dry freeze, so a wording change does not invalidate a running dry run; the real freeze, which the
    # real acceptance checks, includes it.
    files = [E23 / 'source' / f for f in [f'e23_{MODEL}_runner.py', f'e23_{MODEL}_accept.py', 'e23_r34_accept.py', 'e23_timing.py']] + chain_files
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E23 / 'preservation-before.json', fake / 'preservation-before.json')
    accept.E = fake
    c = dict(base, output=str(SCRATCH / f'{MODEL}-sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time(); failure = None
    with open(fake / f'slurm/{MODEL}-sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        try:
            print(json.dumps(runner.run(c)))
        except Exception as exc:
            if MODE not in ('score-assert', 'update-error'):
                raise
            failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    report['runner_seconds'] = time.time() - t0
    if MODE in ('score-assert', 'update-error'):          # the runner must fail with the injected error, not record it
        want = AssertionError if MODE == 'score-assert' else RuntimeError
        assert failure and failure['type'] == want.__name__ and 'injected by the E23 CPU dry run' in failure['message'], failure
        assert MODE != 'update-error' or 'CUDNN_STATUS_INTERNAL_ERROR' in failure['message'], failure      # a signature without evidence is re-raised
        assert not (Path(c['output']) / 'result.json').exists()
        cost = json.loads((Path(c['output']) / 'cost.json').read_text()); assert not any(v.get('status', 'completed') != 'completed' for v in cost.values()), cost
        report.update(passed=True, expected_failure=failure, recorded_arms=list(cost))
    else:
        cost = json.loads((Path(c['output']) / 'cost.json').read_text()); arms = list(cost); win = accept_module.windows(cost)
        span = lambda a: win[a][1] - win[a][0]                 # the longest arms, so the synthesized gap and the foreign row stay inside one window
        foreign_arm = max((a for a in arms if a.startswith('P0')), key=span)
        gap_arm = max((a for a in arms if a.startswith('F0') and cost[a].get('status', 'completed') == 'completed'), key=span)
        synthesize_samplers(fake / 'slurm', '999999', os.getpid(), cost, gap_arm, foreign_arm, c['allowed_uuids'][0])
        with open(SCRATCH / f'accept-{MODEL}-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
            accept.check('sanity')
        acc = json.loads((fake / f'{MODEL}-sanity-acceptance.json').read_text()); ex = acc['exclusivity']
        assert ex['arms_unknown'] == [gap_arm] and ex['arms_possibly_affected'] == [foreign_arm] and ex['other_compute_pids'] == [4242424], ex
        assert all(v['reading'] == 'no_other_process_seen' for a, v in ex['arms'].items() if a not in (gap_arm, foreign_arm)), ex
        if MODE == 'oom':
            assert sorted(acc['not_completed']) == sorted(INJECT), acc['not_completed']
            assert all(v['status'] == 'out_of_memory' and v['stage'] == 'update' and v['leftover_hooks_dropped'] >= 1 for v in acc['not_completed'].values()), acc['not_completed']
            assert {a: v['oom_basis'] for a, v in acc['not_completed'].items()} == {a: 'library' if a in LIB_ARMS else 'exception' for a in INJECT}, acc['not_completed']
            assert all(cost[a]['status'] == 'completed' for a in arms if a.startswith('F0rel') and a not in INJECT)
        report.update(passed=acc['passed'], sizes=c['sizes'], arms=sorted(acc['cost_table']), repeat_bitwise_equal=acc['repeat_bitwise_equal'],
                      not_completed=acc['not_completed'], exclusivity_readings={a: v['reading'] for a, v in ex['arms'].items()})
        shutil.copy(fake / f'{MODEL}-sanity-acceptance.json', SCRATCH / f'cpu-{MODEL}-sanity-acceptance.json')
    code = [E23 / 'source' / f for f in [f'e23_{MODEL}_runner.py', f'e23_{MODEL}_accept.py', 'e23_r34_accept.py', 'e23_timing.py']] + \
           [E23 / 'config' / f for f in [f'{MODEL}-sanity.json', f'{MODEL}-main.json']] + [Path(__file__).resolve()]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU logic dry run at reduced sizes; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['tag', 'passed', 'cuda_initialized', 'runner_seconds', 'repeat_bitwise_equal', 'not_completed']}))
    if 'traceback' in report: print(report['traceback'])
