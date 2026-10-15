"""CPU-only logic dry run of one E29 child (runner + sanity acceptance) at sanity sizes (EXPERIMENT_PLAN.md section 6; adapted from the E23 dry
run). Never initializes CUDA. Usage: cpu_dryrun.py <scratch_dir> {r34,r34-oom,r34-libnoev,r34-runtime,vgg}.
R34: the evaluation-transform images and the release batches (after the release transform) are downsampled to 32x32 (the E19/E23 dry
runs downsampled the former), so the R34 runs check their batches against the hash of the same rows built by replica_batches on the same
downsampled transform, not against E23's record. VGG-16: smaller E18 sizes at 224x224 (the classifier needs 7x7 features, as the E18/E23 dry
runs) and real release batches, checked against E23's sanity record. Patches only the GPU guard, CUDA calls and (R34) the image size, along
each child's runner chain. Memory readings are fixed CPU stand-ins, so they are meaningless; the code path that records them is what is tested.
Fault injection (initial review B4), R34 only: inside the SNN forward of the third update call of an arm, while the released saver hook is
registered (in PAIR during the CHB capture, after REL ran in the same call):
  r34-oom      REL-T8 raises torch.cuda.OutOfMemoryError; PAIR-T4 raises a RuntimeError with the pre-registered signature
               CUDNN_STATUS_INTERNAL_ERROR while the device reports 0.5 GiB free; CHS-T8 raises OutOfMemoryError. The run must continue: those
               three arms recorded out_of_memory at the update stage (bases exception, library, exception) with the hook dropped and the export
               restored, PAIR-T8 not_run, every other arm completed; the acceptance valid, T2 read pass and T4, T8 not decided, the gate held, CHS feasible at T2 and T4.
  r34-libnoev  REL-T2 raises the same signature without allocator evidence (40 GiB free, num_ooms unchanged): the runner fails with it and
               leaves failure.json, and no result.json.
  r34-runtime  CHS-T2 raises an unqualified RuntimeError: the runner fails with it and leaves failure.json, and no result.json.
Follow-up 1 R1, in the r34 run only: after its acceptance passes, the acceptance is run again on four tampered copies of the run's output,
and each must give the reading of section 4: PAIR-T4's corrections hash changed without allocator evidence -> invalid; the same with
num_ooms_increase = 1 in REL-T4 -> T4 not decided, the gate holds on T2 and T8; two PAIR-T4 calls whose CHB fingerprints differ, with
allocator evidence and a CHB-REL difference of 0.01 theta -> T4 not decided, the gate holds; the same with 0.2 theta -> T4 fails and the gate
does not hold."""
import contextlib, hashlib, io, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve(); TAG = sys.argv[2]; assert TAG in ('r34', 'r34-oom', 'r34-libnoev', 'r34-runtime', 'vgg')
MODEL = TAG.split('-')[0]
INJECT = {'r34-oom': {'REL-T8': 'oom', 'PAIR-T4': 'library', 'CHS-T8': 'oom'}, 'r34-libnoev': {'REL-T2': 'library'},
          'r34-runtime': {'CHS-T2': 'runtime'}}.get(TAG, {})
E29 = Path(__file__).resolve().parents[1]
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'     # as e29_job.py sets it; the acceptance asserts the recorded value
import torch, torch.nn.functional as F
REAL_IS_INITIALIZED = torch.cuda.is_initialized
torch.cuda.is_available = lambda: False
torch.cuda.synchronize = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 3 << 30
torch.cuda.max_memory_reserved = lambda *a, **k: 4 << 30
torch.cuda.memory_allocated = lambda *a, **k: 1 << 30
torch.cuda.memory_reserved = lambda *a, **k: 2 << 30
torch.cuda.mem_get_info = lambda *a, **k: ((512 << 20) if TAG == 'r34-oom' else (40 << 30), 80 << 30)   # read only for a library signature
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.empty_cache = lambda *a, **k: None
torch.cuda.manual_seed_all = lambda *a, **k: None
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
if MODEL == 'vgg':                                     # as the E23 dry run: VGG-16 runs with num_workers=0 (no DataLoader fork), so its thread count can be fixed;
    torch.set_num_threads(int(os.environ.get('E29_DRYRUN_THREADS', '12')))   # R34 keeps the runner's own setting, because forked loader workers
    torch.set_num_threads = lambda n: None             # deadlock after an override (memory note on CPU dry runs)
sys.path.insert(0, str(E29 / 'source'))
runner = __import__('e29_runner'); accept = __import__('e29_accept')
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


def small_release(module):
    """R34 only: the E15 AugmentedRows images are downsampled to 32x32 after the release transform, so the fifteen release-default fits stay
    short on the CPU (the code path does not depend on the image size)."""
    real = module.AugmentedRows.__getitem__
    def getitem(self, i):
        image, label, index = real(self, i)
        return small(image[None])[0], label, index
    module.AugmentedRows.__getitem__ = getitem


def chain(module, names):
    """R34: patch each runner's own loader along E17 -> E16 -> E15 -> E14 -> E13 so E13's globals get the fakes (as the E19/E23 dry runs)."""
    inner = module.load
    def load_next(n, p):
        m = inner(n, p)
        if n == names[0]:
            if n == 'e16_e15':
                small_release(m)
            if len(names) == 1:
                m.gpu_guard = fake_guard; m.load = small_images(m.load)
            else:
                chain(m, names[1:])
        return m
    module.load = load_next


def injecting_fit(real_fit, released):
    """B4: release_fit with a fault raised inside the SNN forward of the third update call of the INJECT arms."""
    def fit(inherited, ann, snn, T, batches, alpha, step, interfaces):
        mode = 'PAIR' if hasattr(step, 'paired') else 'CHS' if hasattr(step, 'chunk') else 'REL'
        assert mode != 'REL' or step is released
        kind = INJECT.get(f'{mode}-T{T}')
        if kind is None:
            return real_fit(inherited, ann, snn, T, batches, alpha, step, interfaces)
        calls = [0]
        def injected(ann_, module_ann, snn_, module_snn, *a, **k):
            calls[0] += 1
            if calls[0] != 3:
                return step(ann_, module_ann, snn_, module_snn, *a, **k)
            seen = [0]
            def fail(*_):                               # raised inside the SNN forward, after the released code registered its saver hook
                seen[0] += 1
                if seen[0] < (2 if mode == 'PAIR' else 1):
                    return None
                if kind == 'oom':
                    raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 1.00 GiB (injected by the E29 CPU dry run)')
                if kind == 'library':
                    raise RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR (injected by the E29 CPU dry run)')
                raise RuntimeError('an unqualified error (injected by the E29 CPU dry run)')
            h = module_snn.register_forward_pre_hook(fail)
            try:
                return step(ann_, module_ann, snn_, module_snn, *a, **k)
            finally:
                h.remove()
        return real_fit(inherited, ann, snn, T, batches, alpha, injected, interfaces)
    return fit


def r1_cases(c, fake, out):
    """Follow-up 1 R1: the acceptance on tampered copies of this run's output (see the module docstring); returns the readings."""
    results = {}
    for case in ['trajectory-no-evidence', 'trajectory-evidence', 'fingerprints-evidence', 'fingerprints-gross']:
        o2 = SCRATCH / f'tampered-{case}'; shutil.copytree(out, o2)
        rec = json.loads((o2 / 'e29.json').read_text())
        if case.startswith('trajectory'):
            rec['arms']['PAIR-T4']['corrections_sha256'] = '0' * 64
            if case == 'trajectory-evidence':
                rec['arms']['REL-T4']['num_ooms_increase'] = 1
        else:
            log = json.loads((o2 / 'pair-T4.json').read_text())
            for r in log[:2]:
                r['fingerprints']['CHB'][1][0][2] += 1; r['num_ooms']['CHB'] = [0, 1]
                r['compare']['CHB_vs_REL']['max'] = 0.01 if case == 'fingerprints-evidence' else 0.2
            (o2 / 'pair-T4.json').write_text(json.dumps(log))
        (o2 / 'e29.json').write_text(json.dumps(rec))
        (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(dict(c, output=str(o2)), indent=2))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                accept.check(MODEL, 'sanity')
            exited = None
        except SystemExit as exc:
            exited = str(exc)
        acc = json.loads((fake / f'{MODEL}-sanity-acceptance.json').read_text())
        results[case] = {'valid': acc['valid'], 'gate': acc.get('gate'), 'T_readings': acc['T_readings'], 'any_T_fail': acc['any_T_fail'], 'exited': exited}
    (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(c, indent=2))
    nd = {'T2': 'pass', 'T4': 'not decided', 'T8': 'pass'}
    assert not results['trajectory-no-evidence']['valid'] and 'invalid' in results['trajectory-no-evidence']['exited'], results
    assert all(results[k]['valid'] and results[k]['gate'] and results[k]['T_readings'] == nd and results[k]['exited'] is None
               for k in ['trajectory-evidence', 'fingerprints-evidence']), results
    v = results['fingerprints-gross']; assert v['valid'] and not v['gate'] and v['T_readings']['T4'] == 'fail' and 'gate' in v['exited'], results
    return results


captured = {}
def patched_load(name, path):
    module = original_load(name, path)
    if name == 'e29_e23_r34':                          # its build loads E17 as 'e23_e17'
        inner = module.load
        def load17(n, p):
            m = inner(n, p)
            if n == 'e23_e17':
                chain(m, ['e17_e16', 'e16_e15', 'e15_e14', 'e14_e13'])
            return m
        module.load = load17
    if name == 'e29_e18_runner':                       # the E18 runner loads E13 for its GPU guard
        inner18 = module.load
        def load13(n, p):
            m = inner18(n, p)
            if n == 'e18_e13':
                m.gpu_guard = fake_guard
            return m
        module.load = load13
    if name == 'e29_release_chunked':
        captured['RC'] = module
    return module


runner.load = patched_load
real_build = runner.build
def build(c):
    out = real_build(c); captured.update(r=out[1], rows=out[2])
    if INJECT:
        captured['RC'].release_fit = injecting_fit(captured['RC'].release_fit, out[1].inherited.bias_corr_step_by_step)
    return out
runner.build = build
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'tag': TAG, 'model': MODEL, 'injected': INJECT, 'python': sys.version, 'torch': torch.__version__}
try:
    base = json.loads((E29 / f'config/{MODEL}-sanity.json').read_text())
    if MODEL == 'vgg':
        base = dict(base, draw_n=8, small_n=6, ftbc_batch=2, eccv_batches=2, eccv_n=4, fresh_count=8, dev_count=4, solver_batch=4, num_workers=0, sizes=[2, 6])
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True)
    chain_keys = ['e23_r34_runner', 'e17_runner', 'e16_runner', 'e15_runner', 'e14_runner', 'e13_runner'] if MODEL == 'r34' else \
                 ['e23_r34_runner', 'e18_runner', 'e15_runner', 'e13_runner', 'step_solver', 'serial_solver']
    files = [E29 / 'source' / f for f in ['e29_runner.py', 'e29_accept.py', 'release_chunked.py']] + [Path(base[k]) for k in chain_keys]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    accept.E = fake
    c = dict(base, output=str(SCRATCH / f'{MODEL}-sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time(); failure = None
    with open(SCRATCH / f'{MODEL}-sanity-runner.out', 'w') as fh, contextlib.redirect_stdout(fh):
        try:
            print(json.dumps(runner.main(c)))
        except Exception as exc:
            if TAG not in ('r34-libnoev', 'r34-runtime'):
                raise
            failure = {'type': type(exc).__name__, 'message': str(exc)}
    report['runner_seconds'] = time.time() - t0
    out = Path(c['output'])
    if TAG in ('r34-libnoev', 'r34-runtime'):               # the runner must fail with the injected error and leave failure.json
        assert failure and failure['type'] == 'RuntimeError' and 'injected by the E29 CPU dry run' in failure['message'], failure
        assert TAG != 'r34-libnoev' or 'CUDNN_STATUS_INTERNAL_ERROR' in failure['message'], failure   # a signature without evidence is re-raised
        written = json.loads((out / 'failure.json').read_text())
        assert written['type'] == 'RuntimeError' and 'injected by the E29 CPU dry run' in written['message'] and not (out / 'result.json').exists()
        rec = json.loads((out / 'e29.json').read_text()) if (out / 'e29.json').exists() else {'arms': {}}
        assert all(v['status'] == 'completed' for v in rec['arms'].values()), rec['arms']
        report.update(passed=True, expected_failure=failure, recorded_arms={a: v['status'] for a, v in rec['arms'].items()})
    else:
        if MODEL == 'r34':                                   # downsampled release batches: their hash from replica_batches on the same rows and transform
            r = captured['r']
            rb, _ = runner.replica_batches(sys.modules['e16_e15'], captured['rows'], r.e0.sha256, c['release_batch'], c['release_n'], c['release_seed'])
            c['e23_release_inputs_sha256'] = runner.tensor_hash([b[0] for b in rb]); report['r34_expected_inputs_from_replica'] = c['e23_release_inputs_sha256']
            (fake / f'config/{MODEL}-sanity.json').write_text(json.dumps(c, indent=2))
        with open(SCRATCH / f'accept-{MODEL}-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
            accept.check(MODEL, 'sanity')
        acc = json.loads((fake / f'{MODEL}-sanity-acceptance.json').read_text())
        rec = json.loads((out / 'e29.json').read_text()); status = {a: v['status'] for a, v in rec['arms'].items()}
        if TAG == 'r34-oom':
            want = {a: 'completed' for a in status}; want.update({'REL-T8': 'out_of_memory', 'PAIR-T4': 'out_of_memory', 'CHS-T8': 'out_of_memory', 'PAIR-T8': 'not_run'})
            assert status == want, status
            bases = {a: rec['arms'][a]['oom_basis'] for a in ['REL-T8', 'PAIR-T4', 'CHS-T8']}
            assert bases == {'REL-T8': 'exception', 'PAIR-T4': 'library', 'CHS-T8': 'exception'}, bases
            assert all(rec['arms'][a]['leftover_hooks_dropped'] >= 1 and rec['arms'][a]['export_restored'] and rec['arms'][a]['stage'] == 'update'
                       for a in bases), {a: rec['arms'][a] for a in bases}
            assert rec['arms']['PAIR-T4']['paired_calls_completed'] == 2
            assert acc['valid'] and acc['gate'] and acc['T_readings'] == {'T2': 'pass', 'T4': 'not decided', 'T8': 'not decided'}, acc
            assert acc['feasible_CHS'] == {'T2': True, 'T4': True, 'T8': False} and not acc['any_T_fail'], acc
        else:
            assert all(s == 'completed' for s in status.values()), status
            assert acc['valid'] and acc['gate'] and acc['T_readings'] == {'T2': 'pass', 'T4': 'pass', 'T8': 'pass'} and all(acc['feasible_CHS'].values()), acc
        shutil.copy(fake / f'{MODEL}-sanity-acceptance.json', SCRATCH / f'cpu-{TAG}-sanity-acceptance.json')   # the run's own acceptance, before any tampering
        if TAG == 'r34':
            report['r1_tampered'] = r1_cases(c, fake, out)
        report.update(passed=acc['valid'] and acc['gate'], arms=status, inputs_equal=acc['inputs_equal_e23_release_default'],
                      paired={T: {k: v['paired'].get(k) for k in ['reading', 'reason', 'P1_failed_calls', 'P1_max_CHB_vs_REL_fingerprints_equal',
                                                                  'P2_max_CHS_vs_CHB_first_interface', 'P3_max_CHS_vs_REL', 'fingerprints_differ_calls',
                                                                  'trajectory']} for T, v in acc['per_T'].items()})
    code = [E29 / 'source' / f for f in ['e29_runner.py', 'e29_accept.py', 'release_chunked.py']] + \
           [E29 / 'config' / f for f in [f'{MODEL}-sanity.json', f'{MODEL}-main.json']] + [Path(__file__).resolve()]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU logic dry run at sanity sizes; memory readings are stand-ins and the numbers are meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['tag', 'passed', 'cuda_initialized', 'runner_seconds', 'arms']}))
    if 'traceback' in report: print(report['traceback'])
