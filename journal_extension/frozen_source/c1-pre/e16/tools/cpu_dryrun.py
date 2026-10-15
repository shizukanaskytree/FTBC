"""CPU-only logic dry run of the E16 runner + acceptance for every stage combination at sanity sizes (EXPERIMENT_PLAN.md section 7):
stages [main, release], [main], [release], plus [release] with an injected out-of-memory error in the middle of the Q2 release recipe
whose ANN is the spatial twin (after the released GetLayerInputOutput hook was registered on the proxy), images downsampled to 32x32.
Also: ftbc_batches_ann with the teacher as ANN reproduces the frozen E15 ftbc_batches bitwise, and with the twin as ANN differs.
Never initializes CUDA."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E15 = Path(__file__).resolve().parents[1]   # E16 directory (variable name kept from the E15 template)
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
import e16_runner as e15_runner, e16_accept as e15_accept
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
    if name == 'e16_e15':                       # E15 -> E14 -> E13 chain: patch each module's own loader so E13's globals get the fakes
        inner15 = module.load
        def load14(n, p):
            m14 = inner15(n, p)
            if n == 'e15_e14':
                inner14 = m14.load
                def load13(n2, p2):
                    m13 = inner14(n2, p2)
                    if n2 == 'e14_e13':
                        m13.gpu_guard = fake_guard; m13.load = small_images(m13.load)
                    return m13
                m14.load = load13
            return m14
        module.load = load14
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
    orig_ftbc = Run.ftbc_batches_ann
    def ftbc_batches_ann(self, arm, T, batches, alpha, images, recipe, ann, ann_tag, extra=None):
        if extra and extra.get('seed') == FAULT['seed'] and ann_tag == 'twin-spatial':   # OOM on the 5th saved activation (layer 3, the proxy's hook)
            cls = self.inherited.ActivationSaverHookEachTimeStep; real_call = cls.__call__; n = {'k': 0}
            def failing(hook, module, inp, outp):
                n['k'] += 1
                if n['k'] == 5: raise torch.cuda.OutOfMemoryError('CUDA out of memory (injected by the CPU dry run)')
                return real_call(hook, module, inp, outp)
            cls.__call__ = failing
            try:
                return orig_ftbc(self, arm, T, batches, alpha, images, recipe, ann, ann_tag, extra)
            finally:
                cls.__call__ = real_call
        return orig_ftbc(self, arm, T, batches, alpha, images, recipe, ann, ann_tag, extra)
    Run.ftbc_batches_ann = ftbc_batches_ann
    orig_clear = Run.clear_release_hooks
    def clear_release_hooks(self):                  # record where leftover hooks sat before the (unchanged) cleanup runs
        cls = self.inherited.ActivationSaverHookEachTimeStep
        count = lambda models: sum(isinstance(h, cls) for mdl in models for m in mdl.modules() for h in m._forward_hooks.values())
        HOOKS.append({'teacher': count([self.teacher]), 'snn': count([N['snn'] for N in self.nets.values()]),
                      'proxy': count([N['proxy'] for N in list(self.nets.values()) + list(self.grid_proxies.values())])})
        return orig_clear(self)
    Run.clear_release_hooks = clear_release_hooks
    return out
e15_runner.build = build
HOOKS = []
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'configs': {}}
try:
    base = json.loads((E15 / 'config/sanity.json').read_text())
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E15 / 'source' / f for f in ['e16_runner.py', 'e16_accept.py']] + [E15 / 'EXPERIMENT_PLAN.md', Path(base['e15_runner']), Path(base['e14_runner']), Path(base['e13_runner'])]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E15 / 'preservation-before.json', fake / 'preservation-before.json')
    e15_accept.E = fake
    for name, stages in [('sanity', ['main', 'release']), ('main', ['main']), ('release', ['release']), ('release-fault', ['release'])]:
        FAULT['seed'] = base['release_seeds']['Q2'] if name == 'release-fault' else None; HOOKS.clear()
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
            report['configs'][name]['Q3b'] = {k: v for k, v in acc['decisions']['Q3b'].items() if k != 'pairs'}
        report['configs'][name]['hooks_seen_by_cleanup'] = list(HOOKS)
        shutil.copy(fake / f'{name}-acceptance.json', SCRATCH / f'cpu-{name}-acceptance.json')
    code = [E15 / 'source' / f for f in ['e16_runner.py', 'e16_accept.py']] + [E15 / 'config' / f for f in ['sanity.json', 'main.json', 'release.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    flt = report['configs']['release-fault']
    assert flt['release']['Q2-tw']['status'] == 'out_of_memory' and flt['release']['Q2-tw']['leftover_hooks_dropped'] >= 1 and not flt['Q3']['pairs_pass']
    assert all(flt['release'][k]['status'] == 'completed' for k in ['Q1-tw', 'Q1-te', 'Q2-te', 'Q3-tw', 'Q3-te'])
    assert 'F0-sp-rel-tw-Q2' not in flt['arms'] and 'F0-sp-rel-te-Q2' in flt['arms']
    # the leftover hook sat on the proxy (the ANN of the failing arm): the E15 cleanup, which only visited the teacher and SNNs, would have missed it
    # (the runner's end-of-run check calls the cleanup once more and must find nothing)
    zero = {'teacher': 0, 'snn': 0, 'proxy': 0}; hk = flt['hooks_seen_by_cleanup']
    assert len(hk) == 2 and hk[0]['proxy'] >= 1 and hk[0]['teacher'] == 0 and hk[1] == zero, hk
    assert all(v['hooks_seen_by_cleanup'] == [zero] for k, v in report['configs'].items() if k != 'release-fault')
    report['fault_injection'] = {'Q2_tw_status': flt['release']['Q2-tw']['status'], 'leftover_hooks_dropped': flt['release']['Q2-tw']['leftover_hooks_dropped'],
                                 'hooks_by_model_before_cleanup': hk[0], 'Q3_pairs_pass': flt['Q3']['pairs_pass'], 'Q2_te_status': flt['release']['Q2-te']['status']}
    # FTBC equivalence: the explicit-ANN variant with the teacher reproduces the frozen E15 ftbc_batches bitwise; with the twin it differs
    ce = dict(base, stages=['main'], output=str(SCRATCH / 'equivalence-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    *_, RunE = e15_runner.build(ce); r = RunE(ce); r.variant = 'spatial'; rows = r.draws['C1'][:ce['small_n']]
    bt = r.registered_batches(rows); ids = [int(x['index']) for x in rows]
    b15 = r.ftbc_batches('E15-te', 4, bt, ce['ftbc_alpha'], ids, 'registered')
    b16 = r.ftbc_batches_ann('E16-te', 4, bt, ce['ftbc_alpha'], ids, 'registered', r.teacher, 'teacher')
    b16tw = r.ftbc_batches_ann('E16-tw', 4, bt, ce['ftbc_alpha'], ids, 'registered', r.net(4)['proxy'], 'twin-spatial')
    same = all(torch.equal(x, y) for n in b15 for x, y in zip(b15[n], b16[n])); differ = any(not torch.equal(x, y) for n in b15 for x, y in zip(b15[n], b16tw[n]))
    assert same and differ and r.ftbc['E15-te']['inputs_sha256'] == r.ftbc['E16-te']['inputs_sha256'] and r.ftbc['E16-tw']['ann_digest'] != r.ftbc['E16-te']['ann_digest']
    report['ftbc_equivalence'] = {'teacher_ann_equals_E15_bitwise': same, 'twin_ann_differs': differ, 'layers': len(b15)}
    o = SCRATCH / 'main-cpu'; tp = torch.load(o / 'targets-C1-twin-plain.pt', weights_only=False)['means']; tsp = torch.load(o / 'targets-C1-twin-spatial.pt', weights_only=False)['means']
    report['twin_targets_differ_plain_vs_spatial'] = any(not torch.equal(tp[n], tsp[n]) for n in tp); assert report['twin_targets_differ_plain_vs_spatial']
    a8 = torch.load(o / 'targets-C1-a8-spatial.pt', weights_only=False); te = torch.load(o / 'targets-C1-teacher.pt', weights_only=False)
    report['target_levels_and_digests'] = {'a8_level': a8['level'], 'teacher_level': te['level'], 'a8_digest_equals_twin_spatial': a8['network_digest'] == torch.load(o / 'targets-C1-twin-spatial.pt', weights_only=False)['network_digest']}
    assert report['target_levels_and_digests'] == {'a8_level': [8], 'teacher_level': [8], 'a8_digest_equals_twin_spatial': True}
    report['passed'] = all(v['passed'] for v in report['configs'].values()) and len(report['configs']) == 4
    report['note'] = 'CPU 32x32 logic dry run at sanity sizes for each stage combination; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized']} | {'configs': {k: v['passed'] for k, v in report['configs'].items()}}))
    if 'traceback' in report: print(report['traceback'])
