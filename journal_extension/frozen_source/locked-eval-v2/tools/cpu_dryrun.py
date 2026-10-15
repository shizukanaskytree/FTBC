"""CPU-only logic dry run of one locked-evaluation child (runner + sanity acceptance) at sanity sizes, with training-image stand-ins for the
evaluation role (no evaluation image is read). Never initializes CUDA. Usage: cpu_dryrun.py <scratch_dir> <run tag>: a job tag (r34-t8,
vgg-t8, c10-vgg16, c100-resnet20) optionally with a variant suffix (-oom, -original, -skip for ImageNet; -test for CIFAR). Adapted from the
E29 dry run: the GPU guard and CUDA calls are patched along each chain; R34 images (evaluation transform and release batches) are
downsampled to 32x32; VGG-16 keeps 224x224 with smaller E18 sizes; CIFAR needs no resize. Every config carries release_mode 'skip'
(plan candidate 4: the release-default arm FTBC-def is not run); the -oom and -original variants are kept only as optional checks of that
unused path.
c10-vgg16-test (initial review N1): the main-phase test path (cifar_test.init_test with its whole-file hash, TestRows, evaluate,
calibration_overlap) on the CIFAR-10 training file data_batch_1 read in place (10,000 rows, the test-file format), its inventory hash in
place of the test file's; the store is cut to 64 rows after the full read: rows 0-61 and two rows of that file among the calibration rows
the run uses (the first small_n of the pools), so the expected calibration overlap is known.
r34-t8-oom (fault injection, as E29 initial review B4): the FTBC-def fit of draw C2 raises torch.cuda.OutOfMemoryError inside the SNN forward
of its third update call, while the released saver hook is registered; the run must record that draw as out_of_memory with the hook dropped
and the export restored, finish every other arm, and the acceptance must leave FTBC-def out of the cell with that status."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve(); RUN_TAG = sys.argv[2]
VARIANT = next((v for v in ('oom', 'original', 'skip', 'test') if RUN_TAG.endswith('-' + v)), None)
TAG = RUN_TAG[:-len(VARIANT) - 1] if VARIANT else RUN_TAG; INJECT = VARIANT == 'oom'
assert TAG in ('r34-t8', 'vgg-t8', 'c10-vgg16', 'c100-resnet20') and (VARIANT != 'test' or TAG == 'c10-vgg16'), RUN_TAG
L1 = Path(__file__).resolve().parents[1]
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, torch.nn.functional as F
REAL_IS_INITIALIZED = torch.cuda.is_initialized
for name, value in [('is_available', lambda: False), ('synchronize', lambda *a, **k: None), ('max_memory_allocated', lambda *a, **k: 3 << 30),
                    ('max_memory_reserved', lambda *a, **k: 4 << 30), ('memory_allocated', lambda *a, **k: 1 << 30), ('memory_reserved', lambda *a, **k: 2 << 30),
                    ('mem_get_info', lambda *a, **k: (40 << 30, 80 << 30)), ('set_device', lambda *a, **k: None), ('empty_cache', lambda *a, **k: None),
                    ('manual_seed_all', lambda *a, **k: None), ('reset_peak_memory_stats', lambda *a, **k: None),
                    ('set_per_process_memory_fraction', lambda *a, **k: None)]:
    setattr(torch.cuda, name, value)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
CIFAR = TAG.startswith('c1')
MODEL = 'cifar' if CIFAR else TAG.split('-')[0]
if MODEL in ('vgg', 'cifar'):                          # no DataLoader fork in these chains (num_workers 0), so the thread count can be fixed
    torch.set_num_threads(int(os.environ.get('L1_DRYRUN_THREADS', '12'))); torch.set_num_threads = lambda n: None
sys.path.insert(0, str(L1 / 'source'))
runner = __import__('l1_cifar_runner' if CIFAR else 'l1_imagenet_runner'); accept = __import__('l1_accept')
original_load = runner.load
captured, TEST_EXTRA = {}, []


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
    real = module.AugmentedRows.__getitem__
    def getitem(self, i):
        image, label, index = real(self, i)
        return small(image[None])[0], label, index
    module.AugmentedRows.__getitem__ = getitem


def chain(module, names):
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


def patched_load(name, path):
    module = original_load(name, path)
    if name == 'l1_e23_r34':
        inner = module.load
        def load17(n, p):
            m = inner(n, p)
            if n == 'e23_e17':
                chain(m, ['e17_e16', 'e16_e15', 'e15_e14', 'e14_e13'])
            return m
        module.load = load17
    if name == 'l1_e18_runner':
        inner18 = module.load
        def load13(n, p):
            m = inner18(n, p)
            if n == 'e18_e13':
                m.gpu_guard = fake_guard
            return m
        module.load = load13
    if name == 'l1_release_chunked' and INJECT:
        real_fit, fits = module.release_fit, [0]
        def fit(inherited, ann, snn, T, batches, alpha, step, interfaces):
            fits[0] += 1
            if fits[0] != 2:                                   # the second FTBC-def fit of the run is draw C2
                return real_fit(inherited, ann, snn, T, batches, alpha, step, interfaces)
            calls = [0]
            def injected(ann_, module_ann, snn_, module_snn, *a, **k):
                calls[0] += 1
                if calls[0] != 3:
                    return step(ann_, module_ann, snn_, module_snn, *a, **k)
                def fail(*_):
                    raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 1.00 GiB (injected by the locked-evaluation CPU dry run)')
                h = module_snn.register_forward_pre_hook(fail)
                try:
                    return step(ann_, module_ann, snn_, module_snn, *a, **k)
                finally:
                    h.remove()
            return real_fit(inherited, ann, snn, T, batches, alpha, injected, interfaces)
        module.release_fit = fit
    if name == 'l1_cifar_test' and VARIANT == 'test':
        real_init = module.init_test
        def init_test(CE, dataset, test_file, expected_sha256):
            store = real_init(CE, dataset, test_file, expected_sha256)            # the full read and the whole-file hash check
            pick = list(range(62)) + TEST_EXTRA; captured['test_pick'] = pick
            data = store['data'][pick]; labels = [store['labels'][i] for i in pick]
            rows = [{'index': j, 'label': labels[j], 'sha256': store['rows'][i]['sha256']} for j, i in enumerate(pick)]
            return dict(store, data=data, labels=labels, rows=rows)
        module.init_test = init_test
    if name == 'l1_e25_runner':                       # E25 build loads E18 (as e25_e18_runner) and its constructor loads E13 (as e25_e13)
        inner25 = module.load
        def load25(n, p):
            m = inner25(n, p)
            if n == 'e25_e13':
                m.gpu_guard = fake_guard
            return m
        module.load = load25
    return module


runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time(), 'tag': TAG, 'python': sys.version, 'torch': torch.__version__}
try:
    base = json.loads((L1 / f'config/{TAG}-sanity.json').read_text())
    if MODEL == 'vgg':
        base = dict(base, draw_n=8, small_n=6, tiny_n=2, ftbc_batch=2, eccv_batches=2, eccv_n=4, fresh_count=8, dev_count=4, solver_batch=4, num_workers=0,
                    sizes=[2, 6], release_batch=4, release_chunk=1, evaluation_n=16, evaluation_workers=0)   # no loader fork after the thread override
    if not CIFAR and VARIANT in ('oom', 'original'):                      # optional variants of the release-default arm (not run by the plan)
        base = dict(base, release_mode={'oom': 'chunked', 'original': 'original'}[VARIANT])
    if VARIANT == 'test':                                                 # data_batch_1 in place of the test pickle (same format, 10,000 rows)
        prep = json.loads((L1 / 'input/preparation.json').read_text())['cifar10']
        train1 = Path(prep['test_file']).parent / 'data_batch_1'
        pool = json.loads((L1 / 'input/cifar10-calibration.json').read_text())['C1']
        pools_all = json.loads((L1 / 'input/cifar10-calibration.json').read_text())
        TEST_EXTRA = sorted({int(r['index']) for d in ['C1', 'C2', 'C3'] for r in pools_all[d][:base['small_n']] if 62 <= int(r['index']) < 10000})[:2]
        assert len(TEST_EXTRA) == 2, TEST_EXTRA
        base = dict(base, evaluation='test', test_file=str(train1), test_file_sha256=prep['train_files']['cifar-10-batches-py/data_batch_1']['sha256'],
                    evaluation_n=64)
    fake = SCRATCH / 'fakeL1'; (fake / 'config').mkdir(parents=True)
    keys = [k for k in ['e23_r34_runner', 'e17_runner', 'e16_runner', 'e15_runner', 'e14_runner', 'e13_runner', 'e18_runner', 'e25_runner', 'cifar_e0',
                        'cifar_test', 'release_chunked', 'step_solver', 'serial_solver'] if base.get(k)]
    files = [L1 / 'source' / f for f in ['l1_imagenet_runner.py', 'l1_cifar_runner.py', 'l1_accept.py', 'cifar_test.py']] + [Path(base[k]) for k in keys]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    accept.E = fake
    c = dict(base, output=str(SCRATCH / f'{TAG}-sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / f'config/{TAG}-sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(SCRATCH / f'{TAG}-sanity-runner.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(runner.run(c)))
    report['runner_seconds'] = time.time() - t0
    with open(SCRATCH / f'accept-{TAG}-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        accept.check(TAG, 'sanity')
    acc = json.loads((fake / f'{TAG}-sanity-acceptance.json').read_text())
    l1rec = json.loads((Path(c['output']) / 'l1.json').read_text())
    if VARIANT == 'original':
        assert all(l1rec['fits'][f'FTBCdef-T8-{d}']['mode'] == 'original' and l1rec['fits'][f'FTBCdef-T8-{d}']['chunk'] is None for d in ['C1', 'C2', 'C3'])
        assert acc['per_T']['T8']['FTBCdef']['status'] == 'run' and 'FTBCdef' in acc['per_T']['T8']['ftbc_arms'], acc['per_T']['T8']['FTBCdef']
    if c['release_mode'] == 'skip':                                       # plan candidate 4: no release-default fit, not in the comparison
        assert not any(k.startswith('FTBCdef') for k in l1rec['fits']), sorted(l1rec['fits'])
        assert all('FTBCdef' not in v['ftbc_arms'] and v['FTBCdef']['status'].startswith('skip') for v in acc['per_T'].values()), acc['per_T']
    if VARIANT == 'test':                                                 # the overlap list is exactly the pool rows that were cut into the store
        pools = json.loads((L1 / 'input/cifar10-calibration.json').read_text()); pick = captured['test_pick']
        n_used = c['small_n']                                             # the runner checks the rows it calibrates with: the first small_n of each pool
        want = sorted((d, int(r['index']), pick.index(int(r['index']))) for d, rows in pools.items() for r in rows[:n_used] if int(r['index']) in pick)
        got = sorted((v['draw'], v['train_index'], v['test_index']) for v in l1rec['evaluation']['calibration_test_overlap'])
        assert got == want and len(want) >= 2 and l1rec['evaluation']['kind'] == 'test' and l1rec['evaluation']['n'] == 64, (got, want)
        report['test_path'] = {'overlap': got, 'rows': 64, 'file': c['test_file']}
    if INJECT:
        fits = l1rec['fits']; v = fits['FTBCdef-T8-C2']
        assert v['status'] == 'out_of_memory' and v['oom_basis'] == 'exception' and v['leftover_hooks_dropped'] >= 1 and v['export_restored'], v
        assert 'status' not in fits['FTBCdef-T8-C1'] and 'status' not in fits['FTBCdef-T8-C3'], (fits['FTBCdef-T8-C1'], fits['FTBCdef-T8-C3'])
        t8 = acc['per_T']['T8']; assert 'FTBCdef' not in t8['ftbc_arms'] and t8['FTBCdef']['draws_out_of_memory'] == ['C2'], t8['FTBCdef']
    report.update(passed=acc['passed'], injected=INJECT, variant=VARIANT, per_T={T: {'best_ftbc': v['best_ftbc'], 'threshold_d': v['questions']['threshold_d']['reading'],
                                                                    'FTBCdef': v['FTBCdef'], 'arms': sorted(v['top1_mean'])} for T, v in acc['per_T'].items()})
    shutil.copy(fake / f'{TAG}-sanity-acceptance.json', SCRATCH / f'cpu-{RUN_TAG}-sanity-acceptance.json')
    code = files[:4] + [L1 / 'config' / f for f in [f'{TAG}-sanity.json', f'{TAG}-main.json']] + [Path(__file__).resolve()]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU logic dry run at sanity sizes with training-image stand-ins; memory readings are stand-ins and the numbers are meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    report['tag'] = RUN_TAG; (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['tag', 'passed', 'cuda_initialized', 'runner_seconds', 'per_T']}))
    if 'traceback' in report: print(report['traceback'])
