"""CPU-only logic dry run of the E18 runner + acceptance (EXPERIMENT_PLAN.md section 7) at reduced sizes and full 224x224 resolution (the VGG-16
ImageNet classifier needs 7x7 features). Never initializes CUDA. Also checks P0 prefix consistency (T=4 equals the first four steps of T=8)."""
import contextlib, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path
SCRATCH = Path(sys.argv[1]).resolve()
E18 = Path(__file__).resolve().parents[1]
import torch
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
torch.set_num_threads(int(os.environ.get('E18_DRYRUN_THREADS', '12')))
torch.set_num_threads = lambda n: None                 # dry run only (no DataLoader workers here): keep the chosen thread count when the runner asks for 8
sys.path.insert(0, str(E18 / 'source'))
import e18_runner, e18_accept
original_load = e18_runner.load


def fake_guard(allowed):
    return {'slurm_job_id': '999999', 'pid': os.getpid(), 'cgroup': '0::/cpu-dryrun/slurm-mock\n', 'gpu_uuid': allowed[0],
            'name': 'CPU dry run (no CUDA)', 'total_memory': 84974239744, 'torch': torch.__version__, 'cuda_runtime': None, 'cudnn': None}


def patched_load(name, path):
    module = original_load(name, path)
    if name == 'e18_e13':
        module.gpu_guard = fake_guard
    return module


e18_runner.load = patched_load
assert not SCRATCH.exists() or not any(SCRATCH.iterdir()); SCRATCH.mkdir(parents=True, exist_ok=True)
report = {'started': time.time()}
try:
    base = json.loads((E18 / 'config/sanity.json').read_text())
    small = dict(draw_n=8, small_n=6, ftbc_batch=2, eccv_batches=2, eccv_n=4, fresh_count=8, dev_count=4, solver_batch=4, num_workers=0)   # CPU-sized
    fake = SCRATCH / 'fakeE'; (fake / 'config').mkdir(parents=True); (fake / 'slurm').mkdir()
    files = [E18 / 'source' / f for f in ['e18_runner.py', 'e18_accept.py']] + [E18 / 'EXPERIMENT_PLAN.md', Path(base['e13_runner']), Path(base['step_solver']), Path(base['serial_solver'])]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (fake / 'protocol-freeze.json').write_text(json.dumps({'files_sha256': frozen}))
    w = Path(base['workspace']); (fake / 'slurm-source-manifest.json').write_text(json.dumps({'files_sha256': {str(Path(p).relative_to(w)): h for p, h in frozen.items()}}))
    shutil.copy(E18 / 'preservation-before.json', fake / 'preservation-before.json')
    e18_accept.E = fake
    c = dict(base, **small, output=str(SCRATCH / 'sanity-cpu'), freeze_manifest=str(fake / 'protocol-freeze.json'))
    (fake / 'config/sanity.json').write_text(json.dumps(c, indent=2))
    t0 = time.time()
    with open(fake / 'slurm/sanity-999999.out', 'w') as fh, contextlib.redirect_stdout(fh):
        print(json.dumps(e18_runner.run(c)))
    with open(SCRATCH / 'accept-sanity.txt', 'w') as fh, contextlib.redirect_stdout(fh):
        e18_accept.check('sanity')
    acc = json.loads((fake / 'sanity-acceptance.json').read_text())
    report.update(passed=acc['passed'], runner_seconds=time.time() - t0, arms=sorted(acc['metrics']), claims=acc['claims'], prefix_consistency=acc['prefix_consistency_gpu'],
                  sizes=small)
    # Prefix consistency is reported, not required (EXPERIMENT_PLAN.md section 7 item 10): a different T changes the merged batch dimension,
    # so BLAS/conv blocking and FP32 summation order can differ; report where T=4 and the first four steps of T=8 first diverge, and by how much.
    first = {}
    for key in acc['prefix_consistency_gpu']:
        a = torch.load(SCRATCH / 'sanity-cpu' / f'bias-{key[:2]}-T4-{key[3:]}.pt', weights_only=False)['runtime_biases']
        b = torch.load(SCRATCH / 'sanity-cpu' / f'bias-{key[:2]}-T8-{key[3:]}.pt', weights_only=False)['runtime_biases']
        diffs = [(i, n, t, float((a[n][t] - b[n][t]).abs().max())) for i, n in enumerate(a) for t in range(4) if not torch.equal(a[n][t], b[n][t])]
        first[key] = {'first_divergence': diffs[0][:3] if diffs else None, 'max_abs_diff': max((d[3] for d in diffs), default=0.0), 'differing_layer_steps': len(diffs)}
    report['prefix_divergence'] = first
    # Binding tolerance (review B3): the 13 convolutional interfaces agree bitwise; the two classifier (vector) interfaces within 1e-5 theta.
    theta = {n: v['theta'] for n, v in torch.load(SCRATCH / 'sanity-cpu' / 'solver.pt', weights_only=False)['P0-T4-C1']['layers'].items()}
    worst = {}
    for key in acc['prefix_consistency_gpu']:
        a = torch.load(SCRATCH / 'sanity-cpu' / f'bias-{key[:2]}-T4-{key[3:]}.pt', weights_only=False)['runtime_biases']
        b = torch.load(SCRATCH / 'sanity-cpu' / f'bias-{key[:2]}-T8-{key[3:]}.pt', weights_only=False)['runtime_biases']
        for n in a:
            d = max(float((a[n][t] - b[n][t]).abs().max()) for t in range(4)) / theta[n]
            if n.startswith('classifier.'):
                assert d <= 1e-5, (key, n, d)
            else:
                assert d == 0.0, (key, n, d)
            worst[n] = max(worst.get(n, 0.0), d)
    report['prefix_max_abs_diff_over_theta_by_interface'] = worst
    shutil.copy(fake / 'sanity-acceptance.json', SCRATCH / 'cpu-sanity-acceptance.json')
    code = [E18 / 'source' / f for f in ['e18_runner.py', 'e18_accept.py']] + [E18 / 'config' / f for f in ['sanity.json', 'main.json']]
    report['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    report['note'] = 'CPU dry run at reduced sizes, 224x224, VGG-16 Q16 source; numbers meaningless'
except Exception:
    report.update(passed=False, traceback=traceback.format_exc())
finally:
    report['cuda_initialized'] = REAL_IS_INITIALIZED(); report['finished'] = time.time()
    (SCRATCH / 'dryrun-report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'cuda_initialized', 'runner_seconds', 'prefix_consistency']}))
    if 'traceback' in report: print(report['traceback'])
