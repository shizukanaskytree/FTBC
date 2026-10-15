"""E23 fitting-cost measurement, VGG-16 child (EXPERIMENT_PLAN.md): on serial VGG-16 Q16, T=2, 4 and 8, the GPU-synchronized wall time and the
peak allocator memory of P0 on the first 32 and 96 images of the pool C1 and of the registered FTBC recipe (the declared same-target
adaptations) on the first 96 with one pass (3 updates) and three passes (9 updates), each twice, one process alone on the card. No accuracy is
reported; every arm is scored on 16+8 images only for the strict-reload check. The release-default recipe is not run on VGG-16 (section 3).

Uses the frozen E18 runner's Run class unchanged (model construction, the Q16 teacher, prefix capture, P0 = step_calibrate with per-layer
`seconds`, the released FTBC update = ftbc_run(..., 'registered') with the counting wrapper, strict-reload scoring with one saved state per T, the
data identity and disjointness checks). The E18 constructor's ECCV-schedule keys stay in the config only to satisfy its assertions. New here:
the timing (e23_timing.timed with the wrapped parts: data, target, replay, write, digest, hash; initial review B1), the memory baseline and
increments (B2), the repeats, the arm loop, and a three-pass registered recipe through two overrides of the E18 Run (recipe: 9 batches;
ftbc_batches: the three registered batches in manifest order, three times); the frozen ftbc_run body is unchanged."""
import argparse, hashlib, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E23 acceptance-critical checks use assert; do not run with -O')
CAL, TS, REPEATS = ['C1', 'C2', 'C3'], [2, 4, 8], [1, 2]          # sizes come from c['sizes']: [32, 96] in main, [4, 12] in sanity


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


@torch.no_grad()
def run(c):
    R18 = load('e23_e18_runner', c['e18_runner'])
    NS = c['sizes']; assert len(NS) == 2 and NS[0] < NS[1] == c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n']

    class Run(R18.Run):
        def recipe(self, name):
            if name == 'registered-3pass':
                return self.c['ftbc_alpha'], self.c['ftbc_batch'], 9
            return super().recipe(name)

        def ftbc_batches(self, rows, batch_size, n_batches):
            if n_batches == 9:                             # three passes over the registered three batches, manifest order in each pass
                return super().ftbc_batches(rows, batch_size, 3) * 3
            return super().ftbc_batches(rows, batch_size, n_batches)

    r = Run(c); rows = r.draws['C1']; small = c['small_n']; cost_rec = {}
    TM = load('e23_timing', str(Path(__file__).resolve().parent / 'e23_timing.py'))
    W = [(r.e0, 'loader', 'data', True), (r, 'teacher_means', 'target', False), (r, 'replay_check', 'replay', False), (r.e0, 'atomic_torch', 'write', False),
         (r.e0, 'atomic_json', 'write', False), (r.e0, 'state_digest', 'digest', False), (R18, 'tensor_hash', 'hash', False)]
    timed = lambda fn: TM.timed(r, fn, W)
    r.zero_biases(2)                                       # initial review B2: the channel table (one teacher forward) that the first arm would build
    def record(arm, T, cost):
        cost_rec[arm] = {'T': T, **cost}; r.e0.atomic_json(r.out / 'cost.json', cost_rec)
        r.progress('cost', arm=arm, cost={k: v for k, v in cost.items() if k not in ('calls', 'traceback')})
    for T in TS:
        r.net(T)                                           # build the T-step network outside every timed call
        for k in REPEATS:
            for n in NS:
                arm = f'P0-{n}-T{T}-r{k}'; key = f'C1-{n}'; sub = rows[:n]
                r.targets_by_n.pop(f'{key}-teacher', None)                 # section 4: the targets are recomputed (and timed) in every repeat
                def fit():
                    r.use_rows(key, sub); return r.step_calibrate(arm, T)
                b, cost = timed(fit)
                s = r.solver[arm]
                cost.update(method='P0', fit_images=n, layer_seconds_sum=sum(v['seconds'] for v in s['layers'].values()), captures=len(s['layers']),
                            capture_batches=math.ceil(n / c['solver_batch']), cache_bytes_max=max(v['cache_bytes'] for v in s['layers'].values()))
                record(arm, T, cost); r.score(arm, T, b, {'solver': 'P0', 'rows': key, 'target_key': r.target_key})
            for passes, recipe in [(1, 'registered'), (3, 'registered-3pass')]:
                arm = f'F0reg{passes}-T{T}-r{k}'
                b, cost = timed(lambda: r.ftbc_run(arm, T, rows[:small], recipe))
                cost.update(method=f'FTBC registered, {passes} pass' + ('es' if passes > 1 else ''), fit_images=small, batches=3 * passes,
                            batch_size=c['ftbc_batch'], calls=r.ftbc[arm]['calls'])
                record(arm, T, cost); r.score(arm, T, b, {'ftbc': r.ftbc[arm]})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for T, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    leftover = sum(isinstance(h, r.inherited.ActivationSaverHookEachTimeStep) for mdl in [r.teacher] + [N['snn'] for N in r.nets.values()]
                   for m in mdl.modules() for h in m._forward_hooks.values())
    assert leftover == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'cost_arms': list(cost_rec), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'teacher_digest': r.teacher_digest, 'proxy_digests': {f'T{T}': v['proxy_digest'] for T, v in r.nets.items()},
              'snn_digests': {f'T{T}': v['immutable_digest'] for T, v in r.nets.items()}, 'interfaces': r.names,
              'channels': {n: r.channels(n) for n in r.names}, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'draws': {d: [int(x['index']) for x in v] for d, v in r.draws.items()}, 'hardware': r.hardware,
              'seconds_total': time.perf_counter() - r.t0, 'allocator_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'peak_allocated_since_last_reset': torch.cuda.max_memory_allocated(), 'peak_reserved_since_last_reset': torch.cuda.max_memory_reserved(),   # N9: every arm resets the peaks
              'data_status': 'fresh-K (16) and dev512 (first 8) used only for strict-reload scoring; no accuracy reported'}
    r.e0.atomic_json(r.out / 'result.json', result); r.progress('completed')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args(); c = json.loads(Path(a.config).read_text())
    try:
        print(json.dumps(run(c)))
    except Exception:
        Path(c['output']).mkdir(parents=True, exist_ok=True)
        (Path(c['output']) / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}))
        raise
