"""E23 fitting-cost measurement, ResNet-34 child (EXPERIMENT_PLAN.md): on serial R34 Q8 (E0 plain), T=2, 4 and 8, the GPU-synchronized wall time
and the peak allocator memory of P0 on the first 32 and 96 images of the 600-image pool C1, the registered FTBC recipe on the first 96 with one
pass (3 updates) and with three passes (9 updates; the alpha grid of the locked evaluation uses 1 or 3 passes) and (T=2 and 4) the
release-default recipe on all 600 images, augmented, each twice, one process alone on the card. No accuracy is reported; every arm is scored
on 16+8 images only for the strict-reload check.

Reuses the frozen E17 runner (and through it E16-E13) unchanged: plain state, calibration draws with the disjointness checks, P0
(step_calibrate, whose per-layer `seconds` are capture + solve + one IF replay assertion), the released FTBC update with the counting wrapper
and an explicit ANN (ftbc_batches_ann), registered batches, the E15 release_batches (release transform, seeded shuffle) and strict-reload
scoring. The registered-recipe call is the E20 one (ftbc_registered), and three passes repeat its three batches. New here: the timing
(e23_timing.timed with the wrapped parts: data, target, replay, write, digest, hash; initial review B1), the memory baseline and increments
(B2), the repeats, the release rows (the whole pool C1), the arm loop and the release-default failure rule (only out-of-memory is tolerated,
B3; with the library-level signatures backed by allocator evidence and a context check, follow-up 1 B5)."""
import argparse, hashlib, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E23 acceptance-critical checks use assert; do not run with -O')
CAL, TS, REPEATS = ['C1', 'C2', 'C3'], [2, 4, 8], [1, 2]          # sizes come from c['sizes']: [32, 96] in main, [4, 12] in sanity
RELEASE_TS = [2, 4, 8]                                 # section 3: T=8 attempted under the out-of-memory rule (E15/E16 reserved 77.4 GiB at T=4; initial review N4)
LIB_OOM = ('unable to find an engine', 'CUDNN_STATUS_ALLOC_FAILED', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_INTERNAL_ERROR')   # section 6 item 9


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(c):
    E17 = load('e23_e17', c['e17_runner'])
    E13, E14, E15, E16, ss, ps, ce, Run17 = E17.build(c)

    class Run(Run17):
        def __init__(self, c):
            super().__init__(c)                                # E17: plain state, draws, disjointness from every earlier role and split
            NS = c['sizes']; assert len(NS) == 2 and NS[0] < NS[1] == c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n'] and c['ftbc_alpha'] == 0.4
            cal = json.loads(Path(c['calibration_manifest']).read_text())   # E17 checked every row of it against every earlier role and split
            assert len(cal['C1']) >= c['release_n'] * c['release_batch'] and self.draws['C1'] == cal['C1'][:c['draw_n']]
            self.release_rows = cal['C1'][:c['release_n'] * c['release_batch']]; self.cost = {}    # main: the whole pool of 600; sanity: a prefix

        def ftbc_registered(self, arm, T, rows, passes=1):
            """E20 ftbc_registered: the released update, alpha 0.4, 3 batches of 32, the Q8 teacher as ANN; `passes` > 1 repeats the three
            batches in the same order (the released update accumulates across the list it is given)."""
            alpha, batches = self.c['ftbc_alpha'], self.registered_batches(rows)
            recipe = 'registered' if passes == 1 else f'registered-{passes}pass'
            return self.ftbc_batches_ann(arm, T, batches * passes, alpha, [int(x['index']) for x in rows], recipe, self.teacher, 'teacher',
                                         extra={'variant': 'released', 'passes': passes, 'batches_per_pass': len(batches)})

    return E13, E14, E15, E16, ss, ps, ce, Run


def oom_basis(exc, ooms_before):
    """Section 6 item 9 (follow-up 1 B5). 'exception': torch.cuda.OutOfMemoryError or an 'out of memory' message. 'library': a RuntimeError
    whose message has one of the pre-registered library signatures (LIB_OOM: cuDNN finding no engine after its workspace allocations failed,
    cuDNN/cuBLAS allocation or internal errors when the device is full), backed by allocator evidence: num_ooms grew during the attempt, or less
    than 1 GiB is free on the device. None otherwise: the caller re-raises (an AssertionError never qualifies)."""
    if isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower():
        return 'exception', None, None
    if isinstance(exc, RuntimeError) and any(sig.lower() in str(exc).lower() for sig in LIB_OOM):
        grew = torch.cuda.memory_stats().get('num_ooms', 0) - ooms_before; free = torch.cuda.mem_get_info()[0]
        if grew > 0 or free < 2 ** 30:
            return 'library', grew, free
    return None, None, None


def cuda_alive():
    """Follow-up 1 B5: after a tolerated out-of-memory error a tiny CUDA operation must still succeed, or the run fails."""
    x = torch.ones(1).cuda() + 1; torch.cuda.synchronize(); return float(x) == 2.0


@torch.no_grad()
def run(c):
    E13, E14, E15, E16, ss, ps, ce, Run = build(c); r = Run(c); rows = r.draws['C1']; small = c['small_n']; NS = c['sizes']
    TM = load('e23_timing', str(Path(__file__).resolve().parent / 'e23_timing.py'))
    W = [(r.e0, 'loader', 'data', True), (r, 'teacher_means', 'target', False), (r, 'replay_check', 'replay', False), (r.e0, 'atomic_torch', 'write', False),
         (r.e0, 'atomic_json', 'write', False), (r.e0, 'state_digest', 'digest', False), (E15, 'tensor_hash', 'hash', False)]
    timed = lambda fn: TM.timed(r, fn, W)
    r.net(4); r.zero_biases(2)                             # initial review B2: the T=4 network and the channel table that the first arm would build
    def score(arm, T, b, detail):
        r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
    def record(arm, T, cost):
        r.cost[arm] = {'T': T, **cost}; r.e0.atomic_json(r.out / 'cost.json', r.cost)
        r.progress('cost', arm=arm, cost={k: v for k, v in cost.items() if k not in ('calls', 'traceback')})
    for T in TS:
        r.net(T)                                           # build the T-step network outside every timed call
        for k in REPEATS:
            for n in NS:
                arm = f'P0-{n}-T{T}-r{k}'; key = f'C1-{n}'; sub = rows[:n]
                r.targets_by_n.pop(f'{key}-teacher', None)                 # section 4: the targets are recomputed (and timed) in every repeat
                def fit():
                    r.use_rows_target(key, sub, 'teacher'); return r.step_calibrate(arm, T)
                b, cost = timed(fit)
                s = r.solver[arm]; s['target_key'] = r.target_key; s['state'] = r.variant; s['T'] = T
                cost.update(method='P0', fit_images=n, layer_seconds_sum=sum(v['seconds'] for v in s['layers'].values()), captures=len(s['layers']),
                            capture_batches=math.ceil(n / c['solver_batch']), cache_bytes_max=max(v['cache_bytes'] for v in s['layers'].values()))
                record(arm, T, cost); score(arm, T, b, {'solver': 'P0', 'target_key': r.target_key, 'rows': key})
            for passes in [1, 3]:
                arm = f'F0reg{passes}-T{T}-r{k}'
                b, cost = timed(lambda: r.ftbc_registered(arm, T, rows[:small], passes))
                cost.update(method=f'FTBC registered, {passes} pass' + ('es' if passes > 1 else ''), fit_images=small, batches=3 * passes,
                            batch_size=c['ftbc_batch'], calls=r.ftbc[arm]['calls'])
                record(arm, T, cost); score(arm, T, b, {'ftbc': r.ftbc[arm]})
            if T in RELEASE_TS:
                arm = f'F0rel-T{T}-r{k}'; seed = c['release_seed']; stage, data_seconds = 'data', None
                t_unix, t_perf, ooms_before = time.time(), time.perf_counter(), torch.cuda.memory_stats().get('num_ooms', 0)   # the window starts with the data step (B4)
                try:                                           # initial review B3: only the data step and the timed update are guarded, only OOM is tolerated
                    batches, order = r.release_batches(r.release_rows, seed); data_seconds = time.perf_counter() - t_perf
                    stage = 'update'
                    b, cost = timed(lambda: r.ftbc_batches_ann(arm, T, batches, c['ftbc_alpha'], order, 'release-default', r.teacher, 'teacher', {'seed': seed}))
                except Exception as exc:
                    basis, grew, free = oom_basis(exc, ooms_before)
                    if basis is None:
                        raise
                    t_end, u_end = time.perf_counter(), time.time(); r.ftbc.pop(arm, None)
                    upd = stage == 'update'                    # follow-up 1 b: after a data-stage failure the peaks would belong to the previous arm
                    rec = {'method': 'FTBC release default', 'status': 'out_of_memory', 'stage': stage, 'oom_basis': basis, 'exception_type': type(exc).__name__,
                           'exception_message': str(exc)[:500], 'num_ooms_increase': torch.cuda.memory_stats().get('num_ooms', 0) - ooms_before, 'device_free_bytes': free,
                           'peak_allocated_bytes': torch.cuda.max_memory_allocated() if upd else None, 'peak_reserved_bytes': torch.cuda.max_memory_reserved() if upd else None,
                           'card_total_bytes': r.hardware['total_memory'], 'data_seconds': data_seconds, 'attempt_unix_start': t_unix, 'attempt_unix_end': u_end,
                           'attempt_perf_start': t_perf, 'attempt_perf_end': t_end, 'traceback': traceback.format_exc()}
                    del exc; batches = None; rec['leftover_hooks_dropped'] = r.clear_release_hooks(); r.free()
                    assert cuda_alive(), f'{arm}: CUDA context unusable after the out-of-memory error'
                    record(arm, T, rec)
                else:
                    del batches
                    cost.update(method='FTBC release default', fit_images=len(r.release_rows), batches=c['release_n'], batch_size=c['release_batch'], data_seconds=data_seconds,
                                attempt_unix_start=t_unix, attempt_perf_start=t_perf, calls=r.ftbc[arm]['calls'], status='completed')
                    record(arm, T, cost); score(arm, T, b, {'ftbc': r.ftbc[arm]})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for key, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    assert r.clear_release_hooks() == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'cost_arms': list(r.cost), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'proxy_digests': {f'{k[0]}-T{k[1]}': v['proxy_digest'] for k, v in r.nets.items()}, 'snn_digests': {f'{k[0]}-T{k[1]}': v['immutable_digest'] for k, v in r.nets.items()},
              'teacher_digest': r.teacher_digest, 'deployment_sha256': r.deployment_sha, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'draws': {d: [int(x['index']) for x in v] for d, v in r.draws.items()}, 'release_rows': [int(x['index']) for x in r.release_rows],
              'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0, 'allocator_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
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
