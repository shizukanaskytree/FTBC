"""E22 TPP versus P0, ResNet-34 child (EXPERIMENT_PLAN.md): on serial R34 Q8 (E0 plain), T=4 and T=8, the TPP probabilistic neuron (five
seeds, no calibration images) against P0 on the 256 images of each calibration draw and the registered FTBC recipe on the first 96. One of the
two child processes of the E22 allocation.

Reuses the frozen E17 runner (and through it E16-E13) unchanged: plain state, calibration draws from the configured manifest with the
disjointness checks against every earlier split, P0 (step_calibrate), the released FTBC update with the counting wrapper and an explicit ANN
(ftbc_batches_ann), registered batches, prefix capture, IF replay checks and strict-reload scoring. The registered-recipe call is the E20 one
(ftbc_registered). New here: the TPP arms (tpp_neuron.py) and the arm loop.
"""
import argparse, contextlib, fcntl, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E22 acceptance-critical checks use assert; do not run with -O')
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


@contextlib.contextmanager
def p0_lock(c, r, arm):
    """Section 8: the two children never run P0 calibration at the same time (an exclusive flock on one file in the shared output root),
    so their P0 memory peaks cannot overlap; the caching allocator is emptied before the lock is released. Scheduling only: no numerical effect."""
    path = Path(c['output']).parent / 'p0-calibration.lock'; t0 = time.perf_counter()
    with open(path, 'a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX); r.progress('p0_lock_acquired', arm=arm, waited_seconds=time.perf_counter() - t0)
        try:
            yield
        finally:
            torch.cuda.empty_cache(); fcntl.flock(fh, fcntl.LOCK_UN)


def build(c):
    E17 = load('e22_e17', c['e17_runner'])
    E13, E14, E15, E16, ss, ps, ce, Run17 = E17.build(c)
    TN = load('e22_r34_tpp', c['tpp_neuron'])

    class Run(Run17):
        def __init__(self, c):
            super().__init__(c)                                # E17: plain state, draws, disjointness from every earlier role and split
            assert c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n'] and c['ftbc_alpha'] == 0.4 and c['seeds'] == sorted(set(c['seeds']))
            assert len(c['seeds']) >= len(CAL)

        def ftbc_registered(self, arm, T, rows):
            """E20 ftbc_registered (E19 ftbc_run(arm, T, rows, 'released', 'registered', 1)): the released update, alpha 0.4, 3 batches of 32, the
            Q8 teacher as ANN."""
            alpha, batches = self.c['ftbc_alpha'], self.registered_batches(rows)
            return self.ftbc_batches_ann(arm, T, batches, alpha, [int(x['index']) for x in rows], 'registered', self.teacher, 'teacher',
                                         extra={'variant': 'released', 'passes': 1, 'batches_per_pass': len(batches)})

    return E13, E14, E15, E16, ss, ps, ce, TN, Run


@torch.no_grad()
def run(c):
    E13, E14, E15, E16, ss, ps, ce, TN, Run = build(c); r = Run(c); small = c['small_n']
    def score(arm, T, b, detail):
        r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
    def p0(arm, T, key):
        with p0_lock(c, r, arm):
            r.free(); b = r.step_calibrate(arm, T)
        r.solver[arm]['target_key'] = r.target_key; r.solver[arm]['state'] = r.variant; r.solver[arm]['T'] = T
        score(arm, T, b, {'solver': 'P0', 'target_key': r.target_key, 'rows': key}); return b
    r.score_ann('teacher', r.teacher, 4)
    for T in TS:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T); score(f'S0-T{T}', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
    for T in TS:
        for s in c['seeds']:
            r.free(); TN.score_tpp(r, f'TPP-T{T}-{s}', T, s, r.classes['spike'])
    for d, rows in r.draws.items():
        r.use_rows_target(d, rows, 'teacher')               # P0 on all draw_n (256) images of the draw
        for T in TS:
            p0(f'P0-T{T}-{d}', T, d)
        r.use_rows_target(f'{d}-{small}', rows[:small], 'teacher')
        for T in TS:
            arm = f'F0-T{T}-{d}'; r.free(); b = r.ftbc_registered(arm, T, rows[:small]); score(arm, T, b, {'ftbc': r.ftbc[arm]})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for key, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
        assert not any('forward' in vars(m) for m in N['snn'].modules()), 'a swapped TPP forward was left on a network'
    assert r.clear_release_hooks() == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True, 'tpp_forward_restored': True,
              'proxy_digests': {f'{k[0]}-T{k[1]}': v['proxy_digest'] for k, v in r.nets.items()}, 'snn_digests': {f'{k[0]}-T{k[1]}': v['immutable_digest'] for k, v in r.nets.items()},
              'teacher_digest': r.teacher_digest, 'deployment_sha256': r.deployment_sha, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'seeds': c['seeds'],
              'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-J development (prospective) and exposed dev512 continuity'}
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
