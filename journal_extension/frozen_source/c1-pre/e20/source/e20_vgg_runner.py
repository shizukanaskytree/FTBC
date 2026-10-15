"""E20 calibration-budget screen, VGG-16 child (EXPERIMENT_PLAN.md): P0 on the first N in {8, 16, 32, 96} images of each calibration draw
against registered FTBC on the first 96, at T=4 and T=8, on serial VGG-16 Q16. One of the two child processes of the E20 allocation.

Uses the frozen E18 runner's Run class unchanged (model construction, the Q16 teacher, prefix capture, P0 = step_calibrate, the released FTBC
update with the declared same-target adaptations = ftbc_run(..., 'registered'), strict-reload scoring with one saved state per T, the data
identity and disjointness checks). Only the arm loop below is new. The E18 constructor's ECCV-schedule keys are kept in the config only to
satisfy its assertions; no ECCV-schedule arm runs here.
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E20 acceptance-critical checks use assert; do not run with -O')
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]          # calibration sizes come from c['sizes']: [8, 16, 32, 96] in main, [2, 4, 8, 12] in sanity


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


@torch.no_grad()
def run(c):
    R18 = load('e20_e18_runner', c['e18_runner'])
    SIZES = c['sizes']; assert SIZES == sorted(set(SIZES)) and len(SIZES) == 4 and SIZES[-1] == c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n']
    r = R18.Run(c); small = c['small_n']
    def p0(arm, T, key):
        r.free(); b = r.step_calibrate(arm, T); r.score(arm, T, b, {'solver': 'P0', 'rows': key, 'target_key': r.target_key})
    r.score_ann('teacher', r.teacher, 4)
    for T in TS:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T); r.score(f'S0-T{T}', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
    for d, rows in r.draws.items():
        for n in SIZES:
            key = f'{d}-{n}'; r.use_rows(key, rows[:n])
            for T in TS:
                p0(f'P0-T{T}-{key}', T, key)
        r.use_rows(f'{d}-{small}', rows[:small])
        for T in TS:
            arm = f'F0-T{T}-{d}'; r.free(); b = r.ftbc_run(arm, T, rows[:small], 'registered'); r.score(arm, T, b, {'ftbc': r.ftbc[arm]})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for T, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    leftover = sum(isinstance(h, r.inherited.ActivationSaverHookEachTimeStep) for mdl in [r.teacher] + [N['snn'] for N in r.nets.values()]
                   for m in mdl.modules() for h in m._forward_hooks.values())
    assert leftover == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True,
              'teacher_digest': r.teacher_digest, 'proxy_digests': {f'T{T}': v['proxy_digest'] for T, v in r.nets.items()},
              'snn_digests': {f'T{T}': v['immutable_digest'] for T, v in r.nets.items()}, 'interfaces': r.names,
              'channels': {n: r.channels(n) for n in r.names}, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'sizes': SIZES, 'hardware': r.hardware,
              'seconds_total': time.perf_counter() - r.t0, 'max_memory_allocated': torch.cuda.max_memory_allocated(),
              'max_memory_reserved': torch.cuda.max_memory_reserved(), 'data_status': 'fresh-G development (prospective) and exposed dev512 continuity'}
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
