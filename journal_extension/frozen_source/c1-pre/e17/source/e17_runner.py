"""E17 serial solver screen at T=2 and T=4 (EXPERIMENT_PLAN.md): per-step cumulative per-unit error objectives (CE, CE-A) against P0, M0
and registered FTBC on the E0 plain deployment. One process and one Slurm-allocated A100 per job.

Reuses the frozen E16 runner (and through it E15/E14/E13): net(T) by state and T (plain only here), teacher targets with network digest and
level, P0 (step_calibrate), M0 (count_calibrate_fit), the released FTBC update with an explicit ANN (ftbc_batches_ann, the Q8 teacher here),
registered batches, prefix capture, IF replay checks, strict-reload scoring on the fresh and development roles. New here: calibration rows
from the E17 manifest, the per-unit teacher reference output at each interface (reference_output) and the CE/CE-A layer loop
(step_calibrate_ce) around ce_solver.py.
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E17 acceptance-critical checks use assert; do not run with -O')
CAL, TS = ['C1', 'C2', 'C3'], [2, 4]
SOLVERS = ['P0', 'M0', 'CE', 'CE-A']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def build(c):
    E16 = load('e17_e16', c['e16_runner'])
    E13, E14, E15, ss, ps, Run16 = E16.build(c)
    ce = load('e17_ce', c['ce_solver'])
    Run14 = Run16.__mro__[2]                                   # Run16 -> Run15 -> Run14

    class Run(Run16):
        def __init__(self, c):
            Run14.__init__(self, c)                            # E14/E13 initialisation; E15/E16 state and draw set-up are replaced below
            old = [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
            old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
            assert not {int(r['index']) for r in self.rows['fresh']} & {int(r['index']) for r in old}, 'fresh-D overlaps an earlier split (ID)'
            assert not {r['sha256'] for r in self.rows['fresh']} & {r['sha256'] for r in old}, 'fresh-D overlaps an earlier split (content)'
            assert sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
            cal = json.loads(Path(c['calibration_manifest']).read_text())
            assert sorted(cal) == CAL and all(len(cal[k]) >= c['draw_n'] for k in CAL)
            self.draws = {k: cal[k][:c['draw_n']] for k in CAL}
            used = [r for role in self.roles.values() for r in role] + old + self.rows['fresh']
            allc = [r for k in CAL for r in cal[k]]
            assert len({int(r['index']) for r in allc}) == len(allc) and not {int(r['index']) for r in allc} & {int(r['index']) for r in used}
            assert not {r['sha256'] for r in allc} & {r['sha256'] for r in used}, 'calibration rows overlap earlier roles (content)'
            self.states = {'plain': {'proxy': self.proxy_state, 'deploy': self.deploy}}
            self.deployment_sha = {'plain': sha(c['plain_deployment'])}
            self.variant = 'plain'; self.nets = {}; self.grid_proxies = {}; self.release_record = {}; self.release_sets = {}

        @torch.no_grad()
        def reference_output(self, model, name):
            """(N,C,H,W) output of `model` at interface `name` on the active fit rows (batch solver_batch); the forward stops there."""
            mod = dict(model.named_modules())[name]; out = []
            def hook(_, args, y):
                out.append(y.detach().clone()); raise E13.PrefixDone()
            h = mod.register_forward_hook(hook)
            try:
                for x, _, _ in self.fit_batches:
                    try: model(x.cuda())
                    except E13.PrefixDone: pass
                    else: raise AssertionError('reference prefix hook did not fire')
            finally:
                h.remove()
            a = torch.cat(out); assert a.shape[0] == self.fit_n and torch.isfinite(a).all()
            return a

        @torch.no_grad()
        def step_calibrate_ce(self, arm, T, anchor):
            """CE (anchor=False) or CE-A (anchor=True) in forward order with the E14 prefix capture; the Q8 teacher is the per-unit reference."""
            N = self.net(T); targets = self.targets; biases = self.zero_biases(T); self.set_biases(T, biases); info = {}; bound = self.c['bound_ratio']
            ref_before = self.e0.state_digest(self.teacher)
            for i, n in enumerate(N['names']):
                m = N['modules'][n]; thre = m.thresh.data; th = float(thre); t_start = time.perf_counter()
                x = self.capture(T, n); a = self.reference_output(self.teacher, n); C = x.shape[2]
                assert a.shape == x.shape[1:], (n, tuple(a.shape), tuple(x.shape))
                ref_mean = a.sum((0, 2, 3), dtype=torch.float64) / (a.numel() // C)
                b, steps, total = ce.solve_per_step_ce(ps, x, a, thre, bound=bound, anchor_target=targets[n] if anchor else None,
                                                       steps=self.c['bisection_steps'])
                assert b.shape == (T, C) and b.dtype == torch.float32 and torch.isfinite(b).all() and bool((b.abs() <= bound * th).all())
                assert torch.equal(ss.replay(x, b, thre), total), f'{arm}: CE solver totals differ from the IF replay at {n}'
                final_counts = ss.channel_counts(total).cpu()
                m.time_based_bias = [b[t].clone() for t in range(T)]; biases[n] = [b[t].cpu() for t in range(T)]
                info[n] = {'steps': steps, 'theta': th, 'b': b.cpu(), 'final_counts': final_counts, 'reference_mean': ref_mean.cpu(),
                           'target_mean': targets[n].cpu(), 'observations_per_channel_per_step': x[0].numel() // C,
                           'seconds': time.perf_counter() - t_start, 'cache_bytes': x.numel() * 4 + a.numel() * 4}
                del x, a, total
                self.e0.atomic_torch(self.out / f'solver-{arm}.pt', info)
                self.progress('calibrated_layer', arm=arm, index=i + 1, name=n, near_ties=sum(s['near_tie_mismatches'] for s in steps))
            assert self.e0.state_digest(self.teacher) == ref_before == self.teacher_digest
            self.replay_check(arm, T, info)
            self.solver[arm] = {'full_network_replay_equal': True, 'per_step': True, 'objective': 'CE-A' if anchor else 'CE', 'reference': 'teacher',
                                'reference_digest': ref_before, 'fit_images': self.fit_n, 'layers': info}
            return biases

    return E13, E14, E15, E16, ss, ps, ce, Run


@torch.no_grad()
def run(c):
    E13, E14, E15, E16, ss, ps, ce, Run = build(c); r = Run(c); small = c['small_n']
    def score(arm, T, b, detail):
        r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
    def solve(arm, T, kind, detail):
        r.free()
        if kind == 'P0': b = r.step_calibrate(arm, T)
        elif kind == 'M0': b = r.count_calibrate_fit(arm, T)
        else: b = r.step_calibrate_ce(arm, T, anchor=(kind == 'CE-A'))
        r.solver[arm]['target_key'] = r.target_key; r.solver[arm]['state'] = r.variant; r.solver[arm]['T'] = T
        score(arm, T, b, {**detail, 'solver': kind, 'target_key': r.target_key})
    r.score_ann('teacher', r.teacher, 4)
    for T in TS:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T); score(f'S0-T{T}', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
    for d, rows in r.draws.items():
        r.use_rows_target(d, rows, 'teacher')
        for T in TS:
            for kind in SOLVERS:
                solve(f'{kind}-T{T}-{d}', T, kind, {'rows': d})
        r.use_rows_target(f'{d}s', rows[:small], 'teacher')
        for T in TS:
            solve(f'CE-A-T{T}-{d}s', T, 'CE-A', {'rows': f'{d}s'})
            r.free(); b = r.ftbc_batches_ann(f'F0-T{T}-{d}s', T, r.registered_batches(rows[:small]), c['ftbc_alpha'], [int(x['index']) for x in rows[:small]],
                                             'registered', r.teacher, 'teacher')
            score(f'F0-T{T}-{d}s', T, b, {'ftbc': r.ftbc[f'F0-T{T}-{d}s']})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for key, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    assert r.clear_release_hooks() == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True,
              'proxy_digests': {f'{k[0]}-T{k[1]}': v['proxy_digest'] for k, v in r.nets.items()}, 'snn_digests': {f'{k[0]}-T{k[1]}': v['immutable_digest'] for k, v in r.nets.items()},
              'teacher_digest': r.teacher_digest, 'deployment_sha256': r.deployment_sha, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-D development (prospective) and exposed dev512 continuity'}
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
