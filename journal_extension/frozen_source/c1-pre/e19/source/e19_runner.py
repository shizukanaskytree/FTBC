"""E19 mechanism screen (EXPERIMENT_PLAN.md): where P0's gain over the released FTBC update comes from, on serial R34 Q8 (E0 plain).
One process and one Slurm-allocated A100 per job.

Reuses the frozen E17 runner (and through it E16-E13) unchanged: plain state, calibration draws from the configured manifest with the
disjointness checks, P0 (step_calibrate), the released FTBC update with the counting wrapper and an explicit ANN (ftbc_batches_ann),
registered batches, prefix capture, IF replay checks and strict-reload scoring. New here: the FA/FD ablation variants (ftbc_variants.py)
swapped into the released bias_corr_model loop for one call and restored afterwards, the ECCV-schedule batches, and the deployed per-step
channel-mean residual readout (section 4).
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E19 acceptance-critical checks use assert; do not run with -O')
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]
REGISTERED = [('F0', 'released', 1), ('FA', 'FA', 1), ('FD', 'FD', 1), ('FD10', 'FD', 10)]   # (arm, variant, passes) at T=4 on Cks


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
    E17 = load('e19_e17', c['e17_runner'])
    E13, E14, E15, E16, ss, ps, ce, Run17 = E17.build(c)
    V = load('e19_variants', c['ftbc_variants'])

    class Run(Run17):
        def __init__(self, c):
            super().__init__(c)                                # E17: plain state, draws, disjointness from every earlier role and split
            assert c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n'] and c['eccv_n'] == c['eccv_batch'] * c['eccv_batches'] < c['draw_n']
            assert c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2 and c['ftbc_alpha'] == 0.4 and c['diag_n'] <= c['fresh_count']
            self.variants = V.make(self.inherited); self.release_step = self.inherited.bias_corr_step_by_step; self.diag = {}

        def ftbc_batches(self, rows, batch_size, n_batches):
            out = [(images, torch.full(labels.shape, -1, dtype=labels.dtype)) for images, labels, _ in self.e0.loader(rows, self.parent, batch_size)]
            assert len(out) == n_batches and all(b[0].shape[0] == batch_size for b in out) and sum(b[0].shape[0] for b in out) == len(rows)
            return out

        def ftbc_run(self, arm, T, rows, variant, recipe, passes):
            """The released loop (bias_corr_model) with the released per-interface update or an ablation variant swapped in for this call;
            `passes` repeats the same batches in the same order. Counting wrapper, zero start and state checks are E16 ftbc_batches_ann."""
            if recipe == 'registered':
                alpha, batches = self.c['ftbc_alpha'], self.registered_batches(rows)
            else:
                assert recipe == 'eccv-schedule'; alpha, batches = self.c['eccv_alpha'], self.ftbc_batches(rows, self.c['eccv_batch'], self.c['eccv_batches'])
            assert self.inherited.bias_corr_step_by_step is self.release_step
            if variant != 'released':
                self.inherited.bias_corr_step_by_step = self.variants[variant]
            try:
                b = self.ftbc_batches_ann(arm, T, batches * passes, alpha, [int(x['index']) for x in rows], recipe, self.teacher, 'teacher',
                                          extra={'variant': variant, 'passes': passes, 'batches_per_pass': len(batches)})
            finally:
                self.inherited.bias_corr_step_by_step = self.release_step
            return b

        @torch.no_grad()
        def deployed_residuals(self, T, biases):
            """Per interface and step, |m - mean_t| / theta of the deployed SNN with `biases` on the active rows; m = active target
            (teacher channel means on exactly these rows), mean_t = channel mean of the IF output (theta * spikes) at step t."""
            N = self.net(T); self.set_biases(T, biases); sums, n_obs, hs = {}, {}, []
            def hook(n):
                def f(mod, args, y):
                    z = y.view(T, -1, *y.shape[1:])
                    sums[n] = sums.get(n, 0) + z.double().sum((1,) + tuple(range(3, z.dim()))); n_obs[n] = n_obs.get(n, 0) + z[0].numel() // z.shape[2]
                return f
            for n in N['names']: hs.append(N['modules'][n].register_forward_hook(hook(n)))
            try:
                for x, _, _ in self.fit_batches: self.e0.logits(N['snn'], x.cuda(), N['parent'], True)
            finally:
                for h in hs: h.remove()
            out = {}
            for n in N['names']:
                th = float(N['modules'][n].thresh.data); mean_t = sums[n] / n_obs[n]
                r = (mean_t - self.targets[n].to(device=mean_t.device, dtype=torch.float64).view(1, -1)).abs() / th
                assert r.shape == (T, mean_t.shape[1]) and torch.isfinite(r).all()
                out[n] = {'mean_by_step': r.mean(1).cpu().tolist(), 'max_by_step': r.max(1).values.cpu().tolist()}
            return {'target_key': self.target_key, 'rows': self.fit_n, 'layers': out, 'bias_sha256': E14.bias_hash(biases, N['names'], T),
                    'mean': sum(sum(v['mean_by_step']) for v in out.values()) / (len(out) * T), 'max': max(max(v['max_by_step']) for v in out.values())}

    return E13, E14, E15, E16, ss, ps, ce, V, Run


@torch.no_grad()
def run(c):
    E13, E14, E15, E16, ss, ps, ce, V, Run = build(c); r = Run(c); small, ne = c['small_n'], c['eccv_n']
    def score(arm, T, b, detail):
        r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
    def p0(arm, T, key):
        r.free(); b = r.step_calibrate(arm, T)
        r.solver[arm]['target_key'] = r.target_key; r.solver[arm]['state'] = r.variant; r.solver[arm]['T'] = T
        score(arm, T, b, {'solver': 'P0', 'target_key': r.target_key, 'rows': key}); return b
    def ftbc(arm, T, rows, variant, recipe, passes):
        r.free(); b = r.ftbc_run(arm, T, rows, variant, recipe, passes); score(arm, T, b, {'ftbc': r.ftbc[arm]}); return b
    r.score_ann('teacher', r.teacher, 4)
    zero = {}
    for T in TS:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T); zero[T] = r.zero_biases(T); score(f'S0-T{T}', T, zero[T], {'schedule': 'IF v0=theta/2'})
    held = r.rows['fresh'][:c['diag_n']]
    for d, rows in r.draws.items():
        arms = {}
        r.use_rows_target(f'{d}s', rows[:small], 'teacher')
        for name, variant, passes in REGISTERED:
            arms[f'{name}-T4-{d}s'] = (4, f'{d}s', ftbc(f'{name}-T4-{d}s', 4, rows[:small], variant, 'registered', passes))
        arms[f'P0-T4-{d}s'] = (4, f'{d}s', p0(f'P0-T4-{d}s', 4, f'{d}s'))
        arms[f'F0-T8-{d}s'] = (8, f'{d}s', ftbc(f'F0-T8-{d}s', 8, rows[:small], 'released', 'registered', 1))
        arms[f'P0-T8-{d}s'] = (8, f'{d}s', p0(f'P0-T8-{d}s', 8, f'{d}s'))
        r.use_rows_target(f'{d}e', rows[:ne], 'teacher')
        arms[f'FE-T4-{d}e'] = (4, f'{d}e', ftbc(f'FE-T4-{d}e', 4, rows[:ne], 'released', 'eccv-schedule', 1))
        arms[f'P0-T4-{d}e'] = (4, f'{d}e', p0(f'P0-T4-{d}e', 4, f'{d}e'))
        # Section 4 readout with each arm's saved biases: on its own calibration rows, then on the held-out rows.
        for key, n in [(f'{d}s', small), (f'{d}e', ne)]:
            r.use_rows_target(key, rows[:n], 'teacher')
            for arm, (T, k, b) in arms.items():
                if k == key: r.diag[arm] = {'calibration': r.deployed_residuals(T, b)}
        r.use_rows_target('H', held, 'teacher')
        for arm, (T, k, b) in arms.items():
            r.diag[arm]['held_out'] = r.deployed_residuals(T, b)
        r.e0.atomic_json(r.out / 'diagnostics.json', r.diag); r.progress('diagnostics', draw=d)
    r.use_rows_target('H', held, 'teacher')
    for T in TS:
        r.diag[f'S0-T{T}'] = {'held_out': r.deployed_residuals(T, zero[T])}
    r.e0.atomic_json(r.out / 'diagnostics.json', r.diag)
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.inherited.bias_corr_step_by_step is r.release_step
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for key, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    assert r.clear_release_hooks() == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True,
              'proxy_digests': {f'{k[0]}-T{k[1]}': v['proxy_digest'] for k, v in r.nets.items()}, 'snn_digests': {f'{k[0]}-T{k[1]}': v['immutable_digest'] for k, v in r.nets.items()},
              'teacher_digest': r.teacher_digest, 'deployment_sha256': r.deployment_sha, 'all_strict_reloads_passed': True, 'release_update_restored': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'held_out': [int(x['index']) for x in held],
              'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-F development (prospective) and exposed dev512 continuity'}
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
