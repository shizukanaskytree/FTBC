"""E14 exact per-step FTBC objective screen (EXPERIMENT_PLAN.md). One process and one Slurm-allocated A100 per job.

Reuses the frozen E13 runner (E0 construction, GPU guard, prefix capture, the M0 count-mean solver with its full-network
replay, released FTBC through the AST-verified E1 export, E13 diagnostics, strict reload). New here: the fresh development
role, the per-step sequential solver arms (step_solver.py), fit-subset switching for the data-matched arms, the FTBC gain
arm, an evaluation-batch perturbation control, a per-step objective diagnostic and descriptive E13 reproduction anchors.
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step_solver as ps
if not __debug__:
    raise SystemExit('E14 acceptance-critical checks use assert; do not run with -O')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def bias_hash(biases, names, T):
    return hashlib.sha256(b''.join(biases[n][t].detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
                                   for n in names for t in range(T))).hexdigest()


def arm_names(c):
    L, S = c['fit_count'], c['fit_small']
    return {'mL': f'M0-{L}', 'pL': f'P0-{L}', 'mS': f'M0-{S}', 'pS': f'P0-{S}'}


def build(c):
    E13 = load('e14_e13', c['e13_runner'])
    ss = E13.ss

    class Run(E13.Run):
        def __init__(self, c):
            super().__init__(c)
            assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
            fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']
            used = {int(r['index']) for role in self.roles.values() for r in role}
            used_sha = {r['sha256'] for role in self.roles.values() for r in role}
            ids = [int(r['index']) for r in fresh]
            assert len(fresh) >= c['fresh_count'] and len(set(ids)) == len(ids)
            assert not set(ids) & used and not {r['sha256'] for r in fresh} & used_sha, 'fresh development overlaps E0 roles'
            self.rows['fresh'] = fresh[:c['fresh_count']]
            assert c['fit_small'] == 3 * c['ftbc_batch'] and self.rows['ftbc'][:c['fit_small']] == self.rows['fit'][:c['fit_small']]
            assert c['ftbc_arms'][0] == ['F0-n3', 3, c['ftbc_alpha']] and all(c['ftbc_batch'] * n <= len(self.rows['ftbc']) for _, n, _ in c['ftbc_arms'])
            assert tuple(c['score_roles']) == ('fresh', 'development')
            self.step_diag = {}; self.repro = {}; self.bias_hashes = {}; self.targets_by_n = {}

        def free(self):
            torch.cuda.empty_cache()

        def use_fit(self, n):
            rows = self.rows['fit'][:n]; self.fit_n = n
            self.fit_batches = list(self.e0.loader(rows, self.parent, self.c['solver_batch']))
            assert torch.equal(torch.cat([b[2] for b in self.fit_batches]), torch.tensor([int(v['index']) for v in rows]))
            if n not in self.targets_by_n:
                self.targets_by_n[n] = self.teacher_means()
                self.e0.atomic_torch(self.out / f'teacher-targets-{n}.pt', {'means': {k: v.cpu() for k, v in self.targets_by_n[n].items()},
                                                                            'batch': self.c['solver_batch'], 'ids': torch.tensor([int(v['index']) for v in rows])})
            self.targets = self.targets_by_n[n]

        @torch.no_grad()
        def capture(self, T, name):
            """E13 capture, except that the cache is sized by the active fit subset (E13 always used c['fit_count'])."""
            N = self.net(T); m = N['modules'][name]; cache = None; off = 0; n_fit = self.fit_n
            def hook(_, args, y):
                nonlocal cache, off
                x = args[0].view(T, -1, *args[0].shape[1:])
                if cache is None:
                    cache = torch.empty((T, n_fit, *x.shape[2:]), device=x.device, dtype=x.dtype)
                cache[:, off:off + x.shape[1]].copy_(x); off += x.shape[1]
                raise E13.PrefixDone()
            h = m.register_forward_hook(hook)
            try:
                for x, _, _ in self.fit_batches:
                    try: self.e0.logits(N['snn'], x.cuda(), N['parent'], True)
                    except E13.PrefixDone: pass
                    else: raise AssertionError('prefix hook did not fire')
            finally:
                h.remove()
            assert off == n_fit
            return cache

        @torch.no_grad()
        def replay_check(self, arm, T, info):
            """The E13 full-network replay: every layer's channel counts on the active fit subset must equal the solver records."""
            N = self.net(T); counts, hs, order = {}, [], []
            def hook(n):
                def f(mod, args, y):
                    cnt = torch.count_nonzero(y.view(T, -1, *y.shape[1:]), dim=(0, 1, 3, 4)).cpu()
                    if n not in counts: order.append(n)
                    counts[n] = counts.get(n, 0) + cnt
                return f
            for n in N['names']: hs.append(N['modules'][n].register_forward_hook(hook(n)))
            try:
                for x, _, _ in self.fit_batches: self.e0.logits(N['snn'], x.cuda(), N['parent'], True)
            finally:
                for h in hs: h.remove()
            assert order == N['names'], 'interface order differs from forward traversal'
            for n in N['names']:
                assert torch.equal(counts[n], info[n]['final_counts']), f'{arm}: full-network replay differs at {n}'

        @torch.no_grad()
        def step_calibrate(self, arm, T):
            """P0: per-step sequential exact solve of the FTBC objective, layer by layer in forward order (step_solver.py)."""
            N = self.net(T); targets = self.targets; biases = self.zero_biases(T); self.set_biases(T, biases); info = {}
            bound = self.c['bound_ratio']
            for i, n in enumerate(N['names']):
                m = N['modules'][n]; thre = m.thresh.data; th = float(thre); t_start = time.perf_counter()
                x = self.capture(T, n); C = x.shape[2]
                b, sinfo, total = ps.solve_per_step(x, thre, targets[n], steps=self.c['bisection_steps'], bound=bound)
                assert b.shape == (T, C) and b.dtype == torch.float32 and torch.isfinite(b).all() and bool((b.abs() <= bound * th).all())
                assert torch.equal(ss.replay(x, b, thre), total), f'{arm}: per-step solver totals differ from the IF replay at {n}'
                final_counts = ss.channel_counts(total).cpu()
                m.time_based_bias = [b[t].clone() for t in range(T)]; biases[n] = [b[t].cpu() for t in range(T)]
                info[n] = {'steps': sinfo['steps'], 'target_mean': sinfo['target_mean'], 'theta': th, 'b': b.cpu(),
                           'observations_per_channel_per_step': sinfo['observations_per_channel_per_step'], 'final_counts': final_counts,
                           'seconds': time.perf_counter() - t_start, 'cache_bytes': x.numel() * 4}
                del x, total
                self.e0.atomic_torch(self.out / f'solver-{arm}.pt', info)
                self.progress('calibrated_layer', arm=arm, index=i + 1, name=n,
                              mean_abs_step_residual=sum(float(s['normalized_residual'].abs().mean()) for s in sinfo['steps']) / T)
            self.replay_check(arm, T, info)
            self.solver[arm] = {'full_network_replay_equal': True, 'per_step': True, 'fit_images': self.fit_n, 'layers': info}
            return biases

        def count_calibrate_fit(self, arm, T):
            """M0 through the frozen E13 count_calibrate (time-shared beta from zero offsets) on the active fit subset."""
            b = self.count_calibrate(arm, T, torch.zeros(T, dtype=torch.float64), False)
            self.solver[arm]['fit_images'] = self.fit_n
            return b

        def ftbc_with_alpha(self, arm, T, n_batches, init_biases, alpha):
            """Frozen E13 ftbc_calibrate with the released update gain alpha set for this call only."""
            saved = self.c['ftbc_alpha']; self.c['ftbc_alpha'] = alpha
            try:
                b = self.ftbc_calibrate(arm, T, n_batches, init_biases)
            finally:
                self.c['ftbc_alpha'] = saved
            assert self.ftbc[arm]['alpha'] == alpha and self.ftbc[arm]['batches'] == n_batches
            return b

        @torch.no_grad()
        def score(self, arm, T, biases, detail):
            """E13 score() on the fresh and development roles; the same strict-reload check on four fit images."""
            N = self.net(T); self.set_biases(T, biases)
            p = {role: self.e0.evaluate(N['snn'], self.rows[role], N['parent'], True) for role in self.c['score_roles']}
            before = self.e0.logits(N['snn'], self.test_images, N['parent'], True).cpu()
            saved = {'state_dict': {k: v.detach().cpu() for k, v in N['snn'].state_dict().items()},
                     'runtime_biases': self.e0.runtime_biases(N['snn'], N['parent'], self.classes), 'arm': arm, 'T': T, 'detail': detail,
                     'plain_deployment_sha256': sha(self.c['plain_deployment'])}
            path = self.out / f'bias-{arm}.pt'; self.e0.atomic_torch(path, saved)
            re = torch.load(path, map_location='cpu', weights_only=False)
            pT = N['parent']; proxy = self.e0.make_proxy(self.teacher, pT, self.classes); proxy.load_state_dict(self.proxy_state, strict=True)
            restored = self.e0.make_snn(proxy, pT, self.classes); restored.load_state_dict(re['state_dict'], strict=True)
            mods = dict(restored.named_modules())
            for n, vals in re['runtime_biases'].items():
                mods[n].time_based_bias = [v.cuda() for v in vals]
            after = self.e0.logits(restored, self.test_images, pT, True).cpu()
            assert torch.equal(before, after), 'strict reload changed logits: ' + arm
            assert self.e0.state_digest(N['snn']) == N['immutable_digest'], 'non-bias SNN state changed'
            del restored, proxy
            h = bias_hash(re['runtime_biases'], N['names'], T); assert h == bias_hash(biases, N['names'], T); self.bias_hashes[arm] = h
            self.predictions[arm] = p
            self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'strict_reload': True, 'detail': detail,
                                 'bias_sha256': h}
            self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
            self.progress('scored', arm=arm, T=T, **{role: p[role]['metrics'] for role in p})

        @torch.no_grad()
        def score_perturbed(self, arm, T, batch):
            """Numerical-perturbation control: the scored arm's biases, unchanged, evaluated with another evaluation batch size."""
            N = self.net(T); current = {n: N['modules'][n].time_based_bias for n in N['names']}
            assert bias_hash(current, N['names'], T) == self.bias_hashes[arm], 'perturbation must reuse the scored biases'
            pT = dict(N['parent']); pT['eval_batch_size'] = batch
            p = {role: self.e0.evaluate(N['snn'], self.rows[role], pT, True) for role in self.c['score_roles']}
            name = f'{arm}@b{batch}'; self.predictions[name] = p
            self.records[name] = {'arm': name, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'perturbation_of': arm,
                                  'eval_batch_size': batch, 'bias_sha256': self.bias_hashes[arm]}
            self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
            self.progress('scored', arm=name, T=T, **{role: p[role]['metrics'] for role in p})

        @torch.no_grad()
        def score_ann(self, arm, model, T):
            p = {role: self.e0.evaluate(model, self.rows[role], self.parent_T(T), False) for role in self.c['score_roles']}
            self.predictions[arm] = p; self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'ann': True}
            self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
            self.progress('scored', arm=arm, T=T, **{role: p[role]['metrics'] for role in p})

        @torch.no_grad()
        def teacher_means_on(self, rows, batch):
            totals, sizes, hs = {}, {}, []
            N = self.net(4); tm = dict(self.teacher.named_modules())
            def hook(n):
                def f(m, args, y):
                    totals[n] = totals.get(n, 0) + y.double().sum((0, 2, 3)); sizes[n] = sizes.get(n, 0) + y.numel() // y.shape[1]
                return f
            for n in N['names']: hs.append(tm[n].register_forward_hook(hook(n)))
            try:
                for x, _, _ in self.e0.loader(rows, self.parent, batch): self.e0.logits(self.teacher, x.cuda(), self.parent)
            finally:
                for h in hs: h.remove()
            return {n: totals[n] / sizes[n] for n in N['names']}

        @torch.no_grad()
        def step_objective(self, arm, T):
            """Out-of-sample FTBC objective on the diagnostic images: per layer and step, channel mean output against the teacher channel mean."""
            N = self.net(T)
            if not hasattr(self, 'diag_targets'):
                self.diag_targets = self.teacher_means_on(self.rows['diagnostic'], 8)
            sums, cnt, hs = {}, {}, []
            def hook(n):
                def f(m, args, y):
                    v = y.view(T, -1, *y.shape[1:]).double()
                    sums[n] = sums.get(n, 0) + v.sum((1, 3, 4)); cnt[n] = cnt.get(n, 0) + v[0].numel() // v.shape[2]
                return f
            for n in N['names']: hs.append(N['modules'][n].register_forward_hook(hook(n)))
            try:
                for x, _, _ in self.e0.loader(self.rows['diagnostic'], self.parent, 8): self.e0.logits(N['snn'], x.cuda(), N['parent'], True)
            finally:
                for h in hs: h.remove()
            out = {}
            for n in N['names']:
                th = float(N['modules'][n].thresh.data); m_t = sums[n] / cnt[n]; a = self.diag_targets[n]
                out[n] = {'step_abs_residual_over_theta': ((m_t - a.view(1, -1)).abs().mean(1) / th).tolist(),
                          'total_abs_residual_over_theta': float((m_t.mean(0) - a).abs().mean() / th),
                          'step_mean_over_theta': (m_t.mean(1) / th).tolist(), 'teacher_mean_over_theta': float(a.mean() / th)}
            self.step_diag[arm] = out; self.e0.atomic_json(self.out / 'step-objective.json', self.step_diag)
            self.progress('step_objective', arm=arm)

        @torch.no_grad()
        def e13_anchors(self, names):
            """Descriptive reproduction of E13 main on dev512 (bitwise identity is expected only on the same card). Non-fatal by design."""
            try:
                self._e13_anchors(names)
            except Exception:
                self.repro = {'error': traceback.format_exc()}; self.e0.atomic_json(self.out / 'e13-anchors.json', self.repro)
                self.progress('e13_anchors_failed')

        @torch.no_grad()
        def _e13_anchors(self, names):
            d = Path(self.c['e13_output']); old = torch.load(d / 'predictions.pt', map_location='cpu', weights_only=False)
            rep = {'e13_gpu_uuid': json.loads((d / 'hardware.json').read_text())['gpu_uuid']}
            rep['same_card_as_e13'] = rep['e13_gpu_uuid'] == self.hardware['gpu_uuid']
            for arm, key in [('teacher', 'teacher'), ('twin-T4', 'twin-T4'), ('S0', 'S0'), (names['mL'], 'M0'), ('F0-n3', 'F0-n3')]:
                a = self.predictions[arm]['development']; o = old[key]['development']
                assert torch.equal(a['ids'], o['ids']) and torch.equal(a['labels'], o['labels'])
                rep[arm] = {'e13_key': key, 'logits_bitwise_equal': bool(torch.equal(a['logits'], o['logits'])),
                            'max_logit_delta': float((a['logits'] - o['logits']).abs().max()),
                            'prediction_disagreements': int((a['predictions'] != o['predictions']).sum())}
            for arm, key in [(names['mL'], 'M0'), ('F0-n3', 'F0-n3')]:
                ob = torch.load(d / f'bias-{key}.pt', map_location='cpu', weights_only=False)['runtime_biases']
                nb = torch.load(self.out / f'bias-{arm}.pt', map_location='cpu', weights_only=False)['runtime_biases']
                assert set(ob) == set(nb)
                rep[arm]['biases_bitwise_equal'] = all(torch.equal(x, y) for n in ob for x, y in zip(ob[n], nb[n]))
                rep[arm]['max_bias_delta'] = max(float((x - y).abs().max()) for n in ob for x, y in zip(ob[n], nb[n]))
            ot = torch.load(d / 'teacher-targets.pt', map_location='cpu', weights_only=False)
            assert ot['ids'].tolist() == [int(r['index']) for r in self.rows['fit'][:self.c['fit_count']]]
            rep['teacher_targets_bitwise_equal'] = all(torch.equal(ot['means'][n], self.targets_by_n[self.c['fit_count']][n].cpu()) for n in ot['means'])
            self.repro = rep; self.e0.atomic_json(self.out / 'e13-anchors.json', rep)
            self.progress('e13_anchors', same_card=rep['same_card_as_e13'])

    return E13, ss, Run


@torch.no_grad()
def run(c):
    E13, ss, Run = build(c); r = Run(c); names = arm_names(c); stages = c['stages']
    assert set(stages) <= {'main', 'extra'} and stages
    if 'main' in stages:
        T = 4
        r.score_ann('teacher', r.teacher, T); r.score_ann('twin-T4', r.net(T)['proxy'], T)
        zero = r.zero_biases(T)
        r.score('S0', T, zero, {'schedule': 'IF v0=theta/2'}); r.diagnostics('S0', T); r.step_objective('S0', T)
        r.anchors()
        r.use_fit(c['fit_count'])
        r.free(); b = r.count_calibrate_fit(names['mL'], T)
        r.score(names['mL'], T, b, {'solver': 'M0 count mean (time-shared beta)', 'fit_images': c['fit_count']})
        r.diagnostics(names['mL'], T); r.step_objective(names['mL'], T); r.score_perturbed(names['mL'], T, c['perturb_batch'])
        r.free(); b = r.step_calibrate(names['pL'], T)
        r.score(names['pL'], T, b, {'solver': 'P0 per-step sequential', 'fit_images': c['fit_count']})
        r.diagnostics(names['pL'], T); r.step_objective(names['pL'], T); r.score_perturbed(names['pL'], T, c['perturb_batch'])
        r.use_fit(c['fit_small'])
        r.free(); b = r.count_calibrate_fit(names['mS'], T)
        r.score(names['mS'], T, b, {'solver': 'M0 count mean (time-shared beta)', 'fit_images': c['fit_small']})
        r.free(); b = r.step_calibrate(names['pS'], T)
        r.score(names['pS'], T, b, {'solver': 'P0 per-step sequential', 'fit_images': c['fit_small']})
        for arm, n_batches, alpha in c['ftbc_arms']:      # registered FTBC variants; F0-n3 first (update rule, alpha 0.4 and n=3 of the release)
            r.free(); b = r.ftbc_with_alpha(arm, T, n_batches, zero, alpha)
            r.score(arm, T, b, {'ftbc': r.ftbc[arm]})
            if arm == 'F0-n3':
                r.diagnostics(arm, T); r.step_objective(arm, T)
        if c['e13_anchor']:
            r.e13_anchors(names)
    if 'extra' in stages:
        r.use_fit(c['fit_count'])
        for Tx in c['extra_T']:
            r.score_ann(f'twin-T{Tx}', r.net(Tx)['proxy'], Tx)
            r.score(f'S0-T{Tx}', Tx, r.zero_biases(Tx), {'schedule': 'IF v0=theta/2'})
            r.free(); b = r.count_calibrate_fit(f"{names['mL']}-T{Tx}", Tx)
            r.score(f"{names['mL']}-T{Tx}", Tx, b, {'solver': 'M0 count mean (time-shared beta)', 'fit_images': c['fit_count']})
            r.free(); b = r.step_calibrate(f"{names['pL']}-T{Tx}", Tx)
            r.score(f"{names['pL']}-T{Tx}", Tx, b, {'solver': 'P0 per-step sequential', 'fit_images': c['fit_count']})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for Tx, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    torch.cuda.synchronize()
    result = {'status': 'completed', 'stages': stages, 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'twin_unchanged': True, 'proxy_digests': {str(k): v['proxy_digest'] for k, v in r.nets.items()},
              'diagnostic_ids': getattr(r, 'diag_ids', None), 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()},
              'bias_sha256': r.bias_hashes, 'teacher_targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest()
                                                                       for n, t in r.targets_by_n.items()},
              'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0, 'max_memory_allocated': torch.cuda.max_memory_allocated(),
              'max_memory_reserved': torch.cuda.max_memory_reserved(), 'anchors_e0': r.anchor, 'anchors_e13': r.repro,
              'data_status': 'fresh development (prospective) and exposed dev512 continuity'}
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
