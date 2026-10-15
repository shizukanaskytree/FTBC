"""E16 two-stage serial screen (EXPERIMENT_PLAN.md): P0 on the E0 BN-adapted (spatial) deployment against P0 alone and BN->FTBC.
One process and one Slurm-allocated A100 per job.

Reuses the frozen E15 runner (and through it E14/E13): prefix capture, M0, P0 with per-layer and full-network replay, strict-reload
scoring on the fresh and development roles, registered and release-default FTBC via ftbc_batches/release_batches with hook cleanup.
New here: a deployment-state switch (plain or spatial) behind net(T); twin targets PT(phi) (channel means of the active state's T=4
proxy) and descriptive A8(phi) targets (the same proxy parameters on the Q8 grid) next to teacher targets A8(phi0); the released FTBC
update with its ANN argument made explicit (spatial twin or Q8 teacher, ftbc_batches_ann); calibration/release rows from the E16
manifest (outside the E0 fit pool that trained the BN state).
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
import PIL
import torchvision
if not __debug__:
    raise SystemExit('E16 acceptance-critical checks use assert; do not run with -O')
STAGES = ('main', 'release')
CAL, REL = ['C1', 'C2', 'C3'], ['Q1', 'Q2', 'Q3']


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
    E15 = load('e16_e15', c['e15_runner'])
    E13, E14, ss, ps, Run15 = E15.build(c)
    Run14 = Run15.__mro__[1]

    class Run(Run15):
        def __init__(self, c):
            Run14.__init__(self, c)                       # E14/E13 initialisation; the E15 fit-pool draws are not used in E16
            old = [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
            assert not {int(r['index']) for r in self.rows['fresh']} & {int(r['index']) for r in old}, 'fresh-C overlaps an earlier fresh split (ID)'
            assert not {r['sha256'] for r in self.rows['fresh']} & {r['sha256'] for r in old}, 'fresh-C overlaps an earlier fresh split (content)'
            assert sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
            cal = json.loads(Path(c['calibration_manifest']).read_text())
            nr = c['release_n'] * c['release_batch']
            assert all(len(cal[k]) >= c['draw_n'] for k in CAL) and all(len(cal[k]) >= nr for k in REL)
            self.draws = {k: cal[k][:c['draw_n']] for k in CAL}; self.release_sets = {k: cal[k][:nr] for k in REL}   # sanity uses prefixes
            used = [r for role in self.roles.values() for r in role] + old + self.rows['fresh']
            allc = [r for k in CAL + REL for r in cal[k]]                                                   # the whole manifest, not only the prefixes
            assert len({int(r['index']) for r in allc}) == len(allc) and not {int(r['index']) for r in allc} & {int(r['index']) for r in used}
            assert not {r['sha256'] for r in allc} & {r['sha256'] for r in used}, 'calibration rows overlap earlier roles (content)'
            assert all(len(self.draws[k]) == c['draw_n'] for k in CAL) and all(len(self.release_sets[k]) == c['release_n'] * c['release_batch'] for k in REL)
            spatial = torch.load(c['spatial_deployment'], map_location='cpu', weights_only=False)
            assert all(not torch.count_nonzero(x) for v in spatial['runtime_biases'].values() for x in v) and len(spatial['runtime_biases']) == 33
            sproxy = torch.load(c['spatial_proxy'], map_location='cpu', weights_only=True)
            assert sproxy['proxy_quantization_level'] == 4 and sproxy['family'] == 'serial'
            self.states = {'plain': {'proxy': self.proxy_state, 'deploy': self.deploy}, 'spatial': {'proxy': sproxy['state_dict'], 'deploy': spatial}}
            self.deployment_sha = {'plain': sha(c['plain_deployment']), 'spatial': sha(c['spatial_deployment'])}
            self.variant = 'plain'; self.nets = {}; self.grid_proxies = {}; self.release_record = {}

        def net(self, T):
            """E13 net(T) keyed by (deployment state, T)."""
            key = (self.variant, T)
            if key not in self.nets:
                pT = self.parent_T(T); st = self.states[self.variant]
                proxy = self.e0.make_proxy(self.teacher, pT, self.classes); proxy.load_state_dict(st['proxy'], strict=True)
                snn = self.e0.make_snn(proxy, pT, self.classes); snn.load_state_dict(st['deploy']['state_dict'], strict=True)
                names = [n for n, m in snn.named_modules() if isinstance(m, self.classes['spike'])]
                assert len(names) == 33
                proxy = proxy.cuda().eval()
                self.nets[key] = {'proxy': proxy, 'snn': snn, 'names': names, 'modules': dict(snn.named_modules()),
                                  'immutable_digest': self.e0.state_digest(snn), 'proxy_digest': self.e0.state_digest(proxy), 'parent': pT}
            return self.nets[key]

        def grid_proxy(self, L):
            """The active state's proxy parameters on the QCFS grid L, without an SNN (A8(phi) for L=8 in the spatial state,
            compatibility-prior-art.md:15). Same state_dict as the T=4 proxy; only the activation level L differs."""
            key = (self.variant, L)
            if key not in self.grid_proxies:
                pL = self.parent_T(L); proxy = self.e0.make_proxy(self.teacher, pL, self.classes)
                proxy.load_state_dict(self.states[self.variant]['proxy'], strict=True); proxy = proxy.cuda().eval()
                self.grid_proxies[key] = {'proxy': proxy, 'proxy_digest': self.e0.state_digest(proxy), 'parent': pL}
            return self.grid_proxies[key]

        def levels(self, model):
            return sorted({int(m.L) for m in model.modules() if isinstance(m, self.classes['activation'])})

        @torch.no_grad()
        def proxy_means(self, proxy, parent):
            """Channel means of a QCFS network's activation outputs at the 33 interfaces, on the active fit rows (batch solver_batch);
            same reduction as E13 teacher_means."""
            totals, sizes, hs = {}, {}, []
            names = self.net(4)['names']; pm = dict(proxy.named_modules())
            def hook(n):
                def f(m, args, y):
                    totals[n] = totals.get(n, 0) + y.double().sum((0, 2, 3)); sizes[n] = sizes.get(n, 0) + y.numel() // y.shape[1]
                return f
            for n in names: hs.append(pm[n].register_forward_hook(hook(n)))
            try:
                for x, _, _ in self.fit_batches: self.e0.logits(proxy, x.cuda(), parent)
            finally:
                for h in hs: h.remove()
            return {n: totals[n] / sizes[n] for n in names}

        def use_rows_target(self, key, rows, target):
            """Active fit rows and solver targets: 'teacher' (Q8 source A8(phi0), state-independent), 'twin' (active state's T=4 proxy,
            PT(phi)) or 'a8' (active state's proxy parameters on the Q8 grid, A8(phi); descriptive). The target file records the digest
            and the read-back activation level of the network whose means were taken."""
            assert target in ('teacher', 'twin', 'a8')
            self.fit_n = len(rows)
            self.fit_batches = list(self.e0.loader(rows, self.parent, self.c['solver_batch']))
            assert torch.equal(torch.cat([b[2] for b in self.fit_batches]), torch.tensor([int(v['index']) for v in rows]))
            tkey = f'{key}-teacher' if target == 'teacher' else f'{key}-{target}-{self.variant}'
            if tkey not in self.targets_by_n:
                if target == 'teacher':
                    model, digest = self.teacher, self.teacher_digest; self.targets_by_n[tkey] = self.teacher_means()
                else:
                    G = self.net(4) if target == 'twin' else self.grid_proxy(8); model, digest = G['proxy'], G['proxy_digest']
                    self.targets_by_n[tkey] = self.proxy_means(G['proxy'], G['parent'])
                self.e0.atomic_torch(self.out / f'targets-{tkey}.pt', {'means': {k: v.cpu() for k, v in self.targets_by_n[tkey].items()}, 'target': target,
                                                                       'state': None if target == 'teacher' else self.variant, 'batch': self.c['solver_batch'],
                                                                       'network_digest': digest, 'level': self.levels(model),
                                                                       'ids': torch.tensor([int(v['index']) for v in rows])})
            self.targets = self.targets_by_n[tkey]; self.target_key = tkey

        def ftbc_batches_ann(self, arm, T, batches, alpha, images, recipe, ann, ann_tag, extra=None):
            """E15 ftbc_batches with the released update's ANN argument made explicit: the Q8 source teacher ('teacher') or the active
            state's T-level proxy ('twin-<state>'). Same counting wrapper (the ANN forward calls are counted on that ANN), zero start,
            unchanged-state checks on both the teacher and the ANN used."""
            N = self.net(T); zero = self.zero_biases(T); self.set_biases(T, zero)
            init_hash = hashlib.sha256(b''.join(zero[n][t].to(torch.float32).contiguous().numpy().tobytes() for n in N['names'] for t in range(T))).hexdigest()
            n_batches = len(batches)
            count = {'ann': 0, 'snn': 0, 'updates': {}}
            hs = [ann.register_forward_pre_hook(lambda *a: count.__setitem__('ann', count['ann'] + 1)),
                  N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
            ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step
            def counted(*args, **kw):
                key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
            self.inherited.bias_corr_step_by_step = counted
            ann_before = self.e0.state_digest(ann); teacher_before = self.e0.state_digest(self.teacher)
            try:
                with torch.no_grad():
                    self.inherited.bias_corr_model(ann, N['snn'], T, batches, curr_t_alpha=alpha, num_cali_sample_batches=n_batches)
            finally:
                self.inherited.bias_corr_step_by_step = original
                for h in hs: h.remove()
            assert count['ann'] == 33 * n_batches and count['snn'] == 33 * n_batches, count
            assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
            assert self.e0.state_digest(ann) == ann_before and self.e0.state_digest(self.teacher) == teacher_before == self.teacher_digest
            biases = {n: [b.detach().cpu().clone() for b in N['modules'][n].time_based_bias] for n in N['names']}
            assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
            self.ftbc[arm] = {'batches': n_batches, 'alpha': alpha, 'batch_size': int(batches[0][0].shape[0]), 'calls': count, 'init_bias_sha256': init_hash,
                              'images': images, 'recipe': recipe, 'inputs_sha256': E15.tensor_hash([b[0] for b in batches]), 'ann': ann_tag, 'ann_digest': ann_before,
                              'ann_level': self.levels(ann), 'state': self.variant, **(extra or {}),
                              'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
            return biases

        def clear_release_hooks(self):
            """E15 hook cleanup extended to the proxies, which can now be the released update's ANN."""
            dropped = 0
            for model in [self.teacher] + [N['snn'] for N in self.nets.values()] + [N['proxy'] for N in list(self.nets.values()) + list(self.grid_proxies.values())]:
                for m in model.modules():
                    for key, hook in list(m._forward_hooks.items()):
                        if isinstance(hook, self.inherited.ActivationSaverHookEachTimeStep):
                            del m._forward_hooks[key]; dropped += 1
            return dropped

    return E13, E14, E15, ss, ps, Run


@torch.no_grad()
def run(c):
    stages = c['stages']; assert stages and set(stages) <= set(STAGES)
    if 'release' in stages and c.get('expandable_segments'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # before the first CUDA allocation; no numerical effect
    E13, E14, E15, ss, ps, Run = build(c); r = Run(c); small = c['small_n']; T = 4
    def score(arm, b, detail):                          # every SNN arm declares its state and that state's deployment file hash
        r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
    def solve(arm, kind, detail):
        r.free(); b = r.step_calibrate(arm, T) if kind == 'P0' else r.count_calibrate_fit(arm, T)
        r.solver[arm]['target_key'] = r.target_key; r.solver[arm]['state'] = r.variant
        score(arm, b, {**detail, 'solver': kind, 'target_key': r.target_key})
    if 'main' in stages:
        r.score_ann('teacher', r.teacher, T)
        for v, tag in [('plain', 'pl'), ('spatial', 'sp')]:
            r.variant = v; r.score_ann(f'twin-{tag}', r.net(T)['proxy'], T); score(f'S0-{tag}', r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
        r.variant = 'spatial'; r.score_ann('a8-sp', r.grid_proxy(8)['proxy'], T)     # A8(phi) ANN reference (QCFS L=8); T only selects the scoring config
        for d, rows in r.draws.items():
            r.variant = 'plain'
            r.use_rows_target(d, rows, 'teacher'); solve(f'P0-pl-te-{d}', 'P0', {'rows': d})
            r.use_rows_target(d, rows, 'twin'); solve(f'P0-pl-tw-{d}', 'P0', {'rows': d})
            r.variant = 'spatial'
            r.use_rows_target(d, rows, 'teacher'); solve(f'P0-sp-te-{d}', 'P0', {'rows': d})
            r.use_rows_target(d, rows, 'a8'); solve(f'P0-sp-a8-{d}', 'P0', {'rows': d})
            r.use_rows_target(d, rows, 'twin'); solve(f'P0-sp-tw-{d}', 'P0', {'rows': d}); solve(f'M0-sp-tw-{d}', 'M0', {'rows': d})
            r.use_rows_target(f'{d}s', rows[:small], 'twin'); solve(f'P0-sp-tw-{d}s', 'P0', {'rows': f'{d}s'})
            for tag, ann, ann_tag in [('tw', r.net(T)['proxy'], 'twin-spatial'), ('te', r.teacher, 'teacher')]:
                r.free(); b = r.ftbc_batches_ann(f'F0-sp-{tag}-{d}s', T, r.registered_batches(rows[:small]), c['ftbc_alpha'], [int(x['index']) for x in rows[:small]],
                                                 'registered', ann, ann_tag)
                score(f'F0-sp-{tag}-{d}s', b, {'ftbc': r.ftbc[f'F0-sp-{tag}-{d}s']})
    if 'release' in stages:
        r.variant = 'spatial'
        if 'main' not in stages:
            score('S0-sp', r.zero_biases(T), {'schedule': 'IF v0=theta/2', 'role': 'cross-job anchor'})
        versions = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'PIL': PIL.__version__}
        for k, rows in r.release_sets.items():
            seed = c['release_seeds'][k]
            r.use_rows_target(k, rows, 'twin'); solve(f'P0-sp-tw-{k}', 'P0', {'rows': k})
            for tag, ann, ann_tag in [('tw', r.net(T)['proxy'], 'twin-spatial'), ('te', r.teacher, 'teacher')]:
                arm = f'F0-sp-rel-{tag}-{k}'; rk = f'{k}-{tag}'
                try:
                    r.free(); batches, order = r.release_batches(rows, seed)      # same seed: identical augmented inputs for both ANN references
                    b = r.ftbc_batches_ann(arm, T, batches, c['ftbc_alpha'], order, 'release-default', ann, ann_tag, {'seed': seed, 'versions': versions})
                    del batches; r.free(); score(arm, b, {'ftbc': r.ftbc[arm]})
                    r.release_record[rk] = {'status': 'completed', 'seed': seed, 'ann': ann_tag}
                except Exception as exc:                    # any failure is recorded; the remaining realizations continue
                    r.ftbc.pop(arm, None)
                    r.release_record[rk] = {'status': 'out_of_memory' if 'out of memory' in str(exc).lower() else 'failed', 'seed': seed, 'ann': ann_tag,
                                            'exception_type': type(exc).__name__, 'traceback': traceback.format_exc()}
                    del exc
                    r.release_record[rk]['leftover_hooks_dropped'] = r.clear_release_hooks()
                    r.free(); r.progress('release_failed', set=rk, status=r.release_record[rk]['status'])
                r.e0.atomic_json(r.out / 'release.json', r.release_record)
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for key, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    for key, G in r.grid_proxies.items():
        assert r.e0.state_digest(G['proxy']) == G['proxy_digest']
    assert r.clear_release_hooks() == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'stages': stages, 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True,
              'proxy_digests': {f'{k[0]}-T{k[1]}': v['proxy_digest'] for k, v in r.nets.items()}, 'snn_digests': {f'{k[0]}-T{k[1]}': v['immutable_digest'] for k, v in r.nets.items()},
              'grid_proxy_digests': {f'{k[0]}-L{k[1]}': v['proxy_digest'] for k, v in r.grid_proxies.items()}, 'teacher_digest': r.teacher_digest,
              'deployment_sha256': r.deployment_sha,
              'all_strict_reloads_passed': True, 'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'release_sets': {k: [int(x['index']) for x in rows] for k, rows in r.release_sets.items()},
              'release': r.release_record, 'alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-C development (prospective) and exposed dev512 continuity'}
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
