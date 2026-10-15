"""E13 serial deferral and count-calibration screen (EXPERIMENT_PLAN.md). One process, one Slurm-allocated A100.

Reuses frozen E0 run_probe (source/proxy/SNN construction, loader, evaluate, logits, digests) and the E1
AST-verified FTBC export. New code: arm orchestration, serial count-mean solver and deferral schedules
(serial_solver.py), diagnostics.
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import serial_solver as ss
if not __debug__:
    raise SystemExit('E13 acceptance-critical checks use assert; do not run with -O')

DIAG_ARMS = ['S0', 'S1', 'M1', 'L1', 'F0-n3']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def gpu_guard(allowed):
    job = os.environ.get('SLURM_JOB_ID'); cgroup = Path('/proc/self/cgroup').read_text()
    if not job or 'slurm' not in cgroup.lower():
        raise RuntimeError('requires a Slurm cgroup and SLURM_JOB_ID')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('expected exactly one Slurm-visible GPU')
    torch.cuda.set_device(0); token = torch.zeros(1, device='cuda'); torch.cuda.synchronize()
    q = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], text=True)
    own = [l.split(',')[1].strip() for l in q.splitlines() if l.split(',')[0].strip() == str(os.getpid())]
    if len(set(own)) != 1 or own[0] not in allowed:
        raise RuntimeError(f'PID GPU UUID outside allowlist: {own}')
    props = torch.cuda.get_device_properties(0); del token
    return {'slurm_job_id': job, 'pid': os.getpid(), 'cgroup': cgroup, 'gpu_uuid': own[0], 'name': props.name,
            'total_memory': props.total_memory, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version()}


class PrefixDone(Exception):
    pass


class Run:
    def __init__(self, c):
        self.c = c; self.t0 = time.perf_counter(); self.out = Path(c['output'])
        self.out.mkdir(parents=True, exist_ok=True)
        assert not any((self.out / f).exists() for f in ['progress.json', 'result.json', 'failure.json'])
        manifest = json.loads(Path(c['freeze_manifest']).read_text())['files_sha256']
        for p, h in manifest.items():
            assert sha(p) == h, 'frozen file changed: ' + p
        self.e0 = load('e13_e0', c['e0_runner'])
        self.parent = json.loads(Path(c['parent_config']).read_text())
        assert self.parent['family'] == 'serial' and self.parent['timesteps'] == 4
        self.roles = self.e0.validate_config(self.parent)
        self.hardware = gpu_guard(c['allowed_uuids'])
        self.e0.atomic_json(self.out / 'hardware.json', self.hardware); self.e0.atomic_json(self.out / 'config.json', c)
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
        source, self.classes = self.e0.source_model(self.parent)
        self.teacher = source.cuda().eval(); self.teacher_digest = self.e0.state_digest(self.teacher)
        exp = json.loads(Path(c['calibration_export_manifest']).read_text())['families']['serial']
        assert sha(c['calibration_export']) == exp['export_sha256'] and exp['definition_ast_identity'] and not exp['protected_entrypoint_exported']
        self.inherited = load('e13_ftbc_export', c['calibration_export'])
        self.deploy = torch.load(c['plain_deployment'], map_location='cpu', weights_only=False)
        assert all(not torch.count_nonzero(x) for v in self.deploy['runtime_biases'].values() for x in v)
        self.proxy_state = torch.load(c['plain_proxy'], map_location='cpu', weights_only=True)['state_dict']
        self.rows = {'fit': self.roles['fit'][:c['fit_count']], 'development': self.roles['development'][:c['dev_count']],
                     'diagnostic': self.roles['diagnostic'][:c['diag_count']],
                     'ftbc': self.roles['fit'][:c['ftbc_batch'] * max(max(c['ftbc_ns']), c['f1_n'])]}
        self.nets = {}; self.records = {}; self.predictions = {}; self.diag = {}; self.solver = {}; self.ftbc = {}; self.anchor = {}
        self.test_images = next(iter(self.e0.loader(self.rows['fit'][:4], self.parent, 4)))[0].cuda()
        self.progress('initialized')

    def progress(self, stage, **kw):
        rec = {'stage': stage, 'seconds': time.perf_counter() - self.t0, **kw}
        self.e0.atomic_json(self.out / 'progress.json', rec); print(json.dumps(rec), flush=True)

    def parent_T(self, T):
        p = dict(self.parent); p['timesteps'] = T; return p

    def net(self, T):
        if T not in self.nets:
            pT = self.parent_T(T)
            proxy = self.e0.make_proxy(self.teacher, pT, self.classes); proxy.load_state_dict(self.proxy_state, strict=True)
            snn = self.e0.make_snn(proxy, pT, self.classes); snn.load_state_dict(self.deploy['state_dict'], strict=True)
            names = [n for n, m in snn.named_modules() if isinstance(m, self.classes['spike'])]
            assert len(names) == 33
            proxy = proxy.cuda().eval()
            self.nets[T] = {'proxy': proxy, 'snn': snn, 'names': names, 'modules': dict(snn.named_modules()),
                            'immutable_digest': self.e0.state_digest(snn), 'proxy_digest': self.e0.state_digest(proxy), 'parent': pT}
        return self.nets[T]

    def set_biases(self, T, biases):
        N = self.net(T)
        for n in N['names']:
            assert len(biases[n]) == T and all(torch.isfinite(b).all() for b in biases[n])
            N['modules'][n].time_based_bias = [b.detach().to(torch.float32).clone().cuda() for b in biases[n]]

    def schedule_biases(self, T, offsets_theta):
        """Layer-wide per-step biases from cumulative offsets in theta units (same for every channel)."""
        N = self.net(T); b = ss.biases_from_offsets(torch.as_tensor(offsets_theta, dtype=torch.float64), T)
        out = {}
        for n in N['names']:
            m = N['modules'][n]; th = float(m.thresh.data); C = self.layer_channels(T, n)
            out[n] = [(b[t] * th).to(torch.float32).expand(C).clone() for t in range(T)]
        return out

    def layer_channels(self, T, n):
        if not hasattr(self, '_channels'):
            self._channels = {}
            N = self.net(4); sizes = {}
            def hook(name):
                def f(m, args, y): sizes[name] = args[0].shape[1]
                return f
            hs = [N['modules'][k].register_forward_hook(hook(k)) for k in N['names']]
            try:
                self.e0.logits(N['snn'], self.test_images[:1], self.parent, True)
            finally:
                for h in hs: h.remove()
            self._channels = sizes
        return self._channels[n]

    def zero_biases(self, T):
        return {n: [torch.zeros(self.layer_channels(T, n)) for _ in range(T)] for n in self.net(T)['names']}

    @torch.no_grad()
    def score(self, arm, T, biases, detail):
        N = self.net(T); self.set_biases(T, biases)
        p = {role: self.e0.evaluate(N['snn'], self.rows[role], N['parent'], True) for role in ['development']}
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
        self.predictions[arm] = p; self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()},
                                                       'strict_reload': True, 'detail': detail}
        self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
        self.progress('scored', arm=arm, T=T, development=p['development']['metrics'])

    @torch.no_grad()
    def teacher_means(self):
        totals, sizes, hs = {}, {}, []
        N = self.net(4); tm = dict(self.teacher.named_modules())
        def hook(n):
            def f(m, args, y):
                totals[n] = totals.get(n, 0) + y.double().sum((0, 2, 3)); sizes[n] = sizes.get(n, 0) + y.numel() // y.shape[1]
            return f
        for n in N['names']: hs.append(tm[n].register_forward_hook(hook(n)))
        try:
            for x, _, _ in self.fit_batches: self.e0.logits(self.teacher, x.cuda(), self.parent)
        finally:
            for h in hs: h.remove()
        return {n: totals[n] / sizes[n] for n in N['names']}

    @torch.no_grad()
    def capture(self, T, name):
        N = self.net(T); m = N['modules'][name]; cache = None; off = 0
        def hook(_, args, y):
            nonlocal cache, off
            x = args[0].view(T, -1, *args[0].shape[1:])
            if cache is None:
                cache = torch.empty((T, self.c['fit_count'], *x.shape[2:]), device=x.device, dtype=x.dtype)
            cache[:, off:off + x.shape[1]].copy_(x); off += x.shape[1]
            raise PrefixDone()
        h = m.register_forward_hook(hook)
        try:
            for x, _, _ in self.fit_batches:
                try: self.e0.logits(N['snn'], x.cuda(), N['parent'], True)
                except PrefixDone: pass
                else: raise AssertionError('prefix hook did not fire')
        finally:
            h.remove()
        assert off == self.c['fit_count']
        return cache

    @torch.no_grad()
    def count_calibrate(self, arm, T, init_offsets, fit_schedule):
        N = self.net(T); targets = self.targets; biases = self.zero_biases(T); self.set_biases(T, biases); info = {}
        base = ss.biases_from_offsets(torch.as_tensor(init_offsets, dtype=torch.float64), T)
        for i, n in enumerate(N['names']):
            m = N['modules'][n]; thre = m.thresh.data; th = float(thre); t_start = time.perf_counter()
            x = self.capture(T, n); C = x.shape[2]
            if fit_schedule:
                off, sched = ss.fit_schedule(x, thre, T, sweeps=self.c['schedule_sweeps'])
                b0 = (ss.biases_from_offsets(off, T) * th).to(torch.float32).cuda().view(T, 1).expand(T, C).contiguous()
            else:
                sched = None; b0 = (base * th).to(torch.float32).cuda().view(T, 1).expand(T, C).contiguous()
            beta, sinfo = ss.solve_mean(x, b0, thre, targets[n], steps=self.c['bisection_steps'], bound=self.c['bound_ratio'])
            b = b0 - beta.view(1, C)
            final_counts = ss.channel_counts(ss.replay(x, b, thre)).cpu()
            m.time_based_bias = [b[t].clone() for t in range(T)]; biases[n] = [b[t].cpu() for t in range(T)]
            info[n] = {**sinfo, 'final_counts': final_counts, 'schedule': sched, 'seconds': time.perf_counter() - t_start,
                       'cache_bytes': x.numel() * 4}
            del x
            self.e0.atomic_torch(self.out / f'solver-{arm}.pt', info)
            self.progress('calibrated_layer', arm=arm, index=i + 1, name=n, mean_abs_residual=float(sinfo['normalized_residual'].abs().mean()),
                          schedule=None if sched is None else sched['history'][-1]['offsets'])
        # Full-network replay: every layer's channel counts on fit256 must equal the solver records.
        counts, hs, order = {}, [], []
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
        self.solver[arm] = {'full_network_replay_equal': True, 'layers': info}
        return biases

    def ftbc_calibrate(self, arm, T, n_batches, init_biases):
        N = self.net(T); self.set_biases(T, init_biases)
        init_hash = hashlib.sha256(b''.join(init_biases[n][t].to(torch.float32).contiguous().numpy().tobytes() for n in N['names'] for t in range(T))).hexdigest()
        rows = self.rows['ftbc'][:self.c['ftbc_batch'] * n_batches]
        batches = []
        for images, labels, ids in self.e0.loader(rows, self.parent, self.c['ftbc_batch']):
            batches.append((images, torch.full(labels.shape, -1, dtype=labels.dtype)))
        assert len(batches) == n_batches
        count = {'teacher': 0, 'snn': 0, 'updates': {}}
        hs = [self.teacher.register_forward_pre_hook(lambda *a: count.__setitem__('teacher', count['teacher'] + 1)),
              N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
        ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step
        def counted(*args, **kw):
            key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
        self.inherited.bias_corr_step_by_step = counted
        teacher_before = self.e0.state_digest(self.teacher)
        try:
            with torch.no_grad():
                self.inherited.bias_corr_model(self.teacher, N['snn'], T, batches, curr_t_alpha=self.c['ftbc_alpha'], num_cali_sample_batches=n_batches)
        finally:
            self.inherited.bias_corr_step_by_step = original
            for h in hs: h.remove()
        assert count['teacher'] == 33 * n_batches and count['snn'] == 33 * n_batches, count
        assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
        assert self.e0.state_digest(self.teacher) == teacher_before == self.teacher_digest
        biases = {n: [b.detach().cpu().clone() for b in N['modules'][n].time_based_bias] for n in N['names']}
        assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
        self.ftbc[arm] = {'batches': n_batches, 'alpha': self.c['ftbc_alpha'], 'batch_size': self.c['ftbc_batch'], 'calls': count, 'init_bias_sha256': init_hash,
                          'images': [int(r['index']) for r in rows],
                          'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
        return biases

    @torch.no_grad()
    def diagnostics(self, arm, T):
        N = self.net(T); acc = {n: {'abs_local': 0., 'signed_local': 0., 'over': 0, 'under': 0, 'abs_twin': 0., 'signed_twin': 0., 'n': 0,
                                    'spikes_by_step': torch.zeros(T, dtype=torch.float64)} for n in N['names']}
        cap_s, cap_t, hs = {}, {}, []
        def snn_hook(n):
            def f(m, args, y): cap_s[n] = (args[0].view(T, -1, *args[0].shape[1:]), y.view(T, -1, *y.shape[1:]))
            return f
        tw = dict(N['proxy'].named_modules())
        def twin_hook(n):
            def f(m, args, y): cap_t[n] = y
            return f
        for n in N['names']:
            hs.append(N['modules'][n].register_forward_hook(snn_hook(n))); hs.append(tw[n].register_forward_hook(twin_hook(n)))
        try:
            for x, _, _ in self.e0.loader(self.rows['diagnostic'], self.parent, 8):
                cap_s.clear(); cap_t.clear(); x = x.cuda()
                self.e0.logits(N['snn'], x, N['parent'], True); self.e0.logits(N['proxy'], x, N['parent'])
                for n in N['names']:
                    xin, y = cap_s[n]; m = N['modules'][n]; th = float(m.thresh.data)
                    b = torch.stack(m.time_based_bias).to(torch.float64)
                    cnt = (y != 0).sum(0).to(torch.float64)
                    eff = xin.double().sum(0) - b.sum(0).view(1, -1, 1, 1)
                    twin_local = torch.clamp(torch.floor((eff + 0.5 * th) / th), 0, T)
                    twin_net = torch.round(cap_t[n].double() / th * T)
                    e = cnt - twin_local; d = cnt - twin_net; a = acc[n]
                    a['abs_local'] += float(e.abs().sum()); a['signed_local'] += float(e.sum()); a['over'] += int((e > 0).sum()); a['under'] += int((e < 0).sum())
                    a['abs_twin'] += float(d.abs().sum()); a['signed_twin'] += float(d.sum()); a['n'] += e.numel()
                    a['spikes_by_step'] += (y != 0).double().sum((1, 2, 3, 4)).cpu()
        finally:
            for h in hs: h.remove()
        out = {}
        for n, a in acc.items():
            k = a['n']; out[n] = {'mean_abs_local': a['abs_local'] / k, 'mean_signed_local': a['signed_local'] / k, 'over_fraction': a['over'] / k,
                                  'under_fraction': a['under'] / k, 'mean_abs_vs_twin': a['abs_twin'] / k, 'mean_signed_vs_twin': a['signed_twin'] / k,
                                  'spike_share_by_step': (a['spikes_by_step'] / a['spikes_by_step'].sum().clamp(min=1)).tolist(), 'elements': k}
        self.diag[arm] = out; self.diag_ids = [int(r['index']) for r in self.rows['diagnostic']]
        self.e0.atomic_json(self.out / 'diagnostics.json', self.diag)
        self.progress('diagnostics', arm=arm)

    @torch.no_grad()
    def anchors(self):
        old = torch.load(self.c['e0_predictions'], map_location='cpu', weights_only=True)
        rep = {}
        for arm, key in [('S0', 'plain_snn'), ('twin-T4', 'plain_proxy'), ('teacher', 'teacher')]:
            a = self.predictions[arm]['development']; o = old[key]; n = len(a['ids'])
            assert torch.equal(a['ids'], o['ids'][:n]) and torch.equal(a['labels'], o['labels'][:n])
            rep[arm] = {'e0_key': key, 'max_logit_delta': float((a['logits'] - o['logits'][:n]).abs().max()),
                        'prediction_disagreements': int((a['predictions'] != o['predictions'][:n]).sum()),
                        'cross_hardware': self.hardware['gpu_uuid'] != self.parent['allowed_uuid']}
        self.anchor = rep; self.e0.atomic_json(self.out / 'anchors.json', rep)

    @torch.no_grad()
    def score_ann(self, arm, model, T):
        p = {'development': self.e0.evaluate(model, self.rows['development'], self.parent_T(T), False)}
        self.predictions[arm] = p; self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'ann': True}
        self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
        self.progress('scored', arm=arm, T=T, development=p['development']['metrics'])


@torch.no_grad()
def run(c):
    r = Run(c); T = 4
    r.fit_batches = list(r.e0.loader(r.rows['fit'], r.parent, c['solver_batch']))
    assert torch.equal(torch.cat([b[2] for b in r.fit_batches]), torch.tensor([v['index'] for v in r.rows['fit']]))
    r.score_ann('teacher', r.teacher, T); r.score_ann('twin-T4', r.net(T)['proxy'], T)
    zero = r.zero_biases(T)
    r.score('S0', T, zero, {'schedule': 'IF v0=theta/2'}); r.diagnostics('S0', T)
    r.anchors()
    s1 = r.schedule_biases(T, ss.deferral(T, 1.0)); r.score('S1', T, s1, {'schedule': 'exact linear deferral kappa=1'}); r.diagnostics('S1', T)
    r.score('S2', T, r.schedule_biases(T, ss.deferral(T, 0.5)), {'schedule': 'kappa=0.5'})
    r.score('S3', T, r.schedule_biases(T, c['s3_offsets_theta']), {'schedule': 'diagnostic-fitted global', 'offsets_theta': c['s3_offsets_theta']})
    r.targets = r.teacher_means()
    r.e0.atomic_torch(r.out / 'teacher-targets.pt', {'means': {n: v.cpu() for n, v in r.targets.items()}, 'batch': c['solver_batch'],
                                                       'ids': torch.cat([b[2] for b in r.fit_batches])})
    for arm, init, fit in [('M0', torch.zeros(T, dtype=torch.float64), False), ('M1', ss.deferral(T, 1.0), False), ('L1', ss.deferral(T, 1.0), True)]:
        b = r.count_calibrate(arm, T, init, fit)
        r.score(arm, T, b, {'count_mean_solver': True, 'init': init.tolist(), 'layer_schedule_fit': fit})
        if arm in DIAG_ARMS: r.diagnostics(arm, T)
    for n in c['ftbc_ns']:
        arm = f'F0-n{n}'; b = r.ftbc_calibrate(arm, T, n, zero); r.score(arm, T, b, {'ftbc': r.ftbc[arm]})
        if arm in DIAG_ARMS: r.diagnostics(arm, T)
    f1 = f"F1-n{c['f1_n']}"; b = r.ftbc_calibrate(f1, T, c['f1_n'], s1); r.score(f1, T, b, {'ftbc': r.ftbc[f1], 'init': 'S1 deferral'})
    for Tx in c['extra_T']:
        r.score_ann(f'twin-T{Tx}', r.net(Tx)['proxy'], Tx)
        r.score(f'S0-T{Tx}', Tx, r.zero_biases(Tx), {'schedule': 'IF v0=theta/2'})
        r.score(f'S1-T{Tx}', Tx, r.schedule_biases(Tx, ss.deferral(Tx, 1.0)), {'schedule': 'exact linear deferral kappa=1'})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for Tx, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True, 'twin_unchanged': True,
              'diagnostic_ids': r.diag_ids, 'proxy_digests': {str(k): v['proxy_digest'] for k, v in r.nets.items()},
              'all_strict_reloads_passed': True, 'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()},
              'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0, 'max_memory_allocated': torch.cuda.max_memory_allocated(),
              'max_memory_reserved': torch.cuda.max_memory_reserved(), 'anchors': r.anchor, 'data_status': 'exposed development only'}
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
