"""E18 second-model screen (EXPERIMENT_PLAN.md): serial VGG-16 Q16, P0 against M0 and registered FTBC at T=4 and T=8.
One process and one Slurm-allocated A100 per job.

Self-contained runner. The E13-E17 runners assume ResNet-34 (33 interfaces) and the E0 deployment files, so this file builds the VGG-16
source, its T-level twins and SNNs directly from the frozen QCFS source snapshot. It reuses unchanged:
  - E0 run_probe helpers: whitelisted image loader (eval transform), evaluate/logits, atomic writes, state_digest;
  - the E13 GPU guard (Slurm cgroup, one visible GPU, PID GPU UUID in the allowlist);
  - the E14 per-step solver (P0, step_solver.py) and the E13 count-mean solver with its IF replay (M0, serial_solver.py);
  - the released FTBC update (E1 calibration export, AST-verified) with the E16 counting wrapper semantics, run with two recipes:
    the registered recipe (alpha 0.4, 3 batches of 32) and the ECCV supplement FTBC+QCFS ImageNet schedule (alpha 0.5, 50 batches of 2).
Method code paths mirror E13/E14/E16 (prefix capture, IF replay of solver totals, full-network replay, strict reload); only the model
construction and the interface reductions (4D convolutional and 2D classifier interfaces) are new.
"""
import argparse, copy, hashlib, importlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E18 acceptance-critical checks use assert; do not run with -O')
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]


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
    return hashlib.sha256(b''.join(biases[n][t].detach().to(torch.float32).cpu().contiguous().numpy().tobytes() for n in names for t in range(T))).hexdigest()


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


def renamed_source_state(path):
    """QCFS checkpoint keys -> source snapshot keys ('.up' thresholds are 'thresh'; same rule as route-c/source/serial_cpu_probe.py)."""
    state = torch.load(path, map_location='cpu', weights_only=True); out = {}
    for k, v in state.items():
        k2 = k.removeprefix('module.'); k2 = k2[:-3] + '.thresh' if k2.endswith('.up') else k2
        assert k2 not in out, 'checkpoint key collision'; out[k2] = v
    return out


def channel_dims(y):
    return (0,) + tuple(range(2, y.dim()))                 # (B,C,H,W) -> (0,2,3); (B,C) -> (0,)


class PrefixDone(Exception):
    pass


class Run:
    def __init__(self, c):
        self.c = c; self.t0 = time.perf_counter(); self.out = Path(c['output'])
        self.out.mkdir(parents=True, exist_ok=True)
        assert not any((self.out / f).exists() for f in ['progress.json', 'result.json', 'failure.json'])
        for p, h in json.loads(Path(c['freeze_manifest']).read_text())['files_sha256'].items():
            assert sha(p) == h, 'frozen file changed: ' + p
        self.e0 = load('e18_e0', c['e0_runner']); E13 = load('e18_e13', c['e13_runner'])
        self.ps = load('e18_ps', c['step_solver']); self.ss = load('e18_ss', c['serial_solver'])
        self.hardware = E13.gpu_guard(c['allowed_uuids'])
        self.e0.atomic_json(self.out / 'hardware.json', self.hardware); self.e0.atomic_json(self.out / 'config.json', c)
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
        # Source model: VGG-16 with QCFS L=16 from the frozen serial source snapshot.
        sys.path.insert(0, c['source_snapshot']); self.layer = importlib.import_module('Models.layer'); VGG = importlib.import_module('Models.VGG')
        assert Path(self.layer.__file__).resolve().is_relative_to(Path(c['source_snapshot']).resolve())
        assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
        renamed = renamed_source_state(c['source_checkpoint'])
        src = VGG.vgg16(1000, 0.0); src.set_L(c['source_level'])
        assert set(src.state_dict()) == set(renamed), 'strict source key mismatch'
        src.load_state_dict(renamed, strict=True); assert all(torch.equal(src.state_dict()[k], v) for k, v in renamed.items())
        self.IF = self.layer.IF
        ifs = [(n, m) for n, m in src.named_modules() if isinstance(m, self.IF)]
        assert len(ifs) == c['interfaces'] and all(m.L == c['source_level'] and m.thresh.numel() == 1 and float(m.thresh) > 0 for _, m in ifs)
        self.names = [n for n, _ in ifs]
        self.source = self.freeze(src)                                   # CPU copy for building twins and SNNs
        self.teacher = self.freeze(copy.deepcopy(src)).cuda(); self.teacher_digest = self.e0.state_digest(self.teacher)
        exp = json.loads(Path(c['calibration_export_manifest']).read_text())['families']['serial']
        assert sha(c['calibration_export']) == exp['export_sha256'] and exp['definition_ast_identity'] and not exp['protected_entrypoint_exported']
        self.inherited = load('e18_ftbc_export', c['calibration_export'])
        assert self.inherited.IF is self.IF, 'the released update must see the same IF class as the model'
        # Data: fresh-E (evaluation), exposed dev512 (continuity), calibration draws C1-C3, all outside every earlier role or split.
        self.parent = {'family': 'serial', 'timesteps': 4, 'seed': c['seed'], 'num_workers': c['num_workers'], 'eval_batch_size': c['eval_batch_size']}
        roles = json.loads(Path(c['dataset_manifest']).read_text())
        assert sha(c['dataset_manifest']) == c['dataset_manifest_sha256'] and sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
        assert sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
        fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']; cal = json.loads(Path(c['calibration_manifest']).read_text())
        assert sorted(cal) == CAL and all(len(cal[k]) >= c['draw_n'] for k in CAL) and len(fresh) >= c['fresh_count']
        old = [r for v in roles.values() for r in v] + [r for v in json.loads(Path(c['e5_roles']).read_text()).values() for r in v]
        old += [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
        old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
        new = fresh + [r for k in CAL for r in cal[k]]
        assert len({int(r['index']) for r in new}) == len(new) == len({r['sha256'] for r in new}), 'duplicate rows'
        assert not {int(r['index']) for r in new} & {int(r['index']) for r in old} and not {r['sha256'] for r in new} & {r['sha256'] for r in old}
        self.rows = {'fresh': fresh[:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
        self.draws = {k: cal[k][:c['draw_n']] for k in CAL}
        assert c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n'] and tuple(c['score_roles']) == ('fresh', 'development')
        assert c['eccv_n'] == c['eccv_batch'] * c['eccv_batches'] < c['draw_n'] and c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2
        self.test_images = next(iter(self.e0.loader(self.draws['C1'][:4], self.parent, 4)))[0].cuda()
        self.nets = {}; self.records = {}; self.predictions = {}; self.solver = {}; self.ftbc = {}; self.bias_hashes = {}; self.targets_by_n = {}
        self.progress('initialized', interfaces=len(self.names))

    @staticmethod
    def freeze(model):
        model.eval(); model.requires_grad_(False); return model

    def progress(self, stage, **kw):
        rec = {'stage': stage, 'seconds': time.perf_counter() - self.t0, **kw}
        self.e0.atomic_json(self.out / 'progress.json', rec); print(json.dumps(rec), flush=True)

    def free(self):
        torch.cuda.empty_cache()

    def build_snn(self, T):
        proxy = copy.deepcopy(self.source); proxy.set_L(T)
        snn = copy.deepcopy(proxy); snn.set_T(T)
        for m in snn.modules():
            if isinstance(m, self.IF) and hasattr(m, 'time_based_bias'):
                del m.time_based_bias
        return self.freeze(proxy), self.freeze(snn)

    def net(self, T):
        """T-level twin (QCFS L=T, ANN mode) and the T-step serial SNN, both from the VGG-16 Q16 source parameters."""
        if T not in self.nets:
            proxy, snn = self.build_snn(T); proxy = proxy.cuda(); snn = snn.cuda()
            names = [n for n, m in snn.named_modules() if isinstance(m, self.IF)]; assert names == self.names
            assert sorted({m.L for m in proxy.modules() if isinstance(m, self.IF)}) == [T] and all(m.T == T for m in snn.modules() if isinstance(m, self.IF))
            self.nets[T] = {'proxy': proxy, 'snn': snn, 'names': names, 'modules': dict(snn.named_modules()), 'parent': dict(self.parent, timesteps=T),
                            'immutable_digest': self.e0.state_digest(snn), 'proxy_digest': self.e0.state_digest(proxy)}
        return self.nets[T]

    def channels(self, n):
        if not hasattr(self, '_channels'):
            sizes = {}; tm = dict(self.teacher.named_modules())
            hs = [tm[k].register_forward_hook(lambda m, a, y, k=k: sizes.__setitem__(k, y.shape[1])) for k in self.names]
            try:
                with torch.no_grad(): self.teacher(self.test_images[:1])
            finally:
                for h in hs: h.remove()
            self._channels = sizes
        return self._channels[n]

    def zero_biases(self, T):
        return {n: [torch.zeros(self.channels(n)) for _ in range(T)] for n in self.names}

    def set_biases(self, T, biases):
        N = self.net(T)
        for n in N['names']:
            assert len(biases[n]) == T and all(torch.isfinite(b).all() for b in biases[n])
            N['modules'][n].time_based_bias = [b.detach().to(torch.float32).clone().cuda() for b in biases[n]]

    def runtime_biases(self, T):
        N = self.net(T); return {n: [b.detach().cpu().clone() for b in N['modules'][n].time_based_bias] for n in N['names']}

    @torch.no_grad()
    def teacher_means(self):
        totals, sizes, hs = {}, {}, []; tm = dict(self.teacher.named_modules())
        def hook(n):
            def f(m, args, y):
                totals[n] = totals.get(n, 0) + y.double().sum(channel_dims(y)); sizes[n] = sizes.get(n, 0) + y.numel() // y.shape[1]
            return f
        for n in self.names: hs.append(tm[n].register_forward_hook(hook(n)))
        try:
            for x, _, _ in self.fit_batches: self.e0.logits(self.teacher, x.cuda(), self.parent)
        finally:
            for h in hs: h.remove()
        return {n: totals[n] / sizes[n] for n in self.names}

    def use_rows(self, key, rows):
        """Active fit rows (batch solver_batch) and teacher channel-mean targets on exactly these rows (saved with the row IDs)."""
        self.fit_n = len(rows)
        self.fit_batches = list(self.e0.loader(rows, self.parent, self.c['solver_batch']))
        assert torch.equal(torch.cat([b[2] for b in self.fit_batches]), torch.tensor([int(v['index']) for v in rows]))
        tkey = f'{key}-teacher'
        if tkey not in self.targets_by_n:
            self.targets_by_n[tkey] = self.teacher_means()
            self.e0.atomic_torch(self.out / f'targets-{tkey}.pt', {'means': {k: v.cpu() for k, v in self.targets_by_n[tkey].items()}, 'target': 'teacher',
                                                                   'batch': self.c['solver_batch'], 'network_digest': self.teacher_digest,
                                                                   'level': sorted({int(m.L) for m in self.teacher.modules() if isinstance(m, self.IF)}),
                                                                   'ids': torch.tensor([int(v['index']) for v in rows])})
        self.targets = self.targets_by_n[tkey]; self.target_key = tkey

    @torch.no_grad()
    def capture(self, T, name):
        """Input currents (T, n_fit, ...) of interface `name` from the deployed SNN with the current upstream corrections (prefix stop)."""
        N = self.net(T); m = N['modules'][name]; cache = None; off = 0
        def hook(_, args, y):
            nonlocal cache, off
            x = args[0].view(T, -1, *args[0].shape[1:])
            if cache is None:
                cache = torch.empty((T, self.fit_n, *x.shape[2:]), device=x.device, dtype=x.dtype)
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
        assert off == self.fit_n
        return cache

    @torch.no_grad()
    def replay_check(self, arm, T, info):
        N = self.net(T); counts, hs, order = {}, [], []
        def hook(n):
            def f(mod, args, y):
                z = y.view(T, -1, *y.shape[1:]); cnt = torch.count_nonzero(z, dim=(0, 1) + tuple(range(3, z.dim()))).cpu()
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
        """P0 (E14 step_solver): per-step sequential exact solve of the FTBC channel-mean objective, interfaces in forward order."""
        N = self.net(T); targets = self.targets; biases = self.zero_biases(T); self.set_biases(T, biases); info = {}; bound = self.c['bound_ratio']
        for i, n in enumerate(N['names']):
            m = N['modules'][n]; thre = m.thresh.data; th = float(thre); t_start = time.perf_counter()
            x = self.capture(T, n); C = x.shape[2]
            b, sinfo, total = self.ps.solve_per_step(x, thre, targets[n], steps=self.c['bisection_steps'], bound=bound)
            assert b.shape == (T, C) and b.dtype == torch.float32 and torch.isfinite(b).all() and bool((b.abs() <= bound * th).all())
            assert torch.equal(self.ss.replay(x, b, thre), total), f'{arm}: per-step solver totals differ from the IF replay at {n}'
            final_counts = self.ss.channel_counts(total).cpu()
            m.time_based_bias = [b[t].clone() for t in range(T)]; biases[n] = [b[t].cpu() for t in range(T)]
            info[n] = {'steps': sinfo['steps'], 'target_mean': sinfo['target_mean'], 'theta': th, 'b': b.cpu(),
                       'observations_per_channel_per_step': sinfo['observations_per_channel_per_step'], 'final_counts': final_counts,
                       'seconds': time.perf_counter() - t_start, 'cache_bytes': x.numel() * 4}
            del x, total
            self.e0.atomic_torch(self.out / f'solver-{arm}.pt', info); self.progress('calibrated_layer', arm=arm, index=i + 1, name=n)
        self.replay_check(arm, T, info)
        self.solver[arm] = {'full_network_replay_equal': True, 'per_step': True, 'objective': 'P0', 'fit_images': self.fit_n, 'T': T,
                            'target_key': self.target_key, 'layers': info}
        return biases

    @torch.no_grad()
    def count_calibrate(self, arm, T):
        """M0 (E13 serial_solver.solve_mean): one time-shared channel bias matching the T-step count mean, interfaces in forward order."""
        N = self.net(T); targets = self.targets; biases = self.zero_biases(T); self.set_biases(T, biases); info = {}
        for i, n in enumerate(N['names']):
            m = N['modules'][n]; thre = m.thresh.data; th = float(thre); t_start = time.perf_counter()
            x = self.capture(T, n); C = x.shape[2]
            b0 = torch.zeros(T, C, dtype=torch.float32, device=x.device)
            beta, sinfo = self.ss.solve_mean(x, b0, thre, targets[n], steps=self.c['bisection_steps'], bound=self.c['bound_ratio'])
            b = b0 - beta.view(1, C)
            final_counts = self.ss.channel_counts(self.ss.replay(x, b, thre)).cpu()
            m.time_based_bias = [b[t].clone() for t in range(T)]; biases[n] = [b[t].cpu() for t in range(T)]
            info[n] = {**sinfo, 'theta': th, 'final_counts': final_counts, 'seconds': time.perf_counter() - t_start, 'cache_bytes': x.numel() * 4}
            del x
            self.e0.atomic_torch(self.out / f'solver-{arm}.pt', info); self.progress('calibrated_layer', arm=arm, index=i + 1, name=n)
        self.replay_check(arm, T, info)
        self.solver[arm] = {'full_network_replay_equal': True, 'per_step': False, 'objective': 'M0', 'fit_images': self.fit_n, 'T': T,
                            'target_key': self.target_key, 'layers': info}
        return biases

    def recipe(self, name):
        """(alpha, batch size, number of batches) of the two FTBC recipes. ECCV schedule: ECCV appendix Table 6, ImageNet VGG-16 Ours(+QCFS),
        50 iterations of 1 batch of 2 with alpha 0.5; the released loader shuffles, so each iteration is a new batch: 50 consecutive batches
        of 2 in one pass."""
        c = self.c
        return {'registered': (c['ftbc_alpha'], c['ftbc_batch'], 3), 'eccv-schedule': (c['eccv_alpha'], c['eccv_batch'], c['eccv_batches'])}[name]

    def ftbc_batches(self, rows, batch_size, n_batches):
        raw = list(self.e0.loader(rows, self.parent, batch_size))
        assert torch.equal(torch.cat([ids for _, _, ids in raw]), torch.tensor([int(r['index']) for r in rows])), 'FTBC batch IDs differ from the manifest rows'
        out = [(images, torch.full(labels.shape, -1, dtype=labels.dtype)) for images, labels, _ in raw]
        assert len(out) == n_batches and all(b[0].shape[0] == batch_size for b in out) and sum(b[0].shape[0] for b in out) == len(rows)
        return out

    def ftbc_run(self, arm, T, rows, recipe):
        """Released FTBC update (E1 export) with one recipe: alpha, n batches of eval-transform images in manifest order, Q16 teacher as ANN,
        zero start; the E16 counting wrapper (ANN and SNN forward calls, per-interface update calls) and unchanged-state checks."""
        alpha, batch_size, n_batches = self.recipe(recipe)
        N = self.net(T); zero = self.zero_biases(T); self.set_biases(T, zero); batches = self.ftbc_batches(rows, batch_size, n_batches)
        init_hash = bias_hash(zero, N['names'], T); ann = self.teacher
        count = {'ann': 0, 'snn': 0, 'updates': {}}
        hs = [ann.register_forward_pre_hook(lambda *a: count.__setitem__('ann', count['ann'] + 1)),
              N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
        ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step
        def counted(*args, **kw):
            key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
        self.inherited.bias_corr_step_by_step = counted
        ann_before = self.e0.state_digest(ann)
        try:
            with torch.no_grad():
                self.inherited.bias_corr_model(ann, N['snn'], T, batches, curr_t_alpha=alpha, num_cali_sample_batches=n_batches)
        finally:
            self.inherited.bias_corr_step_by_step = original
            for h in hs: h.remove()
        L = len(N['names'])
        assert count['ann'] == L * n_batches and count['snn'] == L * n_batches, count
        assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
        assert self.e0.state_digest(ann) == ann_before == self.teacher_digest
        biases = self.runtime_biases(T); assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
        self.ftbc[arm] = {'batches': n_batches, 'alpha': alpha, 'batch_size': int(batches[0][0].shape[0]), 'calls': count, 'init_bias_sha256': init_hash,
                          'images': [int(x['index']) for x in rows], 'recipe': recipe, 'inputs_sha256': tensor_hash([b[0] for b in batches]),
                          'ann': 'teacher', 'ann_digest': ann_before, 'ann_level': sorted({int(m.L) for m in ann.modules() if isinstance(m, self.IF)}), 'T': T,
                          'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
        return biases

    @torch.no_grad()
    def score(self, arm, T, biases, detail):
        """Scores on the fresh and development roles and the E13 strict-reload check on four calibration images. The non-bias SNN state is
        immutable and identical for all arms of one T (VGG-16: 553 MB), so it is saved once per T (state-T{T}.pt); each arm file holds its
        runtime biases and that state file's SHA-256, and the strict reload rebuilds the SNN from both files."""
        N = self.net(T); self.set_biases(T, biases)
        p = {role: self.e0.evaluate(N['snn'], self.rows[role], N['parent'], True) for role in self.c['score_roles']}
        before = self.e0.logits(N['snn'], self.test_images, N['parent'], True).cpu()
        assert self.e0.state_digest(N['snn']) == N['immutable_digest'], 'non-bias SNN state changed'
        spath = self.out / f'state-T{T}.pt'
        if 'state_sha256' not in N:
            self.e0.atomic_torch(spath, {'state_dict': {k: v.detach().cpu() for k, v in N['snn'].state_dict().items()}, 'T': T,
                                         'source_checkpoint_sha256': self.c['source_checkpoint_sha256'], 'state_digest': N['immutable_digest']})
            N['state_sha256'] = sha(spath)
        saved = {'runtime_biases': self.runtime_biases(T), 'state_file': spath.name, 'state_sha256': N['state_sha256'], 'state_digest': N['immutable_digest'],
                 'arm': arm, 'T': T, 'detail': detail, 'source_checkpoint_sha256': self.c['source_checkpoint_sha256']}
        path = self.out / f'bias-{arm}.pt'; self.e0.atomic_torch(path, saved)
        re = torch.load(path, map_location='cpu', weights_only=False); sfile = self.out / re['state_file']
        assert sha(sfile) == re['state_sha256'] == N['state_sha256'], 'saved state file changed: ' + arm
        st = torch.load(sfile, map_location='cpu', weights_only=False)
        _, restored = self.build_snn(T); restored.load_state_dict(st['state_dict'], strict=True); restored = restored.cuda()
        mods = dict(restored.named_modules())
        for n, vals in re['runtime_biases'].items():
            mods[n].time_based_bias = [v.cuda() for v in vals]
        after = self.e0.logits(restored, self.test_images, N['parent'], True).cpu()
        assert torch.equal(before, after), 'strict reload changed logits: ' + arm
        del restored, st
        h = bias_hash(re['runtime_biases'], N['names'], T); assert h == bias_hash(biases, N['names'], T); self.bias_hashes[arm] = h
        self.predictions[arm] = p
        self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'strict_reload': True, 'detail': detail, 'bias_sha256': h}
        self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
        self.progress('scored', arm=arm, T=T, **{role: p[role]['metrics'] for role in p})

    @torch.no_grad()
    def score_ann(self, arm, model, T):
        p = {role: self.e0.evaluate(model, self.rows[role], dict(self.parent, timesteps=T), False) for role in self.c['score_roles']}
        self.predictions[arm] = p; self.records[arm] = {'arm': arm, 'T': T, 'metrics': {k: v['metrics'] for k, v in p.items()}, 'ann': True}
        self.e0.atomic_json(self.out / 'metrics.json', self.records); self.e0.atomic_torch(self.out / 'predictions.pt', self.predictions)
        self.progress('scored', arm=arm, T=T, **{role: p[role]['metrics'] for role in p})


@torch.no_grad()
def run(c):
    r = Run(c); small = c['small_n']
    def solve(arm, T, kind, rows_key):
        r.free(); b = r.step_calibrate(arm, T) if kind == 'P0' else r.count_calibrate(arm, T)
        r.score(arm, T, b, {'solver': kind, 'rows': rows_key, 'target_key': r.target_key})
    r.score_ann('teacher', r.teacher, 4)
    for T in TS:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T); r.score(f'S0-T{T}', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
    for d, rows in r.draws.items():
        r.use_rows(d, rows)
        for T in TS:
            solve(f'M0-T{T}-{d}', T, 'M0', d); solve(f'P0-T{T}-{d}', T, 'P0', d)
        for key, arm, n, recipe in [(f'{d}s', 'F0', small, 'registered'), (f'{d}e', 'FE', c['eccv_n'], 'eccv-schedule')]:
            r.use_rows(key, rows[:n])
            for T in TS:
                solve(f'P0-T{T}-{key}', T, 'P0', key)
                r.free(); b = r.ftbc_run(f'{arm}-T{T}-{key}', T, rows[:n], recipe); r.score(f'{arm}-T{T}-{key}', T, b, {'ftbc': r.ftbc[f'{arm}-T{T}-{key}']})
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
              'draws': {d: [int(x['index']) for x in rows] for d, rows in r.draws.items()}, 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-E development (prospective) and exposed dev512 continuity'}
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
