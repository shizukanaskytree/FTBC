"""E25 CIFAR development screen, one source model per run (EXPERIMENT_PLAN.md): serial VGG-16 or ResNet-20 (CIFAR-10 Q4, CIFAR-100 Q8),
subtraction-reset IF with v0 = theta/2, T = 2, 4 and 8. Arms per T and calibration pool Ck (first rows of the 600-image pool, manifest order):
S0 (once per T), M0-96, P0-32, P0-96, P0-256, the released FTBC update on the first 96 images with the E24 grid (alpha in section 3, one pass
of 3 updates or three passes of 9; passes 2-3 continue from pass 1) and the release-default FTBC (R0: all 600 images through the released
training augmentation, shuffled with a pre-set seed, 3 batches of 200, alpha 0.4). Every arm is scored on dev-L (10000: selection half,
test half); the source ANN (teacher) and the T-level twins once.

Uses the frozen E18 runner's Run methods unchanged: T-level twins and SNNs, prefix capture, P0 = step_calibrate (E14 step solver), M0 =
count_calibrate (E13 count-mean solver), teacher targets on exactly the fit rows, strict-reload scoring with one saved state per T, and the
released update (E1 calibration export) with the E16 counting wrapper (ftbc_run, used for the sanity equivalence). New here: the CIFAR
construction (this __init__ replaces E18's VGG-16 ImageNet one), the CIFAR data helpers (cifar_e0.py) as `self.e0`, `ftbc_from` (the E24
VGG-16 runner's, unchanged), `release_from`, and the arm loop."""
import argparse, copy, hashlib, importlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('E25 acceptance-critical checks use assert; do not run with -O')
CAL = ['C1', 'C2', 'C3']
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]                       # section 3 (the E24 grid)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def renamed_cifar_state(path, key):
    """Checkpoint keys -> serial source-snapshot keys (the E0 serial rule, as the E26 CPU check): drop 'module.', '.relu.' -> '.act.',
    a trailing '.up' -> '.thresh'. `key`: the entry holding the state dict (CIFAR-100 ResNet-20: 'model'), or None."""
    raw = torch.load(path, map_location='cpu', weights_only=True); state = raw[key] if key else raw; out = {}
    for k, v in state.items():
        k2 = k.removeprefix('module.').replace('.relu.', '.act.')
        if k2.endswith('.up'): k2 = k2[:-len('.up')] + '.thresh'
        assert k2 not in out, 'checkpoint key collision'; out[k2] = v
    return out


def build(c):
    R18 = load('e25_e18_runner', c['e18_runner']); CE = load('e25_cifar_e0', c['cifar_e0'])

    class Run(R18.Run):
        def __init__(self, c):
            """E18 Run.__init__ with the CIFAR model and data in place of VGG-16 ImageNet (same checks, same determinism settings)."""
            self.c = c; self.t0 = time.perf_counter(); self.out = Path(c['output'])
            self.out.mkdir(parents=True, exist_ok=True)
            assert not any((self.out / f).exists() for f in ['progress.json', 'result.json', 'failure.json'])
            for p, h in json.loads(Path(c['freeze_manifest']).read_text())['files_sha256'].items():
                assert R18.sha(p) == h, 'frozen file changed: ' + p
            self.e0 = CE; E13 = load('e25_e13', c['e13_runner'])
            self.ps = load('e25_ps', c['step_solver']); self.ss = load('e25_ss', c['serial_solver'])
            self.hardware = E13.gpu_guard(c['allowed_uuids'])
            # Initial review N-e: the shard is a memory budget, not isolation; cap this process's caching allocator below the request so an
            # overrun fails this job and not the jobs sharing the card. N-j: library versions the release-default augmentation depends on.
            cap = c['gpu_mem_cap_gib'] * 2 ** 30; frac = cap / self.hardware['total_memory']; assert 0 < frac < 1, frac
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            import numpy, PIL, torchvision
            self.hardware.update(memory_cap_bytes=cap, memory_fraction=frac, versions={'numpy': numpy.__version__, 'PIL': PIL.__version__, 'torchvision': torchvision.__version__})
            self.e0.atomic_json(self.out / 'hardware.json', self.hardware); self.e0.atomic_json(self.out / 'config.json', c)
            torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); torch.set_num_threads(8)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
            # Source model from the frozen serial source snapshot.
            sys.path.insert(0, c['source_snapshot']); self.layer = importlib.import_module('Models.layer')
            arch = importlib.import_module('Models.VGG' if c['arch'] == 'vgg16' else 'Models.ResNet')
            assert Path(self.layer.__file__).resolve().is_relative_to(Path(c['source_snapshot']).resolve())
            assert R18.sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
            renamed = renamed_cifar_state(c['source_checkpoint'], c['checkpoint_key'])
            classes = CE.CLASSES[c['dataset']]; assert classes == c['num_classes']
            src = arch.vgg16(classes, 0.0) if c['arch'] == 'vgg16' else arch.resnet20(classes); src.set_L(c['source_level'])
            assert set(src.state_dict()) == set(renamed), 'strict source key mismatch'
            src.load_state_dict(renamed, strict=True); assert all(torch.equal(src.state_dict()[k], v) for k, v in renamed.items())
            self.IF = self.layer.IF
            ifs = [(n, m) for n, m in src.named_modules() if isinstance(m, self.IF)]
            assert len(ifs) == c['interfaces'] and all(m.L == c['source_level'] and m.thresh.numel() == 1 and float(m.thresh) > 0 for _, m in ifs)
            self.names = [n for n, _ in ifs]
            self.source = self.freeze(src)
            self.teacher = self.freeze(copy.deepcopy(src)).cuda(); self.teacher_digest = self.e0.state_digest(self.teacher)
            exp = json.loads(Path(c['calibration_export_manifest']).read_text())['families']['serial']
            assert R18.sha(c['calibration_export']) == exp['export_sha256'] and exp['definition_ast_identity'] and not exp['protected_entrypoint_exported']
            self.inherited = load('e25_ftbc_export', c['calibration_export'])
            assert self.inherited.IF is self.IF, 'the released update must see the same IF class as the model'
            # Data: dev-L (evaluation, two halves) and the calibration pools C1-C3, from the E25 manifest (exclusions in preparation.json).
            assert R18.sha(c['preparation']) == c['preparation_sha256'] and R18.sha(c['rows_manifest']) == c['rows_manifest_sha256']
            CE.init(c['dataset'], c['preparation'])
            self.parent = {'family': 'serial', 'timesteps': 4, 'seed': c['seed'], 'num_workers': 0, 'eval_batch_size': c['eval_batch_size'], 'dataset': c['dataset']}
            rows = json.loads(Path(c['rows_manifest']).read_text()); assert sorted(rows) == sorted(CAL + ['development'])
            new = rows['development'] + [r for k in CAL for r in rows[k]]
            assert len({int(r['index']) for r in new}) == len(new) == len({r['sha256'] for r in new}), 'duplicate rows'
            assert len(rows['development']) >= c['dev_count'] and all(len(rows[k]) >= c['draw_n'] for k in CAL)
            self.rows = {'fresh': rows['development'][:c['dev_count']]}; self.draws = {k: rows[k][:c['draw_n']] for k in CAL}
            assert tuple(c['score_roles']) == ('fresh',) and c['dev_count'] % 2 == 0
            assert c['small_n'] == 3 * c['ftbc_batch'] and (c['sanity'] or c['small_n'] == 96)
            assert c['p0_sizes'][1] == c['small_n'] and max(c['p0_sizes']) <= c['draw_n'] and c['m0_size'] == c['small_n']
            assert c['release_n'] == 3 * c['release_batch'] <= c['draw_n'] and c['release_alpha'] == c['ftbc_alpha'] == 0.4
            self.test_images = next(iter(self.e0.loader(self.draws['C1'][:4], self.parent, 4)))[0].cuda()
            self.release_step = self.inherited.bias_corr_step_by_step
            self.nets = {}; self.records = {}; self.predictions = {}; self.solver = {}; self.ftbc = {}; self.bias_hashes = {}; self.targets_by_n = {}
            self.progress('initialized', interfaces=len(self.names), dataset=c['dataset'], arch=c['arch'])

        def ftbc_from(self, arm, T, rows, alpha, init, passes, extra):
            """E24 VGG-16 runner's ftbc_from, unchanged: E18 ftbc_run with an explicit alpha, the start `init` and `passes` repetitions of
            the registered batches of `rows` (manifest order in every pass) in one released loop; the same counting wrapper and checks."""
            _, batch_size, per_pass = self.recipe('registered')
            N = self.net(T); self.set_biases(T, init); batches = self.ftbc_batches(rows, batch_size, per_pass) * passes; n_batches = len(batches)
            init_hash = R18.bias_hash(init, N['names'], T); ann = self.teacher
            count = {'ann': 0, 'snn': 0, 'updates': {}}
            hs = [ann.register_forward_pre_hook(lambda *a: count.__setitem__('ann', count['ann'] + 1)),
                  N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
            ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step; assert original is self.release_step
            def counted(*args, **kw):
                key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
            self.inherited.bias_corr_step_by_step = counted
            ann_before = self.e0.state_digest(ann)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    self.inherited.bias_corr_model(ann, N['snn'], T, batches, curr_t_alpha=alpha, num_cali_sample_batches=n_batches)
                torch.cuda.synchronize(); seconds = time.perf_counter() - t0
            finally:
                self.inherited.bias_corr_step_by_step = original
                for h in hs: h.remove()
            L = len(N['names'])
            assert count['ann'] == L * n_batches and count['snn'] == L * n_batches, count
            assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
            assert self.e0.state_digest(ann) == ann_before == self.teacher_digest
            biases = self.runtime_biases(T); assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
            self.ftbc[arm] = {'variant': 'released', 'recipe': 'registered-grid', 'batches': n_batches, 'batches_per_pass': per_pass, 'passes_this_call': passes,
                              'alpha': alpha, 'batch_size': int(batches[0][0].shape[0]), 'calls': count, 'init_bias_sha256': init_hash,
                              'bias_sha256': R18.bias_hash(biases, N['names'], T), 'images': [int(x['index']) for x in rows],
                              'inputs_sha256': R18.tensor_hash([b[0] for b in batches[:per_pass]]), 'ann': 'teacher', 'ann_digest': ann_before,
                              'ann_level': sorted({int(m.L) for m in ann.modules() if isinstance(m, self.IF)}), 'T': T, 'seconds': seconds, **extra,
                              'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
            return biases

        def release_from(self, arm, T, rows, seed):
            """Release-default FTBC (section 3, R0): the released loop from zero on the release-default batches of `rows` (cifar_e0
            release_batches: released training augmentation, shuffled, seeded), alpha 0.4, the source-level teacher as ANN (the declared
            same-target adaptation of E18); the same counting wrapper and unchanged-state checks as ftbc_from."""
            c = self.c; N = self.net(T); zero = self.zero_biases(T); self.set_biases(T, zero)
            batches, batch_ids = self.e0.release_batches(rows, c['dataset'], seed, c['release_batch'], 3); n_batches = len(batches)
            assert sorted(i for b in batch_ids for i in b) == sorted(int(x['index']) for x in rows), 'release batches must use every pool image once'
            init_hash = R18.bias_hash(zero, N['names'], T); ann = self.teacher
            count = {'ann': 0, 'snn': 0, 'updates': {}}
            hs = [ann.register_forward_pre_hook(lambda *a: count.__setitem__('ann', count['ann'] + 1)),
                  N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
            ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step; assert original is self.release_step
            def counted(*args, **kw):
                key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
            self.inherited.bias_corr_step_by_step = counted
            ann_before = self.e0.state_digest(ann)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    self.inherited.bias_corr_model(ann, N['snn'], T, batches, curr_t_alpha=c['release_alpha'], num_cali_sample_batches=n_batches)
                torch.cuda.synchronize(); seconds = time.perf_counter() - t0
            finally:
                self.inherited.bias_corr_step_by_step = original
                for h in hs: h.remove()
            L = len(N['names'])
            assert count['ann'] == L * n_batches and count['snn'] == L * n_batches, count
            assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
            assert self.e0.state_digest(ann) == ann_before == self.teacher_digest
            biases = self.runtime_biases(T); assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
            self.ftbc[arm] = {'variant': 'released', 'recipe': 'release-default', 'batches': n_batches, 'alpha': c['release_alpha'],
                              'batch_size': int(batches[0][0].shape[0]), 'calls': count, 'init_bias_sha256': init_hash, 'seed': seed,
                              'bias_sha256': R18.bias_hash(biases, N['names'], T), 'images': [int(x['index']) for x in rows], 'batch_ids': batch_ids,
                              'inputs_sha256': R18.tensor_hash([b[0] for b in batches]), 'ann': 'teacher', 'ann_digest': ann_before,
                              'ann_level': sorted({int(m.L) for m in ann.modules() if isinstance(m, self.IF)}), 'T': T, 'seconds': seconds,
                              'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
            return biases

    return R18, CE, Run


@torch.no_grad()
def run(c):
    R18, CE, Run = build(c); r = Run(c); small = c['small_n']; equivalence = {}; timing = {}
    nominal = lambda alpha, updates: round(alpha * updates, 10)
    def solve(arm, T, kind, rows_key):
        r.free(); torch.cuda.synchronize(); t0 = time.perf_counter()
        b = r.step_calibrate(arm, T) if kind == 'P0' else r.count_calibrate(arm, T); torch.cuda.synchronize(); timing[arm] = time.perf_counter() - t0
        r.score(arm, T, b, {'solver': kind, 'rows': rows_key, 'target_key': r.target_key})
    r.score_ann('teacher', r.teacher, max(c['Ts']))
    for T in c['Ts']:
        r.score_ann(f'twin-T{T}', r.net(T)['proxy'], T)
        zero = r.zero_biases(T); r.score(f'S0-T{T}', T, zero, {'schedule': 'IF v0=theta/2'})
        for k, d in enumerate(CAL, 1):
            pool = r.draws[d]
            r.use_rows(f'{d}-{small}', pool[:small]); solve(f'M0-{small}-T{T}-{d}', T, 'M0', f'{d}-{small}'); solve(f'P0-{small}-T{T}-{d}', T, 'P0', f'{d}-{small}')
            for n in [m for m in c['p0_sizes'] if m != small]:
                r.use_rows(f'{d}-{n}', pool[:n]); solve(f'P0-{n}-T{T}-{d}', T, 'P0', f'{d}-{n}')
            rows = pool[:small]
            for alpha in c['alphas']:
                a1 = f'G-{alpha}-1-T{T}-{d}'; r.free()
                b1 = r.ftbc_from(a1, T, rows, alpha, zero, 1, {'passes_total': 1, 'updates_total': 3, 'nominal_gain': nominal(alpha, 3), 'start': 'zero'})
                r.score(a1, T, b1, {'ftbc': {k2: v for k2, v in r.ftbc[a1].items() if k2 != 'calls'}})
                a3 = f'G-{alpha}-3-T{T}-{d}'; r.free()
                b3 = r.ftbc_from(a3, T, rows, alpha, b1, 2, {'passes_total': 3, 'updates_total': 9, 'nominal_gain': nominal(alpha, 9), 'start': a1})
                r.score(a3, T, b3, {'ftbc': {k2: v for k2, v in r.ftbc[a3].items() if k2 != 'calls'}})
                if alpha == 0.4 and d == 'C1':      # sanity and main: G-0.4-1 equals the frozen E18 registered FTBC; continuing equals one call
                    frozen1 = r.ftbc_run(f'G1frozen-T{T}-{d}', T, rows, 'registered')
                    single = r.ftbc_from(f'G3single-T{T}-{d}', T, rows, alpha, zero, 3, {'passes_total': 3, 'updates_total': 9, 'nominal_gain': nominal(alpha, 9), 'start': 'zero'})
                    names = r.net(T)['names']
                    equivalence[f'T{T}-{d}'] = {'grid_one_pass_sha256': r.ftbc[a1]['bias_sha256'], 'frozen_registered_sha256': R18.bias_hash(frozen1, names, T),
                                                'continued_sha256': r.ftbc[a3]['bias_sha256'], 'single_call_sha256': r.ftbc[f'G3single-T{T}-{d}']['bias_sha256']}
                    e = equivalence[f'T{T}-{d}']
                    assert e['grid_one_pass_sha256'] == e['frozen_registered_sha256'], 'G-0.4-1 differs from the frozen E18 registered FTBC'
                    assert e['continued_sha256'] == e['single_call_sha256'], 'continued passes differ from one call'
            arm = f'R0-T{T}-{d}'; r.free()
            b = r.release_from(arm, T, pool[:c['release_n']], c['release_seed'] + k)
            r.score(arm, T, b, {'ftbc': {k2: v for k2, v in r.ftbc[arm].items() if k2 not in ('calls', 'batch_ids')}})
            r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc); r.progress('draw_done', T=T, draw=d)
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.inherited.bias_corr_step_by_step is r.release_step
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for T, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    leftover = sum(isinstance(h, r.inherited.ActivationSaverHookEachTimeStep) for mdl in [r.teacher] + [N['snn'] for N in r.nets.values()]
                   for m in mdl.modules() for h in m._forward_hooks.values())
    assert leftover == 0, 'released-helper hooks left on a model at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'teacher_digest': r.teacher_digest, 'proxy_digests': {f'T{T}': v['proxy_digest'] for T, v in r.nets.items()},
              'snn_digests': {f'T{T}': v['immutable_digest'] for T, v in r.nets.items()}, 'interfaces': r.names,
              'channels': {n: r.channels(n) for n in r.names}, 'all_strict_reloads_passed': True, 'release_update_restored': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in v] for d, v in r.draws.items()}, 'Ts': c['Ts'], 'alphas': c['alphas'], 'timing_seconds': timing,
              'equivalence': equivalence or None, 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'train-derived dev-L (prospective; first half selection, second half test); the source ANN was trained on the training split'}
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
