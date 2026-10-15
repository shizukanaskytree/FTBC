"""E29 runner (EXPERIMENT_PLAN.md sections 2-4): the release-default FTBC recipe (the E15 release batches of the E23 pool C1: 600 augmented
images, seeded shuffle, 3 batches of release_batch; alpha 0.4; the source-level teacher as ANN; zero start) fitted three ways for one model
(c['model']: 'r34' or 'vgg') at T = 2, 4, 8:
  REL   the released bias_corr_step_by_step of the E1 export, one batch at a time (attempted under the E23 out-of-memory rule);
  PAIR  the per-call pairing (release_chunked.paired_step; initial review B1), run only where REL completed at that T: every update call runs
        the released step, the chunked step with one chunk per batch (CHB) and the chunked step with chunks of c['chunk_small'] images (CHS)
        from the same saved state, fingerprints the released hook's captures of REL and CHB, records |b - b_REL| / theta per call and keeps
        the released result, so the arm follows the released trajectory;
  CHS   the chunked step with chunks of c['chunk_small'] images (50 in main), unpaired: the implementation the locked evaluation would use.
Every arm goes through the released bias_corr_model (batch -> layer -> time) with the counting wrapper of release_chunked.release_fit (update
calls per interface, whole-network ANN and SNN forward passes; B3); then its corrections, its predictions on the same pool under the
evaluation transform (agreement only; no accuracy is reported or stored), the batches' hash after the arm (B3) and the frozen runner's
strict-reload score are recorded.

R34 reuses the frozen E23 R34 runner's build (E17 -> E16 -> E15 -> E14 -> E13: plain state, draws, the E15 release_batches, the strict-reload
score) and its oom_basis and cuda_alive; VGG-16 reuses the frozen E18 Run (Q16 teacher, the released update through the E1 export with the
same IF class, strict-reload score) and builds the release batches with replica_batches (the E15 release_batches settings on the E15
AugmentedRows)."""
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
from torch.utils.data import DataLoader
if not __debug__:
    raise SystemExit('E29 acceptance-critical checks use assert; do not run with -O')
TS = [2, 4, 8]
MODES = ['REL', 'PAIR', 'CHS']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


def replica_batches(E15, rows, sha256, release_batch, release_n, seed):
    """The E15 release_batches with the same settings (e15_runner.py:141-151): the global RNG (augmentation) reseeded right before loading,
    shuffling with its own generator, num_workers=0, batch release_batch, all rows."""
    torch.manual_seed(seed)
    loader = DataLoader(E15.AugmentedRows(rows, sha256), batch_size=release_batch, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    out, order = [], []
    for images, labels, ids in loader:
        out.append((images, torch.full(labels.shape, -1, dtype=labels.dtype))); order += [int(i) for i in ids]
    assert len(out) == release_n and sorted(order) == sorted(int(x['index']) for x in rows)
    return out, order


def build(c):
    """(E23 R34 runner module, Run instance, release rows, batches(seed) -> (batches, order), biases(T) -> CPU corrections, score(arm, T, b, d))."""
    E23 = load('e29_e23_r34', c['e23_r34_runner'])
    if c['model'] == 'r34':
        out = E23.build(c); r = out[-1](c); rows = r.release_rows
        assert len(rows) == c['release_n'] * c['release_batch']
        def biases(T):
            N = r.net(T); return {n: [b.detach().cpu().clone() for b in N['modules'][n].time_based_bias] for n in N['names']}
        def score(arm, T, b, detail):
            r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]})
        return E23, r, rows, lambda seed: r.release_batches(rows, seed), biases, score
    assert c['model'] == 'vgg'
    R18 = load('e29_e18_runner', c['e18_runner']); r = R18.Run(c)
    E15 = load('e29_e15_runner', c['e15_runner'])                       # AugmentedRows only: the release training transform on whitelisted rows
    cal = json.loads(Path(c['calibration_manifest']).read_text()); rows = cal['C1'][:c['release_n'] * c['release_batch']]
    assert len(rows) == c['release_n'] * c['release_batch'] and rows[:c['draw_n']] == r.draws['C1'], 'release rows are not the pool C1 prefix'
    return E23, r, rows, lambda seed: replica_batches(E15, rows, r.e0.sha256, c['release_batch'], c['release_n'], seed), r.runtime_biases, r.score


def clear_hooks(r):
    """Drop leftover released saver hooks (and the clones they hold) from the teacher, every SNN and every twin (as E15/E16 clear_release_hooks)."""
    dropped = 0
    for model in [r.teacher] + [N['snn'] for N in r.nets.values()] + [N['proxy'] for N in r.nets.values()]:
        for m in model.modules():
            for key, hook in list(m._forward_hooks.items()):
                if isinstance(hook, r.inherited.ActivationSaverHookEachTimeStep):
                    del m._forward_hooks[key]; dropped += 1
    return dropped


def interfaces(r, N, image):
    """Per IF module of the SNN, in named_modules order (the order of the released bias_corr_model): position, rank of its input (4: conv,
    2: vector) and spatial size, from one forward pass on one image (descriptive; the flip scale of section 5)."""
    shapes, hooks = {}, []
    def record(n):
        def hook(module, inputs):                                       # returns None: the input is left unchanged
            shapes.setdefault(n, list(inputs[0].shape))
        return hook
    for n, m in N['snn'].named_modules():
        if isinstance(m, r.inherited.IF):
            hooks.append(m.register_forward_pre_hook(record(n)))
    try:
        N['snn'].eval(); N['snn'](image)                                 # eval mode, as the released GetLayerInputOutput sets it
    finally:
        for h in hooks:
            h.remove()
    order = [n for n, m in N['snn'].named_modules() if isinstance(m, r.inherited.IF)]
    assert sorted(order) == sorted(N['names']) == sorted(shapes)
    return {n: {'position': i, 'rank': len(shapes[n]), 'spatial': (shapes[n][2] * shapes[n][3]) if len(shapes[n]) == 4 else 1} for i, n in enumerate(order)}


@torch.no_grad()
def run(c):
    RC = load('e29_release_chunked', str(Path(__file__).resolve().parent / 'release_chunked.py'))
    E23, r, rows, make_batches, get_biases, score = build(c)
    alpha, seed, B, small = c['ftbc_alpha'], c['release_seed'], c['release_batch'], c['chunk_small']; assert alpha == 0.4
    assert 0 < small < B and B % small == 0
    released, get_original = r.inherited.bias_corr_step_by_step, r.inherited.GetLayerInputOutput
    batches, order = make_batches(seed); in_hash = tensor_hash([b[0] for b in batches])
    assert all(b[0].shape[0] == B for b in batches)
    rec = {'model': c['model'], 'inputs_sha256': in_hash, 'order': order, 'release_rows': [int(x['index']) for x in rows],
           'chunks': {'REL': None, 'PAIR': {'CHB': B, 'CHS': small}, 'CHS': small}, 'thresholds': {}, 'interfaces': {}, 'arms': {}}
    save = lambda: r.e0.atomic_json(r.out / 'e29.json', rec)
    for T in TS:
        N = r.net(T)                                                    # built outside every attempt
        rec['thresholds'][f'T{T}'] = {n: float(N['modules'][n].thresh.data) for n in N['names']}
        r.set_biases(T, r.zero_biases(T)); rec['interfaces'][f'T{T}'] = interfaces(r, N, batches[0][0][:1].cuda())
        L, nb = len(N['names']), len(batches)
        for mode in MODES:
            arm = f'{mode}-T{T}'
            if mode == 'PAIR' and rec['arms'][f'REL-T{T}']['status'] != 'completed':
                rec['arms'][arm] = {'status': 'not_run', 'reason': 'REL did not complete at this T, so the paired call cannot run either'}; save(); continue
            zero = r.zero_biases(T); r.set_biases(T, zero)
            zero_hash = tensor_hash([b for n in N['names'] for b in zero[n]])
            log = []
            if mode == 'REL':
                step = released
            elif mode == 'CHS':
                step = RC.chunked_step(r.inherited, small)
            else:
                ids = {id(N['modules'][n]): n for n in N['names']}
                step = RC.paired_step(r.inherited, [('REL', released, True), ('CHB', RC.chunked_step(r.inherited, B), True),
                                                    ('CHS', RC.chunked_step(r.inherited, small), False)], log, ids)
            per_call = RC.forwards_per_call(mode, B, small)
            gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            st0 = torch.cuda.memory_stats(); ooms0, retries0 = st0.get('num_ooms', 0), st0.get('num_alloc_retries', 0)
            teacher0 = r.e0.state_digest(r.teacher); t0, u0 = time.perf_counter(), time.time()
            try:
                count = RC.release_fit(r.inherited, r.teacher, N['snn'], T, batches, alpha, step, N['names'])
                torch.cuda.synchronize()
            except Exception as exc:
                basis, grew, free = E23.oom_basis(exc, ooms0)
                if basis is None:
                    raise
                t1, u1 = time.perf_counter(), time.time()
                arec = {'status': 'out_of_memory', 'mode': mode, 'T': T, 'stage': 'update', 'oom_basis': basis,
                        'exception_type': type(exc).__name__, 'exception_message': str(exc)[:500],
                        'num_ooms_increase': torch.cuda.memory_stats().get('num_ooms', 0) - ooms0, 'device_free_bytes': free,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'baseline_allocated_bytes': base_a, 'baseline_reserved_bytes': base_r, 'card_total_bytes': r.hardware['total_memory'],
                        'attempt_perf_start': t0, 'attempt_perf_end': t1, 'attempt_unix_start': u0, 'attempt_unix_end': u1,
                        'paired_calls_completed': len(log), 'traceback': traceback.format_exc()}
                del exc
                arec['leftover_hooks_dropped'] = clear_hooks(r); gc.collect(); torch.cuda.empty_cache()
                arec['export_restored'] = r.inherited.bias_corr_step_by_step is released and r.inherited.GetLayerInputOutput is get_original
                assert arec['export_restored'], f'{arm}: the export was not restored after the error'
                assert E23.cuda_alive(), f'{arm}: CUDA context unusable after the out-of-memory error'
                r.set_biases(T, r.zero_biases(T)); rec['arms'][arm] = arec; save()
                r.progress('arm', arm=arm, status='out_of_memory', basis=basis); continue
            t1, u1 = time.perf_counter(), time.time()
            peak_a, peak_r = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(); st1 = torch.cuda.memory_stats()
            assert r.inherited.bias_corr_step_by_step is released and r.inherited.GetLayerInputOutput is get_original
            assert count['ann_forwards'] == count['snn_forwards'] == L * nb * per_call, (arm, count, L, nb, per_call)
            assert mode != 'PAIR' or len(log) == L * nb, (arm, len(log))
            after_hash = tensor_hash([b[0] for b in batches])
            assert r.e0.state_digest(r.teacher) == teacher0 == r.teacher_digest, 'the teacher changed'
            b = get_biases(T); assert sorted(b) == sorted(N['names']) and all(len(v) == T and all(torch.isfinite(x).all() for x in v) for v in b.values())
            assert clear_hooks(r) == 0, f'{arm}: released saver hooks left after a completed update'
            P = r.e0.evaluate(N['snn'], rows, N['parent'], True)      # evaluation transform; agreement between the implementations only
            assert P['ids'].tolist() == [int(x['index']) for x in rows]
            r.e0.atomic_torch(r.out / f'agree-{arm}.pt', {'ids': P['ids'], 'labels': P['labels'], 'predictions': P['predictions']})
            r.e0.atomic_torch(r.out / f'corrections-{arm}.pt', {'arm': arm, 'T': T, 'biases': b})
            if mode == 'PAIR':
                r.e0.atomic_json(r.out / f'pair-T{T}.json', log)
            score(arm, T, b, {'recipe': 'release-default', 'mode': mode, 'chunks': rec['chunks'][mode], 'seed': seed, 'inputs_sha256': after_hash})
            rec['arms'][arm] = {'status': 'completed', 'mode': mode, 'T': T, 'updates': count['updates'],
                                'forwards': {'ann': count['ann_forwards'], 'snn': count['snn_forwards'], 'per_call': per_call, 'interfaces': L, 'batches': nb},
                                'batches': nb, 'batch_size': B, 'inputs_sha256_after': after_hash, 'zero_sha256': zero_hash, 'seconds': t1 - t0,
                                'attempt_unix_start': u0, 'attempt_unix_end': u1, 'peak_allocated_bytes': peak_a, 'peak_reserved_bytes': peak_r,
                                'baseline_allocated_bytes': base_a, 'baseline_reserved_bytes': base_r,
                                'num_ooms_increase': st1.get('num_ooms', 0) - ooms0, 'num_alloc_retries_increase': st1.get('num_alloc_retries', 0) - retries0,
                                'corrections_sha256': tensor_hash([x for n in N['names'] for x in b[n]]), 'paired_calls': len(log),
                                'agree_metrics': {'n': P['metrics']['n'], 'correct': P['metrics']['correct']},
                                'max_abs_bias_over_theta': max(float(x.abs().max()) / rec['thresholds'][f'T{T}'][n] for n, v in b.items() for x in v)}
            save(); r.progress('arm', arm=arm, status='completed', seconds=t1 - t0); del P, log; gc.collect(); torch.cuda.empty_cache()
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for N in r.nets.values():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    assert clear_hooks(r) == 0, 'released saver hooks left at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'model': c['model'], 'arms': {a: v['status'] for a, v in rec['arms'].items()}, 'teacher_unchanged': True,
              'non_bias_unchanged': True, 'all_strict_reloads_passed': True, 'bias_sha256': r.bias_hashes, 'hardware': r.hardware,
              'seconds_total': time.perf_counter() - r.t0, 'allocator_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
              'data_status': 'the E23 pool C1 (evaluation transform) for agreement and the E23 fresh-K/dev roles for strict reload; no accuracy reported'}
    r.e0.atomic_json(r.out / 'result.json', result); r.progress('completed')
    return result


def main(c):
    """Run one child; any error that is not a qualified out-of-memory record leaves failure.json (type, message, traceback) and is re-raised."""
    try:
        return run(c)
    except Exception as exc:
        Path(c['output']).mkdir(parents=True, exist_ok=True)
        (Path(c['output']) / 'failure.json').write_text(json.dumps({'type': type(exc).__name__, 'message': str(exc)[:2000], 'traceback': traceback.format_exc()}))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args()
    print(json.dumps(main(json.loads(Path(a.config).read_text()))))
