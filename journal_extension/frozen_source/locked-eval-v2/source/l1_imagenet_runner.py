"""Locked evaluation v2, ImageNet child (../innovation/official-evaluation-plan-v2.md sections 3-4): one model (c['model']: 'r34' or 'vgg')
and one T (c['T']) per run. Arms on each calibration pool Ck (manifest order; the first 96 / 32 images):
  FTBC-96     the released update, alpha 0.4, 3 batches of 32, evaluation transform, manifest order, one pass (registered recipe);
  FTBC-tuned  the same with the E24-selected alpha and one pass (all ImageNet G* are one pass); not run where G* is the registered setting;
  FTBC-def    release default: 3 shuffled batches of 200 augmented images (E15 release_batches), alpha 0.4, through the released loop with
              the chunked step of E29 (release_mode 'chunked', chunk release_chunk) or the released step (release_mode 'original');
              skipped when release_mode is 'skip', which every config sets (plan candidate 4: not run, the paper's values are cited); with
              another mode a draw whose fit raises a qualified out-of-memory error (the E23 rule) is recorded as out_of_memory;
  Shared      M0-96, the time-shared count-mean solve (E13), same deployed trajectory, bound and event solver;
  Ours-96, Ours-32  P0, the time-ordered event fitting (E14 step solver, rho 1/2, endpoint, 32 bisections);
plus No corr. (S0) once and the source ANN once. Every arm is scored once on the evaluation role (main: the 50,000 ImageNet validation
images of input/imagenet-val.json, each image hash checked on load; sanity: a training-image stand-in) and passes the frozen strict-reload
check on the E23 fresh/development roles. The source-level teacher is the ANN of every FTBC arm (same-target adaptation).

R34 reuses the frozen E23 R34 runner's build (E17 -> E13 chain: ftbc_registered, release_batches, registered_batches, ftbc_batches_ann,
use_rows_target, count_calibrate_fit, step_calibrate, score); VGG-16 reuses the frozen E18 Run (ftbc_run, use_rows, count_calibrate,
step_calibrate, score) with one recipe override for the tuned alpha. The chunked step is E29's release_chunked.py."""
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
from torch.utils.data import DataLoader
if not __debug__:
    raise SystemExit('locked-evaluation acceptance-critical checks use assert; do not run with -O')
CAL = ['C1', 'C2', 'C3']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


def build(c):
    """Run instance and the per-model operations: fit_registered, fit_tuned, release_batches, fit_m0, fit_p0, biases, score."""
    E23 = load('l1_e23_r34', c['e23_r34_runner'])
    tuned = c['tuned']                                                     # {'alpha': a, 'passes': 1} or None (G* is the registered setting)
    assert tuned is None or (tuned['passes'] == 1 and tuned['alpha'] != c['ftbc_alpha'])
    if c['model'] == 'r34':
        r = E23.build(c)[-1](c)
        ops = {
            'fit_registered': lambda arm, T, rows: r.ftbc_registered(arm, T, rows, 1),
            'fit_tuned': lambda arm, T, rows: r.ftbc_batches_ann(arm, T, r.registered_batches(rows), tuned['alpha'], [int(x['index']) for x in rows], 'tuned',
                                                                 r.teacher, 'teacher', extra={'variant': 'released', 'passes': 1, 'batches_per_pass': 3}),
            'release_batches': lambda rows, seed: r.release_batches(rows, seed),
            'fit_m0': lambda arm, T, key, rows: (r.use_rows_target(key, rows, 'teacher'), r.count_calibrate_fit(arm, T))[1],
            'fit_p0': lambda arm, T, key, rows: (r.use_rows_target(key, rows, 'teacher'), r.step_calibrate(arm, T))[1],
            'biases': lambda T: {n: [b.detach().cpu().clone() for b in r.net(T)['modules'][n].time_based_bias] for n in r.net(T)['names']},
            'score': lambda arm, T, b, detail: r.score(arm, T, b, {**detail, 'state': r.variant, 'deployment_sha256': r.deployment_sha[r.variant]}),
            'clear_hooks': r.clear_release_hooks}
        return E23, r, ops
    assert c['model'] == 'vgg'
    R18 = load('l1_e18_runner', c['e18_runner']); E15 = load('l1_e15_runner', c['e15_runner'])

    class Run(R18.Run):
        def recipe(self, name):
            if name == 'tuned':
                return tuned['alpha'], self.c['ftbc_batch'], 3
            return super().recipe(name)
    r = Run(c)

    def release_batches(rows, seed):
        """E15 release_batches with the same settings (as E29): the global RNG reseeded before loading, shuffling with its own generator."""
        torch.manual_seed(seed)
        loader = DataLoader(E15.AugmentedRows(rows, r.e0.sha256), batch_size=c['release_batch'], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed), drop_last=False)
        out, order = [], []
        for images, labels, ids in loader:
            out.append((images, torch.full(labels.shape, -1, dtype=labels.dtype))); order += [int(i) for i in ids]
        assert len(out) == c['release_n'] and sorted(order) == sorted(int(x['index']) for x in rows)
        return out, order

    def clear_hooks():
        dropped = 0
        for model in [r.teacher] + [N['snn'] for N in r.nets.values()] + [N['proxy'] for N in r.nets.values()]:
            for m in model.modules():
                for key, hook in list(m._forward_hooks.items()):
                    if isinstance(hook, r.inherited.ActivationSaverHookEachTimeStep):
                        del m._forward_hooks[key]; dropped += 1
        return dropped
    ops = {'fit_registered': lambda arm, T, rows: r.ftbc_run(arm, T, rows, 'registered'),
           'fit_tuned': lambda arm, T, rows: r.ftbc_run(arm, T, rows, 'tuned'),
           'release_batches': release_batches,
           'fit_m0': lambda arm, T, key, rows: (r.use_rows(key, rows), r.count_calibrate(arm, T))[1],
           'fit_p0': lambda arm, T, key, rows: (r.use_rows(key, rows), r.step_calibrate(arm, T))[1],
           'biases': r.runtime_biases, 'score': r.score, 'clear_hooks': clear_hooks}
    return E23, r, ops


@torch.no_grad()
def run(c):
    RC = load('l1_release_chunked', c['release_chunked'])
    E23, r, ops = build(c); T = c['T']; small, tiny, pool_n = c['small_n'], c['tiny_n'], c['release_n'] * c['release_batch']
    assert c['ftbc_alpha'] == 0.4 and small == 3 * c['ftbc_batch'] and tiny < small < pool_n and c['release_mode'] in ('chunked', 'original', 'skip')
    cal = json.loads(Path(c['calibration_manifest']).read_text()); assert sorted(cal) == CAL and all(len(cal[d]) >= pool_n for d in CAL)
    assert all(cal[d][:c['draw_n']] == r.draws[d] for d in CAL), 'the runner draws are not the manifest prefixes'
    ev = json.loads(Path(c['evaluation_manifest']).read_text())[c['evaluation_key']]
    assert sha(c['evaluation_manifest']) == c['evaluation_manifest_sha256'] and len(ev) == c['evaluation_n']
    missing = [x['index'] for x in ev if not Path(x['path']).is_file()]          # initial review N9: metadata only, before any fit
    assert not missing, f'{len(missing)} evaluation images are missing, e.g. {missing[:3]}'
    rec = {'model': c['model'], 'T': T, 'evaluation': {'manifest': c['evaluation_manifest'], 'key': c['evaluation_key'], 'n': len(ev),
                                                       'batch_size': c['evaluation_batch_size'], 'workers': c['evaluation_workers']},
           'arms': {}, 'fits': {}}
    save = lambda: r.e0.atomic_json(r.out / 'l1.json', rec)

    def evaluate(arm, model, spiking):
        """Score `model` once on the evaluation role. Logits are reduced to per-image cross-entropy before saving."""
        # the evaluation role is scored with its own batch size and loader workers (plan section 4; the frozen parent keeps 4 and 4 for the
        # strict-reload roles): floating point only, the same for every arm
        parent = dict(r.net(T)['parent'], eval_batch_size=c['evaluation_batch_size'], num_workers=c['evaluation_workers'])
        t0 = time.perf_counter(); P = r.e0.evaluate(model, ev, parent, spiking); seconds = time.perf_counter() - t0
        assert P['ids'].tolist() == [int(x['index']) for x in ev] and torch.equal(P['labels'], torch.tensor([int(x['label']) for x in ev]))
        ce = torch.nn.functional.cross_entropy(P['logits'].double(), P['labels'], reduction='none')
        out = {'ids': P['ids'], 'labels': P['labels'], 'predictions': P['predictions'], 'cross_entropy': ce, 'correct': P['predictions'] == P['labels']}
        r.e0.atomic_torch(r.out / f'eval-{arm}.pt', out)
        rec['arms'][arm] = {'metrics': P['metrics'], 'seconds': seconds, 'n': len(ev), 'predictions_sha256': tensor_hash([P['predictions']])}
        save(); r.progress('evaluated', arm=arm, metrics=P['metrics'], seconds=seconds)
        del P; gc.collect()

    def begin():
        r.free(); gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); return time.perf_counter()

    def finish(arm, b, detail, t0):
        torch.cuda.synchronize()
        rec['fits'][arm] = {'seconds': time.perf_counter() - t0, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **detail}
        ops['score'](arm, T, b, detail)                                    # frozen strict reload on the E23 fresh/development roles
        r.set_biases(T, b); evaluate(arm, r.net(T)['snn'], True)
        return b

    def fitted(arm, fn, detail):
        t0 = begin(); return finish(arm, fn(), detail, t0)

    N = r.net(T); released = r.inherited.bias_corr_step_by_step
    evaluate('ANN', r.teacher, False)                                      # the source ANN, once per run (T only sets the parent config)
    zero = r.zero_biases(T); ops['score'](f'S0-T{T}', T, zero, {'schedule': 'IF v0=theta/2'}); r.set_biases(T, zero); evaluate(f'S0-T{T}', N['snn'], True)
    for k, d in enumerate(CAL):
        pool = cal[d][:pool_n]; rows, rows32 = pool[:small], pool[:tiny]
        fitted(f'FTBC96-T{T}-{d}', lambda: ops['fit_registered'](f'FTBC96-T{T}-{d}', T, rows), {'arm': 'FTBC-96', 'alpha': c['ftbc_alpha'], 'passes': 1, 'images': small})
        if c['tuned'] is not None:
            fitted(f'FTBCtuned-T{T}-{d}', lambda: ops['fit_tuned'](f'FTBCtuned-T{T}-{d}', T, rows), {'arm': 'FTBC-tuned', **c['tuned'], 'images': small})
        if c['release_mode'] != 'skip':
            # plan section 3 (initial review B2): only the data step and the released update may end in a qualified out-of-memory record that
            # leaves this draw not run; the strict-reload score and the evaluation-set score of the arm stay outside, so their errors propagate
            arm, seed = f'FTBCdef-T{T}-{d}', c['release_seed'] + k
            detail = {'arm': 'FTBC-def', 'mode': c['release_mode'], 'chunk': c['release_chunk'] if c['release_mode'] == 'chunked' else None, 'images': pool_n, 'seed': seed}
            step = released if c['release_mode'] == 'original' else RC.chunked_step(r.inherited, c['release_chunk'])
            t0 = begin(); ooms0 = torch.cuda.memory_stats().get('num_ooms', 0); stage = 'data'
            try:
                batches, order = ops['release_batches'](pool, seed)
                stage = 'update'; Nd = r.net(T); r.set_biases(T, r.zero_biases(T)); teacher0 = r.e0.state_digest(r.teacher)
                calls = RC.release_fit(r.inherited, r.teacher, Nd['snn'], T, batches, c['ftbc_alpha'], step, Nd['names'])
                torch.cuda.synchronize()
            except Exception as exc:
                basis, grew, free = E23.oom_basis(exc, ooms0)
                if basis is None:
                    raise
                rec['fits'][arm] = {'status': 'out_of_memory', 'stage': stage, 'oom_basis': basis, 'exception_type': type(exc).__name__,
                                    'exception_message': str(exc)[:500], 'num_ooms_increase': torch.cuda.memory_stats().get('num_ooms', 0) - ooms0,
                                    'device_free_bytes': free, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **detail}
                del exc
                rec['fits'][arm]['leftover_hooks_dropped'] = ops['clear_hooks'](); gc.collect(); torch.cuda.empty_cache()
                rec['fits'][arm]['export_restored'] = r.inherited.bias_corr_step_by_step is released
                assert rec['fits'][arm]['export_restored'] and E23.cuda_alive(), f'{arm}: unusable state after the out-of-memory error'
                r.set_biases(T, r.zero_biases(T)); save(); r.progress('arm', arm=arm, status='out_of_memory', basis=basis, stage=stage)
            else:
                per_call = RC.forwards_per_call('REL' if c['release_mode'] == 'original' else 'CHS', c['release_batch'], c['release_chunk'])
                assert calls['ann_forwards'] == calls['snn_forwards'] == len(Nd['names']) * len(batches) * per_call, calls   # E29 B3: the forward count
                assert r.e0.state_digest(r.teacher) == teacher0 == r.teacher_digest and ops['clear_hooks']() == 0
                rec['fits'][f'{arm}-inputs'] = {'inputs_sha256': tensor_hash([b[0] for b in batches]), 'order': order, 'seed': seed, 'calls': calls}
                b = ops['biases'](T); assert all(torch.isfinite(x).all() for v in b.values() for x in v)
                del batches; finish(arm, b, detail, t0)
        fitted(f'Shared-T{T}-{d}', lambda: ops['fit_m0'](f'Shared-T{T}-{d}', T, f'{d}-{small}', rows), {'arm': 'Shared', 'images': small})
        fitted(f'Ours96-T{T}-{d}', lambda: ops['fit_p0'](f'Ours96-T{T}-{d}', T, f'{d}-{small}', rows), {'arm': 'Ours-96', 'images': small})
        fitted(f'Ours32-T{T}-{d}', lambda: ops['fit_p0'](f'Ours32-T{T}-{d}', T, f'{d}-{tiny}', rows32), {'arm': 'Ours-32', 'images': tiny})
        r.progress('draw_done', draw=d)
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for N in r.nets.values():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    assert ops['clear_hooks']() == 0, 'released saver hooks left at the end of the run'
    torch.cuda.synchronize()
    result = {'status': 'completed', 'model': c['model'], 'T': T, 'arms': sorted(rec['arms']), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'all_strict_reloads_passed': True, 'bias_sha256': r.bias_hashes, 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'allocator_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'release_mode': c['release_mode'],
              'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()},
              'evaluation': rec['evaluation']}
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
