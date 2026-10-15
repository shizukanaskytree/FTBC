"""Locked evaluation v2, CIFAR child (../innovation/official-evaluation-plan-v2.md sections 3-4): one source model per run (CIFAR-10 Q4 or
CIFAR-100 Q8, VGG-16 or ResNet-20), T = 2, 4, 8. Arms on each calibration pool Ck (manifest order; the first 96 / 32 images):
  FTBC-96     the released update, alpha 0.4, one pass of 3 batches of 32 (E24/E25 ftbc_from from zero; equal to the registered recipe);
  FTBC-tuned  the E25-selected alpha and passes (3 passes continue from the 1-pass result, as in E25); not run where G* is the registered setting;
  FTBC-def    release default with the released step (E25 release_from: 600 images, released training augmentation, shuffled, seeded,
              3 batches of 200, alpha 0.4); run only with release_mode 'original': the plan (candidate 4) sets 'skip', so it is not run;
  Shared      M0-96 (E13 count-mean solve);  Ours-96, Ours-32  P0 (E14 step solver);
plus No corr. (S0) once per T and the source ANN once. Every arm is scored once on the evaluation role (main: the official test split,
cifar_test.py; sanity: the E25 dev-L stand-in) and passes the frozen strict-reload check on the E25 'fresh' role.

Reuses the frozen E25 runner's build (the CIFAR construction on the frozen E18 Run: step_calibrate, count_calibrate, use_rows, score,
ftbc_from, release_from) unchanged."""
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
if not __debug__:
    raise SystemExit('locked-evaluation acceptance-critical checks use assert; do not run with -O')
CAL = ['C1', 'C2', 'C3']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


@torch.no_grad()
def run(c):
    E25 = load('l1_e25_runner', c['e25_runner']); R18, CE, Run = E25.build(c); r = Run(c)
    CT = load('l1_cifar_test', c['cifar_test'])
    small, tiny, pool_n = c['small_n'], c['tiny_n'], c['release_n']
    assert c['ftbc_alpha'] == 0.4 and small == 3 * c['ftbc_batch'] and tiny < small < pool_n == 3 * c['release_batch'] and c['release_mode'] in ('original', 'skip')
    cal = json.loads(Path(c['rows_manifest']).read_text()); assert all(len(cal[d]) >= pool_n for d in CAL)
    if c['evaluation'] == 'test':
        store = CT.init_test(CE, c['dataset'], c['test_file'], c['test_file_sha256'])
        overlap = CT.calibration_overlap(store, {d: cal[d][:small] for d in CAL})   # follow-up 1: the rows the run calibrates with (the first 96)
        evaluate_rows = lambda model, parent, spiking: CT.evaluate(CE, model, store, parent, spiking)
        ev_ids, ev_labels, n_eval = [x['index'] for x in store['rows']], [x['label'] for x in store['rows']], len(store['rows'])
    else:                                                                  # sanity stand-in: the E25 dev-L rows (training images)
        ev = json.loads(Path(c['standin_manifest']).read_text())['development'][:c['standin_n']]; overlap = None
        evaluate_rows = lambda model, parent, spiking: CE.evaluate(model, ev, parent, spiking)
        ev_ids, ev_labels, n_eval = [int(x['index']) for x in ev], [int(x['label']) for x in ev], len(ev)
    assert n_eval == c['evaluation_n']
    rec = {'model': c['tag'], 'Ts': c['Ts'], 'evaluation': {'kind': c['evaluation'], 'n': n_eval, 'calibration_test_overlap': overlap}, 'arms': {}, 'fits': {}}
    save = lambda: r.e0.atomic_json(r.out / 'l1.json', rec); save()

    def evaluate(arm, model, T, spiking):
        parent = r.net(T)['parent']
        t0 = time.perf_counter(); P = evaluate_rows(model, parent, spiking); seconds = time.perf_counter() - t0
        assert P['ids'].tolist() == ev_ids and P['labels'].tolist() == ev_labels
        ce = torch.nn.functional.cross_entropy(P['logits'].double(), P['labels'], reduction='none')
        r.e0.atomic_torch(r.out / f'eval-{arm}.pt', {'ids': P['ids'], 'labels': P['labels'], 'predictions': P['predictions'], 'cross_entropy': ce,
                                                     'correct': P['predictions'] == P['labels']})
        rec['arms'][arm] = {'metrics': P['metrics'], 'seconds': seconds, 'n': n_eval, 'T': T, 'predictions_sha256': tensor_hash([P['predictions']])}
        save(); r.progress('evaluated', arm=arm, metrics=P['metrics'], seconds=seconds); del P; gc.collect()

    def fitted(arm, T, fn, detail):
        r.free(); gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); b = fn(); torch.cuda.synchronize()
        rec['fits'][arm] = {'seconds': time.perf_counter() - t0, 'T': T, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),   # initial review N6
                            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **detail}
        r.score(arm, T, b, detail)                                         # frozen strict reload on the E25 fresh role
        r.set_biases(T, b); evaluate(arm, r.net(T)['snn'], T, True)
        return b

    evaluate('ANN', r.teacher, max(c['Ts']), False)
    for T in c['Ts']:
        tuned = c['tuned'][f'T{T}']                                        # {'alpha', 'passes'} or None (G* is the registered setting)
        assert tuned is None or (tuned['passes'] in (1, 3) and (tuned['alpha'], tuned['passes']) != (c['ftbc_alpha'], 1))
        zero = r.zero_biases(T); r.score(f'S0-T{T}', T, zero, {'schedule': 'IF v0=theta/2'}); r.set_biases(T, zero); evaluate(f'S0-T{T}', r.net(T)['snn'], T, True)
        for k, d in enumerate(CAL):
            pool = cal[d][:pool_n]; rows, rows32 = pool[:small], pool[:tiny]
            fitted(f'FTBC96-T{T}-{d}', T, lambda: r.ftbc_from(f'FTBC96-T{T}-{d}', T, rows, c['ftbc_alpha'], zero, 1, {'passes_total': 1, 'start': 'zero'}),
                   {'arm': 'FTBC-96', 'alpha': c['ftbc_alpha'], 'passes': 1, 'images': small})
            if tuned is not None:
                def tune():
                    b1 = r.ftbc_from(f'FTBCtuned1-T{T}-{d}', T, rows, tuned['alpha'], zero, 1, {'passes_total': 1, 'start': 'zero'})
                    if tuned['passes'] == 1:
                        return b1
                    return r.ftbc_from(f'FTBCtuned-T{T}-{d}', T, rows, tuned['alpha'], b1, 2, {'passes_total': 3, 'start': f'FTBCtuned1-T{T}-{d}'})
                fitted(f'FTBCtuned-T{T}-{d}', T, tune, {'arm': 'FTBC-tuned', **tuned, 'images': small})
            if c['release_mode'] == 'original':                             # plan candidate 4: 'skip', the release-default arm is not run
                fitted(f'FTBCdef-T{T}-{d}', T, lambda: r.release_from(f'FTBCdef-T{T}-{d}', T, pool, c['release_seed'] + k),
                       {'arm': 'FTBC-def', 'mode': 'original', 'images': pool_n, 'seed': c['release_seed'] + k})
            r.use_rows(f'{d}-{small}', rows)
            fitted(f'Shared-T{T}-{d}', T, lambda: r.count_calibrate(f'Shared-T{T}-{d}', T), {'arm': 'Shared', 'images': small})
            fitted(f'Ours96-T{T}-{d}', T, lambda: r.step_calibrate(f'Ours96-T{T}-{d}', T), {'arm': 'Ours-96', 'images': small})
            r.use_rows(f'{d}-{tiny}', rows32)
            fitted(f'Ours32-T{T}-{d}', T, lambda: r.step_calibrate(f'Ours32-T{T}-{d}', T), {'arm': 'Ours-32', 'images': tiny})
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
    result = {'status': 'completed', 'model': c['tag'], 'Ts': c['Ts'], 'arms': sorted(rec['arms']), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'all_strict_reloads_passed': True, 'release_update_restored': True, 'bias_sha256': r.bias_hashes, 'hardware': r.hardware,
              'seconds_total': time.perf_counter() - r.t0, 'evaluation': rec['evaluation'],
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved()}
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
