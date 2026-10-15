"""Locked evaluation v2 acceptance and pre-registered analysis (../innovation/official-evaluation-plan-v2.md section 4) for one job:
python3 l1_accept.py <job tag> {sanity,main}. Job tags: r34-t2, r34-t4, r34-t8, vgg-t2, vgg-t4, vgg-t8 (ImageNet) and c10-vgg16, c10-resnet20,
c100-vgg16, c100-resnet20 (CIFAR). CPU only; never initializes CUDA.

Per draw: the frozen E14 paired exact McNemar and decide rule (candidate better: one-sided p < 0.05 and lower mean cross-entropy; candidate
worse: one-sided p < 0.05 the other way); the paired bootstrap 95% interval of the top-1 difference with 10,000 resamples. The paired difference
of one image is +1, -1 or 0, so resampling images with replacement is exactly a multinomial draw of the three counts; seed 20261011 + 100 * cell
+ 10 * PAIR_INDEX[(candidate, control)] + draw. Over the three draws: the four-way reading of E20-E28 (as the frozen E25 reading_e25).
Primary question (threshold d): Ours-96 against the FTBC arm with the highest total correct count over the three draws (the mean top-1, exactly)
among the FTBC arms that ran: FTBC-96 and FTBC-tuned (FTBC-96 alone where G* is the registered setting; FTBC-def is not run, plan candidate 4),
ties to the earlier arm of that list; passed when replicated (candidate better in all three draws); no minimum margin. Secondary: Ours-96 against each FTBC arm and against Shared; Ours-32
against FTBC-96. The fit records are checked against the configuration (initial review N5)."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
E = Path(__file__).resolve().parents[1]; C1 = E.parent / 'c1-pre'
CAL = ['C1', 'C2', 'C3']
TAGS = {'r34-t2': ('imagenet', 'r34', [2]), 'r34-t4': ('imagenet', 'r34', [4]), 'r34-t8': ('imagenet', 'r34', [8]),
        'vgg-t2': ('imagenet', 'vgg', [2]), 'vgg-t4': ('imagenet', 'vgg', [4]), 'vgg-t8': ('imagenet', 'vgg', [8]),
        'c10-vgg16': ('cifar10', 'vgg16', [2, 4, 8]), 'c10-resnet20': ('cifar10', 'resnet20', [2, 4, 8]),
        'c100-vgg16': ('cifar100', 'vgg16', [2, 4, 8]), 'c100-resnet20': ('cifar100', 'resnet20', [2, 4, 8])}
CELL_INDEX = {f'{t}-T{T}': i for i, (t, T) in enumerate((t, T) for t, (_, _, Ts) in TAGS.items() for T in Ts)}
BOOT, SEED = 10000, 20261011
PAIR_INDEX = {('Ours96', 'FTBC96'): 1, ('Ours96', 'FTBCdef'): 2, ('Ours96', 'FTBCtuned'): 3, ('Ours96', 'Shared'): 7, ('Ours32', 'FTBC96'): 8}
# initial review N4: the bootstrap seed of a comparison is 20261011 + 100 * cell + 10 * PAIR_INDEX[(candidate, control)] + draw, so the
# threshold (d) question and the secondary question on the same pair give the same interval
LIB_OOM = ('unable to find an engine', 'CUDNN_STATUS_ALLOC_FAILED', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_INTERNAL_ERROR')   # = the E23 runner's


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


A14 = _load('l1a_e14_accept', C1 / 'e14/source/e14_accept.py')
paired, decide = A14.paired, A14.decide


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def reading(readings):
    """The four-way reading (E20 reading_e20 = E25 reading_e25, unchanged)."""
    b, w = readings.count('candidate better'), readings.count('candidate worse')
    if b == len(readings): return 'replicated'
    if w: return 'failed (candidate worse in a draw)'
    if b: return 'partially replicated (some draws better, none worse)'
    return 'not replicated (no difference detected in any draw)'


def oom_ok(v):
    """A qualified out-of-memory record (the E23 rule, as E29's acceptance): an OutOfMemoryError or 'out of memory' message, or a library
    signature with allocator evidence; at the update stage, with the export restored."""
    if v['oom_basis'] == 'exception':
        ok = v['exception_type'] == 'OutOfMemoryError' or 'out of memory' in v['exception_message'].lower()
    else:
        ok = v['oom_basis'] == 'library' and any(s.lower() in v['exception_message'].lower() for s in LIB_OOM) and \
            (v['num_ooms_increase'] > 0 or (v['device_free_bytes'] is not None and v['device_free_bytes'] < 2 ** 30))
    return ok and v['stage'] == 'update' and v['export_restored']


def bootstrap(xc, yc, seed):
    """Paired bootstrap 95% interval of mean(xc) - mean(yc) by the exact multinomial form of image resampling."""
    d = xc.astype(np.int64) - yc.astype(np.int64); n = d.size
    p = np.array([(d == 1).sum(), (d == -1).sum(), (d == 0).sum()], dtype=np.float64) / n
    counts = np.random.Generator(np.random.PCG64(seed)).multinomial(n, p, size=BOOT)
    diffs = (counts[:, 0] - counts[:, 1]) / n
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {'difference_pp': 100 * float(d.mean()), 'ci95_pp': [100 * float(lo), 100 * float(hi)], 'resamples': BOOT, 'seed': seed}


def check(tag, phase):
    family, model, Ts = TAGS[tag]
    c = json.loads((E / f'config/{tag}-{phase}.json').read_text())
    frozen = json.loads(Path(c['freeze_manifest']).read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, 'frozen file changed: ' + p
    out = Path(c['output']); res = json.loads((out / 'result.json').read_text()); rec = json.loads((out / 'l1.json').read_text())
    metrics = json.loads((out / 'metrics.json').read_text())
    assert res['status'] == 'completed' and res['all_strict_reloads_passed'] and res['teacher_unchanged'] and res['non_bias_unchanged']
    assert family != 'imagenet' or (res['release_mode'] == c['release_mode'] in ('chunked', 'original', 'skip')), 'release mode (initial review N5)'
    assert c['release_mode'] in ('chunked', 'original', 'skip')
    n = c['evaluation_n']
    if phase == 'main':
        if family == 'imagenet':
            assert c['evaluation_manifest'].endswith('input/imagenet-val.json') and n == 50000 and sha(c['evaluation_manifest']) == c['evaluation_manifest_sha256']
        else:
            assert c['evaluation'] == 'test' and n == 10000
            prep = json.loads((E / 'input/preparation.json').read_text()); assert c['test_file_sha256'] == prep[c['dataset']]['test_file_inventory_sha256']
    cal = json.loads(Path(c['calibration_manifest'] if family == 'imagenet' else c['rows_manifest']).read_text())
    report = {'tag': tag, 'phase': phase, 'evaluation_n': n, 'per_T': {}, 'calibration_test_overlap': rec['evaluation'].get('calibration_test_overlap')}

    def ev(arm):
        e = torch.load(out / f'eval-{arm}.pt', map_location='cpu', weights_only=False)
        assert e['ids'].numel() == n and torch.equal(e['correct'], e['predictions'] == e['labels'])
        return e
    ann = ev('ANN'); labels = ann['labels']; ids = ann['ids']
    report['ANN'] = rec['arms']['ANN']['metrics']
    for T in Ts:
        tuned = c['tuned'] if family == 'imagenet' else c['tuned'][f'T{T}']
        ftbc = ['FTBC96'] + (['FTBCdef'] if c['release_mode'] != 'skip' else []) + (['FTBCtuned'] if tuned else [])
        oom = [d for d in CAL if rec['fits'].get(f'FTBCdef-T{T}-{d}', {}).get('status') == 'out_of_memory']
        if 'FTBCdef' in ftbc and oom:                                      # plan section 3: FTBC-def not run in this cell (memory)
            assert family == 'imagenet' and all(oom_ok(rec['fits'][f'FTBCdef-T{T}-{d}']) for d in oom), oom
            ftbc.remove('FTBCdef'); def_status = {'status': 'out_of_memory (显存不可行，未测)', 'draws_out_of_memory': oom,
                                                  'oom_basis': {d: rec['fits'][f'FTBCdef-T{T}-{d}']['oom_basis'] for d in oom}}
        else:
            def_status = {'status': 'run' if 'FTBCdef' in ftbc else 'skip (not run by the plan: the release default is cited from the paper only)'}
        arms = {a: {d: ev(f'{a}-T{T}-{d}') for d in CAL} for a in ftbc + ['Shared', 'Ours96', 'Ours32']}
        mode = 'original' if family != 'imagenet' else c['release_mode']
        for k, d in enumerate(CAL):                                       # initial review N5
            f = lambda a: rec['fits'][f'{a}-T{T}-{d}']
            assert (f('FTBC96')['images'], f('FTBC96')['alpha'], f('FTBC96')['passes']) == (c['small_n'], c['ftbc_alpha'], 1), (T, d)
            assert f('Shared')['images'] == f('Ours96')['images'] == c['small_n'] and f('Ours32')['images'] == c['tiny_n'], (T, d)
            if tuned:
                assert (f('FTBCtuned')['images'], f('FTBCtuned')['alpha'], f('FTBCtuned')['passes']) == (c['small_n'], tuned['alpha'], tuned['passes']), (T, d)
            if 'FTBCdef' in ftbc:
                v = f('FTBCdef'); pool = c['release_n'] * c['release_batch'] if family == 'imagenet' else c['release_n']
                assert v['images'] == pool and v['seed'] == c['release_seed'] + k and v['mode'] == mode, (T, d, v)
                assert v.get('chunk') == (c['release_chunk'] if mode == 'chunked' else None), (T, d, v)
        s0 = ev(f'S0-T{T}')
        for a, per in arms.items():
            for d, e in per.items():
                assert torch.equal(e['ids'], ids) and torch.equal(e['labels'], labels), (a, d)
                assert f'{a}-T{T}-{d}' in metrics and metrics[f'{a}-T{T}-{d}']['strict_reload'], (a, d)
        acc = lambda e: 100 * float(e['correct'].double().mean())
        means = {a: float(np.mean([acc(per[d]) for d in CAL])) for a, per in arms.items()}
        ranges = {a: [min(acc(per[d]) for d in CAL), max(acc(per[d]) for d in CAL)] for a, per in arms.items()}
        means['FTBCtuned'] = means.get('FTBCtuned', means['FTBC96']); ranges['FTBCtuned'] = ranges.get('FTBCtuned', ranges['FTBC96'])
        total = {a: sum(int(per[d]['correct'].sum()) for d in CAL) for a, per in arms.items()}
        best = max(ftbc, key=lambda a: (total[a], -ftbc.index(a)))          # follow-up 1 R1: total correct over the draws (exact; the n are equal), ties to FTBC-96
        cell = CELL_INDEX[f'{tag}-T{T}']

        def compare(x, y):
            qi = PAIR_INDEX[(x, y)]; per_draw, readings = {}, []
            for k, d in enumerate(CAL):
                ex, ey = arms[x][d], arms[y][d]
                pr = paired(labels, ex['predictions'], ey['predictions'])
                dec = decide(pr, float(ex['cross_entropy'].mean()), float(ey['cross_entropy'].mean()))
                dec['bootstrap'] = bootstrap(ex['correct'].numpy(), ey['correct'].numpy(), SEED + 100 * cell + 10 * qi + k)
                dec['top1_candidate'], dec['top1_control'] = acc(ex), acc(ey)
                per_draw[d] = dec; readings.append(dec['reading'])
            return {'candidate': x, 'control': y, 'per_draw': per_draw, 'reading': reading(readings),
                    'mean_difference_pp': float(np.mean([v['bootstrap']['difference_pp'] for v in per_draw.values()]))}
        questions = {'threshold_d': compare('Ours96', best)}
        for a in ftbc:
            questions[f'Ours96_vs_{a}'] = compare('Ours96', a)
        questions['Ours96_vs_Shared'] = compare('Ours96', 'Shared')
        questions['Ours32_vs_FTBC96'] = compare('Ours32', 'FTBC96')
        report['per_T'][f'T{T}'] = {'best_ftbc': best, 'ftbc_total_correct': {a: total[a] for a in ftbc}, 'ftbc_arms': ftbc, 'FTBCdef': def_status, 'tuned': tuned, 'tuned_is_registered': tuned is None,
                                    'top1_mean': means, 'top1_range': ranges, 'S0_top1': acc(s0), 'questions': questions,
                                    'threshold_d_passed': questions['threshold_d']['reading'] == 'replicated'}
    report.update(passed=True, scope=f'{phase} only', frozen_files=len(frozen), cuda_initialized=torch.cuda.is_initialized(),
                  rule='threshold (d): Ours-96 vs the best FTBC arm on the evaluation set (mean over draws); passed when replicated; no minimum margin')
    (E / f'{tag}-{phase}-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'tag': tag, 'phase': phase, **{T: {'best': v['best_ftbc'], 'threshold_d': v['questions']['threshold_d']['reading']} for T, v in report['per_T'].items()}}))
    assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in TAGS and sys.argv[2] in ('sanity', 'main'), sys.argv
    check(sys.argv[1], sys.argv[2])
