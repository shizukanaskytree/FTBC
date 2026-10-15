"""E25 CPU acceptance for one CIFAR source model, the pre-registered selection and analysis (EXPERIMENT_PLAN.md sections 4-7). No CUDA.
Usage: e25_accept.py <job tag> {sanity,main}, job tag in c10-vgg16, c10-resnet20, c100-vgg16, c100-resnet20.
Checks adapted from the E24 VGG-16 acceptance (itself from the frozen E20/E22 acceptances) for the CIFAR construction, M0, P0-256 and the
release-default FTBC; statistics (exact one-sided McNemar, decide rule) unchanged from the frozen E14 acceptance, status and
state_dict_digest from E16, the four-way reading from E20. The selection uses the selection half only; the paired questions and the
descriptive grid range use the test half only."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E25 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
A14 = _load('e25a_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e25a_e16_accept', E.parent / 'e16/source/e16_accept.py')
RUN = _load('e25a_runner_helpers', E / 'source/e25_runner.py')                  # renamed_cifar_state only (no CUDA at import)
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL = ['C1', 'C2', 'C3']; ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]; PASSES = [1, 3]
READINGS = ['candidate better', 'no difference detected', 'candidate worse']
TAGS = {'c10-vgg16': ('cifar10', 'vgg16'), 'c10-resnet20': ('cifar10', 'resnet20'), 'c100-vgg16': ('cifar100', 'vgg16'), 'c100-resnet20': ('cifar100', 'resnet20')}
QNAME = {'QA': 'selected setting vs registered setting', 'QB': 'P0-96 vs the selected FTBC', 'QC': 'P0-96 vs registered FTBC',
         'QD': 'P0-96 vs M0-96 (per-step vs time-shared)', 'QE': 'P0-96 vs release-default FTBC'}
avg = lambda vals: sum(vals) / len(vals)


def git_out(M, *a):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)    # never refreshes or rewrites the index


def manuscript_check(M, before, hard=True):
    """Section 7 item 8: compares HEAD, the index entries and the staged changes with the snapshot and records the status lines and the
    index file. The sanity and main acceptances pass hard=False and only record (the E24 follow-up 1 R1); verify_preservation.py asserts."""
    cur = {'HEAD': git_out(M, 'rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git_out(M, 'ls-files', '--stage').encode()).hexdigest(),
           'staged': git_out(M, 'diff', '--cached', '--name-status'), 'status': git_out(M, 'status', '--porcelain=v1', '--untracked-files=all'),
           'index_sha256': sha(M / '.git/index')}
    same = all(cur[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged'])
    if hard:
        assert same, 'manuscript HEAD, index entries or staged changes differ'
    a, b = before['status'].splitlines(), cur['status'].splitlines()
    return {'head_index_entries_staged_unchanged': same, 'decided_here': hard, 'status_lines_added': sorted(set(b) - set(a)),
            'status_lines_removed': sorted(set(a) - set(b)), 'index_file_unchanged': cur['index_sha256'] == before['index_sha256']}


def reading_e25(readings):
    """Section 5 four-way reading (E20's reading_e20, unchanged)."""
    b, w = readings.count('candidate better'), readings.count('candidate worse')
    if b == len(readings): return 'replicated'
    if w: return 'failed (candidate worse in a draw)'
    if b: return 'partially replicated (some draws better, none worse)'
    return 'not replicated (no difference detected in any draw)'


def grid_arm(alpha, passes, T, d):
    return f'G-{alpha}-{passes}-T{T}-{d}'


def expected_arms(Ts, sizes, small):
    arms = ['teacher']
    for T in Ts:
        arms += [f'twin-T{T}', f'S0-T{T}']
        for d in CAL:
            arms += [f'M0-{small}-T{T}-{d}'] + [f'P0-{n}-T{T}-{d}' for n in sizes] + [grid_arm(a, p, T, d) for a in ALPHAS for p in PASSES] + [f'R0-T{T}-{d}']
    return arms


def halves(n):
    h = n // 2
    return {'sel': slice(0, h), 'test': slice(h, n)}


def correct_on(pred, lab, sl):
    return int((pred['predictions'][sl] == lab[sl]).sum())


def ce_on(pred, lab, sl):
    return float(torch.nn.functional.cross_entropy(pred['logits'][sl].double(), lab[sl]))


def select(pred_fresh, lab, n, T):
    """Section 4 (the E24 rule): the grid setting with the largest sum over C1-C3 of selection-half correct counts; ties to the smaller
    nominal gain (alpha x updates), then to fewer passes."""
    sl = halves(n)['sel']; rows = []
    for a in ALPHAS:
        for p in PASSES:
            tot = sum(correct_on(pred_fresh[grid_arm(a, p, T, d)], lab, sl) for d in CAL)
            rows.append({'alpha': a, 'passes': p, 'updates': 3 * p, 'nominal_gain': round(a * 3 * p, 10), 'selection_correct_sum': tot})
    rows.sort(key=lambda r: (-r['selection_correct_sum'], r['nominal_gain'], r['passes']))
    return rows[0], rows


def sensitivity(pred_fresh, lab, n, T):
    """Descriptive only (section 5, as E24): per pool, the test-half range over the 16 settings and every setting's net against G-0.4-1."""
    sl = halves(n)['test']; out = {}
    for d in CAL:
        cor = {(a, p): correct_on(pred_fresh[grid_arm(a, p, T, d)], lab, sl) for a in ALPHAS for p in PASSES}
        lo, hi = min(cor.values()), max(cor.values()); reg = pred_fresh[grid_arm(0.4, 1, T, d)]['predictions'][sl]
        out[d] = {'test_correct_min': lo, 'test_correct_max': hi, 'range': hi - lo,
                  'settings_at_min': [{'alpha': a, 'passes': p} for (a, p), v in cor.items() if v == lo],
                  'settings_at_max': [{'alpha': a, 'passes': p} for (a, p), v in cor.items() if v == hi],
                  'net_vs_registered': [{'alpha': a, 'passes': p, **{k: v for k, v in paired(lab[sl], pred_fresh[grid_arm(a, p, T, d)]['predictions'][sl], reg).items()
                                                                     if k in ('x_only_correct', 'y_only_correct', 'net')}} for a in ALPHAS for p in PASSES]}
    return out


def analysis(Ts, sizes, small, pred, lab, n, msplit):
    """Selection, the pre-registered questions QA-QE on the test half, the descriptive grid range, the grid table and descriptive pairs."""
    sl = halves(n)['test']; out = {'selection': {}, 'decisions': {}, 'grid': {}, 'sensitivity': {}, 'descriptive_pairs': {}}
    fresh = {a: pred[a]['fresh'] for a in pred}
    for T in Ts:
        star, ranking = select(fresh, lab, n, T); reg = star['alpha'] == 0.4 and star['passes'] == 1
        out['selection'][f'T{T}'] = {'selected': star, 'ranking': ranking, 'selected_is_registered': reg}
        sa = lambda d, T=T, s=star: grid_arm(s['alpha'], s['passes'], T, d); rg = lambda d, T=T: grid_arm(0.4, 1, T, d)
        p96 = lambda d, T=T: f'P0-{small}-T{T}-{d}'
        qs = {} if reg else {'QA': (sa, rg)}
        qs.update({'QB': (p96, sa), 'QC': (p96, rg), 'QD': (p96, lambda d, T=T: f'M0-{small}-T{T}-{d}'), 'QE': (p96, lambda d, T=T: f'R0-T{T}-{d}')})
        for k, (x, y) in qs.items():
            per = {}
            for d in CAL:
                xa, ya = x(d), y(d)
                pr = paired(lab[sl], fresh[xa]['predictions'][sl], fresh[ya]['predictions'][sl])
                per[d] = decide(pr, ce_on(fresh[xa], lab, sl), ce_on(fresh[ya], lab, sl)) | {'candidate': xa, 'control': ya}
            rd = [v['reading'] for v in per.values()]
            out['decisions'][f'{k}-T{T}'] = {'question': QNAME[k], 'per_draw': per, 'status': status(rd), 'counts': {r: rd.count(r) for r in READINGS}, 'reading_e25': reading_e25(rd)}
        out['grid'][f'T{T}'] = [{'alpha': a, 'passes': p, 'selection_correct': [msplit[grid_arm(a, p, T, d)]['sel']['correct'] for d in CAL],
                                 'test_correct': [msplit[grid_arm(a, p, T, d)]['test']['correct'] for d in CAL],
                                 'test_ce': [msplit[grid_arm(a, p, T, d)]['test']['ce'] for d in CAL]} for a in ALPHAS for p in PASSES]
        out['sensitivity'][f'T{T}'] = sensitivity(fresh, lab, n, T)
        others = [m for m in sizes if m != small]
        pairs = {f'P0-{m}-vs-P0-{small}': (lambda d, m=m, T=T: f'P0-{m}-T{T}-{d}', p96) for m in others}
        pairs.update({'R0-vs-registered': (lambda d, T=T: f'R0-T{T}-{d}', rg), 'selected-vs-S0': (sa, lambda d, T=T: f'S0-T{T}'),
                      f'P0-{small}-vs-S0': (p96, lambda d, T=T: f'S0-T{T}'), 'registered-vs-S0': (rg, lambda d, T=T: f'S0-T{T}')})
        for name, (x, y) in pairs.items():
            out['descriptive_pairs'][f'{name}-T{T}'] = {d: paired(lab[sl], fresh[x(d)]['predictions'][sl], fresh[y(d)]['predictions'][sl]) for d in CAL}
        out['descriptive_pairs'][f'twin-vs-teacher-T{T}'] = paired(lab[sl], fresh[f'twin-T{T}']['predictions'][sl], fresh['teacher']['predictions'][sl])
    return out


def claims(an, Ts):
    """Frozen claim flags; QA is None (not tested) when the selected setting is the registered one."""
    q = an['decisions']; R = lambda k: (q[k]['status'] == 'replicated') if k in q else None
    out = {}
    for T in Ts:
        s = an['selection'][f'T{T}']
        out[f'selected_T{T}'] = {'alpha': s['selected']['alpha'], 'passes': s['selected']['passes']}; out[f'selected_is_registered_T{T}'] = s['selected_is_registered']
        for k, name in [('QA', 'selected_better_than_registered'), ('QB', 'P0_96_better_than_selected'), ('QC', 'P0_96_better_than_registered'),
                        ('QD', 'P0_96_better_than_M0_96'), ('QE', 'P0_96_better_than_release_default')]:
            out[f'claim_{name}_T{T}_replicated'] = R(f'{k}-T{T}')
            if f'{k}-T{T}' in q:
                out[f'{k}_T{T}_reading_e25'] = q[f'{k}-T{T}']['reading_e25']; out[f'{k}_T{T}_status_e16'] = q[f'{k}-T{T}']['status']
    return out


def split_metrics(pred, lab, n):
    hs = halves(n)
    return {a: {h: {'correct': correct_on(pred[a]['fresh'], lab, sl), 'n': sl.stop - sl.start, 'ce': ce_on(pred[a]['fresh'], lab, sl)} for h, sl in hs.items()} for a in pred}


def solver_summary(solver):
    """E20's solver summary for P0 arms (per step) and the M0 arms' fit size and time."""
    out = {}
    for arm, s in solver.items():
        if s['objective'] == 'P0':
            out[arm] = {'fit_images': s['fit_images'],
                        'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'], v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in s['layers'].values()) for t in range(s['T'])],
                        'above_range_channels_by_step': [sum(int(v['steps'][t]['above_range'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                        'below_range_channels_by_step': [sum(int(v['steps'][t]['below_range'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                        'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                        'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in s['layers'].values()]) for t in range(s['T'])],
                        'solve_seconds': sum(v['seconds'] for v in s['layers'].values())}
        else:
            out[arm] = {'fit_images': s['fit_images'], 'objective': 'M0', 'solve_seconds': sum(v['seconds'] for v in s['layers'].values())}
    return out


def ftbc_summary(ftbc):
    keys = ['variant', 'recipe', 'alpha', 'passes_total', 'passes_this_call', 'batches', 'batch_size', 'updates_total', 'nominal_gain', 'start', 'seed',
            'max_abs_bias_over_theta', 'ann', 'ann_digest', 'ann_level', 'seconds', 'inputs_sha256']
    return {a: {k: v[k] for k in keys if k in v} | {'images': len(v['images']), 'ann_calls': v['calls']['ann'], 'snn_calls': v['calls']['snn']} for a, v in ftbc.items()}


def excluded_ids(dataset, sources):
    """The exclusion set of prepare_inputs.py, recomputed from the sources (IDs)."""
    main, old, exposure = [json.loads(Path(p).read_text()) for p in sources]
    roles = main['datasets'][{'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100'}[dataset]]['roles']
    ex = set(roles['development']['ids']) | set(roles['confirmation']['ids'])
    if dataset == 'cifar100':
        s = old['settings']['cifar100_r20_t2']; ex |= set(s['tuning']) | set(s['confirmation']) | set(exposure['exclude_from_future_study_confirmation']['cifar100_r20_t2'])
    return ex


def excluded_content(dataset, prep, ex):
    """Content hashes of the excluded training rows, from the official pickles (file SHA-256 from preparation.json)."""
    import pickle
    import numpy as np
    data = []
    for rel, h in prep['datasets'][dataset]['train_files'].items():
        p = Path('/data3/snn_prjs/ftbc_neural_networks/code/parallel-conversion/downloaded_dataset') / rel; assert sha(p) == h['sha256'], rel
        with p.open('rb') as f:
            data.append(np.asarray(pickle.load(f, encoding='bytes')[b'data'], dtype=np.uint8))
    data = np.concatenate(data); assert data.shape == (50000, 3072)
    return {hashlib.sha256(data[i].tobytes()).hexdigest() for i in ex}


def check(tag, which):
    dataset, arch = TAGS[tag]
    c = json.loads((E / 'config' / f'{tag}-{which}.json').read_text()); o = Path(c['output']); w = Path(c['workspace']); Ts = c['Ts']; sizes = c['p0_sizes']; small = c['small_n']
    assert (c['dataset'], c['arch']) == (dataset, arch)
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, p
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    res = json.loads((o / 'result.json').read_text()); assert json.loads((o / 'config.json').read_text()) == c, 'run config differs from the frozen config'
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['release_update_restored']
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower() and res['Ts'] == Ts and res['alphas'] == ALPHAS == c['alphas']
    hw = res['hardware']; assert hw['memory_cap_bytes'] == c['gpu_mem_cap_gib'] * 2 ** 30 and hw['memory_fraction'] == hw['memory_cap_bytes'] / hw['total_memory']   # initial review N-e
    assert sorted(hw['versions']) == ['PIL', 'numpy', 'torchvision']                                                                # N-j
    arms = expected_arms(Ts, sizes, small); assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    # Source identity.
    assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
    src = RUN.renamed_cifar_state(c['source_checkpoint'], c['checkpoint_key']); sdig = state_dict_digest(src)
    assert res['teacher_digest'] == sdig and set(res['proxy_digests']) == {f'T{T}' for T in Ts} == set(res['snn_digests'])
    assert all(v == sdig for v in list(res['proxy_digests'].values()) + list(res['snn_digests'].values()))
    names = res['interfaces']; L = len(names); assert L == c['interfaces'] and set(res['channels']) == set(names)
    assert names == [k[:-len('.thresh')] for k in src if k.endswith('.thresh')]
    # Data identity (section 7 item 3).
    assert sha(c['preparation']) == c['preparation_sha256'] and sha(c['rows_manifest']) == c['rows_manifest_sha256']
    prep = json.loads(Path(c['preparation']).read_text()); assert prep['passed'] and prep['datasets'][dataset]['manifest_sha256'] == c['rows_manifest_sha256']
    for p, h in prep['exclusion_sources'].items():
        assert sha(p) == h, p
    man = json.loads(Path(c['rows_manifest']).read_text()); dev = man['development'][:c['dev_count']]; draws = {k: man[k][:c['draw_n']] for k in CAL}
    allrows = man['development'] + [r for k in CAL for r in man[k]]
    assert len({int(r['index']) for r in allrows}) == len(allrows) == len({r['sha256'] for r in allrows})
    ex = excluded_ids(dataset, list(prep['exclusion_sources'])); assert not {int(r['index']) for r in allrows} & ex
    assert len(ex) == prep['datasets'][dataset]['excluded_ids'] and not {r['sha256'] for r in allrows} & excluded_content(dataset, prep, ex)
    n = c['dev_count']; assert n % 2 == 0 and len(dev) == n and res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    assert small == 3 * c['ftbc_batch'] and sizes[1] == small == c['m0_size'] and max(sizes) <= c['draw_n'] and c['release_n'] == 3 * c['release_batch'] <= c['draw_n']
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    lab = torch.tensor([int(r['label']) for r in dev]); ids = torch.tensor([int(r['index']) for r in dev])
    for arm in arms:
        q = pred[arm]['fresh']
        assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
        assert q['logits'].shape == (n, c['num_classes'])
        correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
        r = recs[arm]['metrics']['fresh']; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
        metrics[arm] = {'fresh': {'correct': correct, 'n': n, 'top1': 100 * correct / n, 'ce': ce}}
    msplit = split_metrics({a: pred[a] for a in arms}, lab, n)
    # States, biases, strict reload, targets, solver records, FTBC grid and release-default (items 4-6).
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text())
    states = {}
    for T in Ts:
        st = tload(o / f'state-T{T}.pt'); assert st['T'] == T and st['source_checkpoint_sha256'] == c['source_checkpoint_sha256'] and st['state_digest'] == sdig
        assert set(st['state_dict']) == set(src) and all(torch.equal(st['state_dict'][k], src[k]) for k in src), T
        states[T] = sha(o / f'state-T{T}.pt'); del st
    bias, hashes = {}, {}
    snn_arms = [a for a in arms if a != 'teacher' and not a.startswith('twin')]
    for a in arms:
        if a == 'teacher' or a.startswith('twin'):
            assert recs[a].get('ann') is True, a
    for arm in snn_arms:
        f = tload(o / f'bias-{arm}.pt'); assert recs[arm]['strict_reload'] and f['source_checkpoint_sha256'] == c['source_checkpoint_sha256']
        Tx = f['T']; assert Tx == int(arm.split('-T')[1].split('-')[0]) and list(f['runtime_biases']) == names and 'state_dict' not in f
        assert f['state_file'] == f'state-T{Tx}.pt' and f['state_sha256'] == states[Tx] and f['state_digest'] == sdig, arm
        assert all(len(f['runtime_biases'][nm]) == Tx and all(b.shape == (res['channels'][nm],) and torch.isfinite(b).all() for b in f['runtime_biases'][nm]) for nm in names)
        bias[arm] = f['runtime_biases']; hashes[arm] = bias_hash(f['runtime_biases'], names, Tx)
        assert hashes[arm] == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v)
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    assert set(res['targets_sha256']) == {f'{d}-{m}-teacher' for d in CAL for m in sizes}, sorted(res['targets_sha256'])
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; d, m = key.split('-')[0], int(key.split('-')[1])
        assert f['ids'].tolist() == [int(x['index']) for x in draws[d][:m]] and f['target'] == 'teacher' and f['network_digest'] == sdig and f['level'] == [c['source_level']], key
        assert list(f['means']) == names and hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt, key
    for arm in [a for a in arms if a.startswith(('P0', 'M0'))]:
        kind = arm.split('-')[0]; m, Tx, d = int(arm.split('-')[1]), int(arm.split('-T')[1].split('-')[0]), arm.split('-')[-1]; s = solver[arm]; tk = f'{d}-{m}-teacher'
        assert s['full_network_replay_equal'] and s['fit_images'] == m and s['T'] == Tx and s['objective'] == kind and s['per_step'] == (kind == 'P0'), arm
        assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == kind, arm
        f = tfiles[tk]; assert list(s['layers']) == list(f['means']) == names
        for nm, info in s['layers'].items():
            if kind == 'P0':
                b = info['b']; assert torch.equal(info['target_mean'], f['means'][nm]) and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta']), (arm, nm)
                assert all(torch.equal(bias[arm][nm][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
            else:                                       # M0: one time-shared channel bias, the same at every step
                assert all(torch.equal(bias[arm][nm][t], bias[arm][nm][0]) for t in range(Tx)), (arm, nm)
                assert within(torch.stack(bias[arm][nm]), c['bound_ratio'] * info['theta']), (arm, nm)
    assert sorted(solver) == sorted(a for a in arms if a.startswith(('P0', 'M0'))) and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32
    grid = [a for a in arms if a.startswith('G-')]; rel = [a for a in arms if a.startswith('R0')]
    extra = sorted(set(ftbc) - set(grid) - set(rel))
    assert extra == sorted([f'G1frozen-T{T}-C1' for T in Ts] + [f'G3single-T{T}-C1' for T in Ts]), extra
    for arm in grid:
        _, alpha, passes, Tpart, d = arm.split('-'); alpha, passes, Tx = float(alpha), int(passes), int(Tpart[1:]); v = ftbc[arm]; calls = v['calls']
        zh = bias_hash({nm: [torch.zeros_like(b) for b in bias[arm][nm]] for nm in names}, names, Tx)
        this = 1 if passes == 1 else 2; start = 'zero' if passes == 1 else grid_arm(alpha, 1, Tx, d)
        assert v['variant'] == 'released' and v['recipe'] == 'registered-grid' and v['alpha'] == alpha and v['T'] == Tx and v['batch_size'] == c['ftbc_batch'], arm
        assert v['passes_this_call'] == this and v['batches'] == 3 * this and v['batches_per_pass'] == 3 and v['passes_total'] == passes and v['updates_total'] == 3 * passes, arm
        assert v['start'] == start and v['init_bias_sha256'] == (zh if passes == 1 else hashes[start]) and v['bias_sha256'] == hashes[arm], arm
        assert v['nominal_gain'] == round(alpha * 3 * passes, 10) and v['images'] == [int(x['index']) for x in draws[d][:small]], arm
        assert calls['ann'] == calls['snn'] == L * v['batches'] and sorted(calls['updates']) == sorted(names) and all(x == v['batches'] for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == sdig and v['ann_level'] == [c['source_level']], arm
    for T in Ts:
        for d in CAL:
            assert len({ftbc[grid_arm(a, p, T, d)]['inputs_sha256'] for a in ALPHAS for p in PASSES}) == 1, (T, d)
    for k, d in enumerate(CAL, 1):                      # release-default FTBC (section 3, R0)
        seeds = set()
        for T in Ts:
            arm = f'R0-T{T}-{d}'; v = ftbc[arm]; calls = v['calls']
            zh = bias_hash({nm: [torch.zeros_like(b) for b in bias[arm][nm]] for nm in names}, names, T)
            assert v['variant'] == 'released' and v['recipe'] == 'release-default' and v['alpha'] == c['release_alpha'] == 0.4 and v['batches'] == 3, arm
            assert v['batch_size'] == c['release_batch'] and v['seed'] == c['release_seed'] + k and v['init_bias_sha256'] == zh and v['bias_sha256'] == hashes[arm], arm
            assert v['images'] == [int(x['index']) for x in draws[d][:c['release_n']]] and sorted(i for b in v['batch_ids'] for i in b) == sorted(v['images']), arm
            assert all(len(b) == c['release_batch'] for b in v['batch_ids']) and v['T'] == T, arm
            assert calls['ann'] == calls['snn'] == L * 3 and sorted(calls['updates']) == sorted(names) and all(x == 3 for x in calls['updates'].values()), arm
            assert v['ann'] == 'teacher' and v['ann_digest'] == sdig and v['ann_level'] == [c['source_level']], arm
            seeds.add((v['seed'], v['inputs_sha256'], json.dumps(v['batch_ids'])))
        assert len(seeds) == 1, (d, 'the release-default batches must be the same at every T')
    eq = res['equivalence']; assert eq and set(eq) == {f'T{T}-C1' for T in Ts}, eq      # section 7 item 5 (sanity and main)
    for T in Ts:
        e = eq[f'T{T}-C1']
        assert e['grid_one_pass_sha256'] == e['frozen_registered_sha256'] == hashes[grid_arm(0.4, 1, T, 'C1')], (T, e)
        assert e['continued_sha256'] == e['single_call_sha256'] == hashes[grid_arm(0.4, 3, T, 'C1')], (T, e)
    manuscript = manuscript_check(w / 'manuscript', json.loads((E / 'preservation-before.json').read_text()), hard=False)  # E24 follow-up 1 R1: recorded in sanity and main; verify_preservation.py decides
    an = analysis(Ts, sizes, small, {a: pred[a] for a in arms}, lab, n, msplit)
    out = {'passed': True, 'job': tag, 'dataset': dataset, 'arch': arch, 'Ts': Ts,
           'scope': 'sanity only' if c['sanity'] else 'CIFAR development screen on train-derived dev-L (selection half, test half); not confirmation or method claim',
           'metrics': metrics, 'metrics_split': msplit, **an, 'equivalence': eq, 'solver_summary': solver_summary(solver), 'ftbc': ftbc_summary(ftbc),
           'timing_seconds': res['timing_seconds'],
           'resources': {k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
                        {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'],
                         'memory_cap_bytes': res['hardware']['memory_cap_bytes'], 'versions': res['hardware']['versions']},
           'manuscript': manuscript, 'not_method_claim': True, 'cuda_initialized': torch.cuda.is_initialized()}
    out['claims'] = claims(an, Ts)
    assert not out['cuda_initialized']
    (E / f'{tag}-{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'passed': True, 'job': tag, 'claims': out['claims']}, indent=1))


if __name__ == '__main__':
    check(sys.argv[1], sys.argv[2])
