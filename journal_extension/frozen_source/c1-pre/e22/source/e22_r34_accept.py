"""E22 ResNet-34 child: CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e22_r34_accept.py {sanity,main}.
Adapted from the frozen E20 R34 acceptance (same checks for the shared machinery); statistics (exact one-sided McNemar, decide rule) come
unchanged from the frozen E14 acceptance module; status and state_dict_digest from E16; the four-way reading is the E20 one, copied."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E22 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e22r_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e22r_e16_accept', E.parent / 'e16/source/e16_accept.py')
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL, TS, MODEL, L = ['C1', 'C2', 'C3'], [4, 8], 'r34', 33
TPP_LAYER = E / 'third_party/tpp-drive-20261010/Models/layer.py'      # read as bytes only, never imported or executed


READINGS = ['candidate better', 'no difference detected', 'candidate worse']
PAIR = {'C1': 0, 'C2': 1, 'C3': 2}                    # section 5: draw k is paired with TPP seed k - 1; seeds 3 and 4 are descriptive


def git_out(M, *a):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)    # never refreshes or rewrites the index


def manuscript_state(M):
    """Section 7 (manuscript): HEAD and the index entries (mode, blob, stage, path; not the stat cache) decide whether anything was staged or
    committed; the working-tree status and the index file hash are recorded only, because other sessions draft in manuscript/ at the same time
    (2026-10-10 16:52: an untracked journal_outline.md)."""
    return {'HEAD': git_out(M, 'rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git_out(M, 'ls-files', '--stage').encode()).hexdigest(),
            'staged': git_out(M, 'diff', '--cached', '--name-status'), 'status': git_out(M, 'status', '--porcelain=v1', '--untracked-files=all'),
            'index_sha256': sha(M / '.git/index')}


def manuscript_check(M, before):
    cur = manuscript_state(M)
    assert all(cur[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged']), 'manuscript HEAD, index entries or staged changes differ'
    a, b = before['status'].splitlines(), cur['status'].splitlines()
    return {'head_index_entries_staged_unchanged': True, 'status_lines_added': sorted(set(b) - set(a)), 'status_lines_removed': sorted(set(a) - set(b)),
            'index_file_unchanged': cur['index_sha256'] == before['index_sha256']}


def reading_e22(readings):
    """Section 5 four-way reading (E20's reading_e20, unchanged)."""
    b, w = readings.count('candidate better'), readings.count('candidate worse')
    if b == len(readings): return 'replicated'
    if w: return 'failed (candidate worse in a draw)'
    if b: return 'partially replicated (some draws better, none worse)'
    return 'not replicated (no difference detected in any draw)'


def questions():
    """Candidate and control arm names per draw."""
    return {'Q1': (lambda d: f'P0-T4-{d}', lambda d: f'TPP-T4-{PAIR[d]}'), 'Q2': (lambda d: f'P0-T8-{d}', lambda d: f'TPP-T8-{PAIR[d]}')}


def expected_arms(seeds):
    arms = ['teacher'] + [f'twin-T{T}' for T in TS] + [f'S0-T{T}' for T in TS] + [f'TPP-T{T}-{s}' for T in TS for s in seeds]
    for d in CAL:
        arms += [f'P0-T{T}-{d}' for T in TS] + [f'F0-T{T}-{d}' for T in TS]
    return arms


def tpp_better_draws(question):
    """Section 5, the other direction with the symmetric criterion: in a draw TPP is better when p(P0 worse) < 0.05 and TPP's CE is lower."""
    return sum(v['one_sided_p_candidate_worse'] < 0.05 and v['ce_control'] < v['ce_candidate'] for v in question['per_draw'].values())


def claims(main):
    """Section 5: each question reads on its own (four-way reading); a claim needs 'replicated'. The TPP-better claim needs all three draws
    TPP-better under the symmetric criterion; 1-2 such draws are reported as a failed replication of P0 with x/3 draws TPP-better."""
    q = main['decisions']; R = lambda k: q[k]['status'] == 'replicated'
    nodiff = lambda k: all(v['reading'] == 'no difference detected' for v in q[k]['per_draw'].values())
    tb = {k: tpp_better_draws(q[k]) for k in ['Q1', 'Q2']}
    return {f'{k}_status_e16': v['status'] for k, v in q.items()} | {f'{k}_reading_e22': v['reading_e22'] for k, v in q.items()} | {
        'claim_P0_better_than_TPP_T4_replicated': R('Q1'), 'claim_P0_better_than_TPP_T8_replicated': R('Q2'),
        'TPP_better_draws_T4': tb['Q1'], 'TPP_better_draws_T8': tb['Q2'],
        'claim_TPP_better_than_P0_T4_all_draws': tb['Q1'] == 3, 'claim_TPP_better_than_P0_T8_all_draws': tb['Q2'] == 3,
        'reading_Q1_all_draws_no_difference': nodiff('Q1'), 'reading_Q2_all_draws_no_difference': nodiff('Q2')}


def tpp_effective(pred, seeds):
    """Section 7 item 7: every TPP arm's fresh logits differ from S0's at the same T (the swap took effect) and not all seeds give identical
    logits (the sampling took effect)."""
    for T in TS:
        s0 = pred[f'S0-T{T}']['fresh']['logits']; lg = [pred[f'TPP-T{T}-{s_}']['fresh']['logits'] for s_ in seeds]
        assert all(not torch.equal(x, s0) for x in lg), f'a TPP arm equals S0 at T={T}: the swap did not take effect'
        assert any(not torch.equal(lg[0], x) for x in lg[1:]), f'all TPP seeds give identical logits at T={T}: sampling did not take effect'


def check(which):
    avg = lambda vals: sum(vals) / len(vals)
    c = json.loads((E / 'config' / f'{MODEL}-{which}.json').read_text()); o = Path(c['output']); w = Path(c['workspace']); seeds = c['seeds']; QUESTIONS = questions()
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, p
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    res = json.loads((o / 'result.json').read_text()); assert json.loads((o / 'config.json').read_text()) == c, 'run config differs from the frozen config'
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['twin_unchanged']
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower() and res['seeds'] == seeds and res['tpp_forward_restored']
    arms = expected_arms(seeds); assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    # Frozen network files and the networks the run built (section 7 item 4).
    art = json.loads(Path(c['e0_artifacts']).read_text())
    for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt')]:
        assert Path(c[key]).name == name and Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
    fdig = state_dict_digest(tload(c['plain_proxy'])['state_dict'])
    assert res['teacher_digest'] == fdig and set(res['proxy_digests']) == {f'plain-T{T}' for T in TS} and all(v == fdig for v in res['proxy_digests'].values())
    dep = sha(c['plain_deployment']); assert res['deployment_sha256'] == {'plain': dep}
    # Data identity (section 7 item 3).
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
    rows = {'fresh': json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
    e5 = json.loads(Path(c['e5_roles']).read_text())
    old = [r for v in roles.values() for r in v] + [r for v in e5.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
    old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
    cal = json.loads(Path(c['calibration_manifest']).read_text()); allc = [r for k in CAL for r in cal[k]]
    for group in [rows['fresh'], allc]:
        assert not {int(r['index']) for r in group} & {int(r['index']) for r in old} and not {r['sha256'] for r in group} & {r['sha256'] for r in old}
    assert not {int(r['index']) for r in allc} & {int(r['index']) for r in rows['fresh']} and not {r['sha256'] for r in allc} & {r['sha256'] for r in rows['fresh']}
    assert len({int(r['index']) for r in allc}) == len(allc) == len({r['sha256'] for r in allc})
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; small = c['small_n']
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()} and all(len(v) == c['draw_n'] for v in draws.values())
    assert small == 3 * c['ftbc_batch'] < c['draw_n'] and c['ftbc_alpha'] == 0.4 and seeds == sorted(set(seeds)) and set(PAIR.values()) <= set(seeds)
    rowsets = {**{d: v for d, v in draws.items()}, **{f'{d}-{small}': v[:small] for d, v in draws.items()}}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    tpp_effective(pred, seeds)                         # item 7: the swap and the sampling took effect
    # Deployment state, biases, solver records and FTBC calls (items 4-6).
    deploy = tload(c['plain_deployment'])['state_dict']
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    for arm in [a for a in arms if a.startswith('TPP')]:          # TPP arms: no biases, the declared mode, seed and source (section 4)
        t = recs[arm]['tpp']; Tx = int(arm.split('-T')[1].split('-')[0]); s_ = int(arm.split('-')[-1])
        assert recs[arm]['T'] == Tx and t['mode'] == 'prob' and t['seed'] == s_ and t['swapped_modules'] == L and t['reproducible_on_strict_reload_images'], arm
        assert t['copied_from_layer_sha256'] == sha(TPP_LAYER) and t['role_seeds'] == {r_: 1000 * s_ + i for i, r_ in enumerate(c['score_roles'])}, arm
        assert 'bias_sha256' not in recs[arm] and not (o / f'bias-{arm}.pt').exists() and arm not in res['bias_sha256'], arm
    for arm in [a for a in arms if a != 'teacher' and not a.startswith(('twin', 'TPP'))]:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == dep and recs[arm]['strict_reload']
        assert recs[arm]['detail']['state'] == 'plain' and recs[arm]['detail']['deployment_sha256'] == dep, arm
        assert set(f['state_dict']) == set(deploy) and all(torch.equal(f['state_dict'][k], deploy[k]) for k in deploy if 'time_based_bias' not in k), arm
        Tx = f['T']; assert Tx == int(arm.split('-T')[1].split('-')[0]) and len(f['runtime_biases']) == L
        assert all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        assert bias_hash(f['runtime_biases'], names, Tx) == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v); continue
        if not arm.startswith('P0'):
            continue
        key = arm.split('-')[-1]; s = solver[arm]
        assert s['full_network_replay_equal'] and s['per_step'] and s['fit_images'] == len(rowsets[key]) and s['state'] == 'plain' and s['T'] == Tx, arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == 'P0', arm
        assert list(s['layers']) == list(tfiles[tk]['means']) == names
        for n, info in s['layers'].items():
            b = info['b']; assert torch.equal(info['target_mean'], tfiles[tk]['means'][n]) and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta']), (arm, n)
            assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    assert sorted(solver) == sorted(a for a in arms if a.startswith('P0'))
    checks = {f'F0-T{T}-{d}': T for T in TS for d in CAL}; assert sorted(ftbc) == sorted(checks)
    for arm, T in checks.items():
        v = ftbc[arm]; calls = v['calls']; d = arm.split('-')[-1]; zh = bias_hash(bias[f'S0-T{T}'], names, T)
        assert v['variant'] == 'released' and v['recipe'] == 'registered' and v['passes'] == 1 and v['batches_per_pass'] == 3 and v['batches'] == 3, arm
        assert v['alpha'] == c['ftbc_alpha'] and v['batch_size'] == c['ftbc_batch'] and v['init_bias_sha256'] == zh, arm
        assert v['images'] == [int(x['index']) for x in rowsets[f'{d}-{small}']], arm
        assert calls['ann'] == calls['snn'] == L * 3 and len(calls['updates']) == L and all(x == 3 for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == fdig and v['ann_level'] == [8] and v['state'] == 'plain', arm
    for d in CAL:                                      # same images in the same order at both T
        assert ftbc[f'F0-T4-{d}']['inputs_sha256'] == ftbc[f'F0-T8-{d}']['inputs_sha256'], d
    want = {f'{d}-teacher' for d in CAL} | {f'{d}-{small}-teacher' for d in CAL}
    assert set(res['targets_sha256']) == want, sorted(set(res['targets_sha256']) ^ want)
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; base = key[:-len('-teacher')]
        assert f['ids'].tolist() == [int(x['index']) for x in rowsets[base]] and f['target'] == 'teacher' and f['state'] is None, key
        assert f['network_digest'] == fdig and f['level'] == [8] and list(f['means']) == names, key
        assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    manuscript = manuscript_check(w / 'manuscript', json.loads((E / 'preservation-before.json').read_text()))
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'model': MODEL, 'seeds': seeds, 'pairing': PAIR, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-J; TPP baseline, not confirmation or method claim',
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    for k, (x, y) in QUESTIONS.items():
        d = {dd: decide(pr('fresh', x(dd), y(dd)), ce(x(dd)), ce(y(dd))) for dd in CAL}
        rd = [v['reading'] for v in d.values()]
        out['decisions'][k] = {'per_draw': d, 'status': status(rd), 'counts': {r: rd.count(r) for r in READINGS}, 'reading_e22': reading_e22(rd),
                               'candidate': x('Ck'), 'control': 'TPP-' + y('C1').split('-')[1] + '-(k-1)'}
    for role in c['score_roles']:
        pairs = {}
        for T in TS:
            for d in CAL:
                for s_ in seeds:
                    pairs[f'P0-T{T}-{d}-vs-TPP-T{T}-{s_}'] = pr(role, f'P0-T{T}-{d}', f'TPP-T{T}-{s_}')
                pairs[f'P0-T{T}-{d}-vs-F0-T{T}-{d}'] = pr(role, f'P0-T{T}-{d}', f'F0-T{T}-{d}')
                pairs[f'TPP-T{T}-{PAIR[d]}-vs-F0-T{T}-{d}'] = pr(role, f'TPP-T{T}-{PAIR[d]}', f'F0-T{T}-{d}')
                pairs[f'F0-T{T}-{d}-vs-S0-T{T}'] = pr(role, f'F0-T{T}-{d}', f'S0-T{T}')
            for s_ in seeds:
                pairs[f'TPP-T{T}-{s_}-vs-S0-T{T}'] = pr(role, f'TPP-T{T}-{s_}', f'S0-T{T}')
        for s_ in seeds:
            pairs[f'TPP-T8-{s_}-vs-TPP-T4-{s_}'] = pr(role, f'TPP-T8-{s_}', f'TPP-T4-{s_}')
        for x, y in [('twin-T4', 'teacher'), ('twin-T8', 'teacher'), ('S0-T8', 'S0-T4')]:
            pairs[f'{x}-vs-{y}'] = pr(role, x, y)
        out['pairs'][role] = pairs
    spread = lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)}
    out['draw_spread'] = {f'{k}-T{T}': spread([metrics[f'{k}-T{T}-{d}']['fresh']['correct'] for d in CAL]) for k in ['P0', 'F0'] for T in TS}
    out['seed_spread'] = {f'TPP-T{T}': spread([metrics[f'TPP-T{T}-{s_}']['fresh']['correct'] for s_ in seeds]) for T in TS}
    out['prefix_consistency_gpu'] = {f'P0-{d}': all(torch.equal(bias[f'P0-T4-{d}'][m][t], bias[f'P0-T8-{d}'][m][t]) for m in names for t in range(4)) for d in CAL}
    theta = {n: float(tload(o / 'bias-S0-T4.pt')['state_dict'][n + '.thresh']) for n in names}
    out['ftbc'] = {a: {k: v[k] for k in ['variant', 'recipe', 'passes', 'batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta', 'inputs_sha256']} |
                   {'mean_abs_b_over_theta_by_layer': {n: avg([float(b.abs().mean()) for b in bias[a][n]]) / theta[n] for n in names}} for a, v in ftbc.items()}
    out['solver_summary'] = {arm: {'fit_images': s['fit_images'],
                                   'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'], v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in s['layers'].values()) for t in range(s['T'])],
                                   'above_range_channels_by_step': [sum(int(v['steps'][t]['above_range'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                                   'below_range_channels_by_step': [sum(int(v['steps'][t]['below_range'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                                   'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                                   'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in s['layers'].values()]) for t in range(s['T'])],
                                   'solve_seconds': sum(v['seconds'] for v in s['layers'].values())} for arm, s in solver.items()}
    out.update(resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
               manuscript=manuscript, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    out['claims'] = claims(out)
    assert not out['cuda_initialized']
    (E / f'{MODEL}-{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'model': MODEL, 'decisions': {k: {'reading_e22': v['reading_e22'], 'counts': v['counts'], 'net': [x['net'] for x in v['per_draw'].values()]} for k, v in out['decisions'].items()},
             'claims': out['claims']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
