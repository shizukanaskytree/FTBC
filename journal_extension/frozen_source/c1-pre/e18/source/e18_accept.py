"""E18 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e18_accept.py {sanity,main}.
Statistics (exact one-sided McNemar, decide rule) come unchanged from the frozen E14 acceptance module; status and state_dict_digest from E16."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E18 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e18_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e18_e16_accept', E.parent / 'e16/source/e16_accept.py')
R18 = _load('e18_runner_helpers', E / 'source/e18_runner.py')
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]
QUESTIONS = {'Q1': ('P0-T4-{}', 'M0-T4-{}'), 'Q2': ('P0-T8-{}', 'M0-T8-{}'), 'Q3': ('P0-T4-{}s', 'F0-T4-{}s'), 'Q4': ('P0-T8-{}s', 'F0-T8-{}s'),
             'Q5': ('P0-T4-{}e', 'FE-T4-{}e'), 'Q6': ('P0-T8-{}e', 'FE-T8-{}e')}
SECOND_MODEL = ['Q1', 'Q2', 'Q3', 'Q4']                  # the combined statement of section 5 (unchanged by Q5/Q6)


def expected_arms():
    arms = ['teacher'] + [f'twin-T{T}' for T in TS] + [f'S0-T{T}' for T in TS]
    arms += [f'{k}-T{T}-{d}' for d in CAL for T in TS for k in ['M0', 'P0']] + [f'{k}-T{T}-{d}s' for d in CAL for T in TS for k in ['P0', 'F0']]
    arms += [f'{k}-T{T}-{d}e' for d in CAL for T in TS for k in ['P0', 'FE']]
    return arms


def claims(main):
    """Section 5: each question reads on its own; the combined second-model statement needs Q1-Q4 replicated (Q5/Q6 read separately).
    Q5/Q6 compare with the ECCV schedule run with the released update rule under the declared same-target adaptation; if FE falls below S0
    (overcorrection) or below the same-draw registered F0 (worse than the registered recipe) in any draw at that T, no method statement."""
    R = lambda k: main['decisions'][k]['status'] == 'replicated'
    below = {T: any(main['fe_below_s0_by_draw'][f'T{T}'].values()) for T in TS}
    worse = {T: any(main['fe_below_f0_by_draw'][f'T{T}'].values()) for T in TS}
    return {k: main['decisions'][k]['status'] for k in QUESTIONS} | {
        'FE_below_S0_by_draw': main['fe_below_s0_by_draw'], 'FE_below_S0_any_draw_T4': below[4], 'FE_below_S0_any_draw_T8': below[8],
        'FE_below_F0_by_draw': main['fe_below_f0_by_draw'], 'FE_below_F0_any_draw_T4': worse[4], 'FE_below_F0_any_draw_T8': worse[8],
        'reading_Q5_ECCV_schedule_overcorrects': below[4], 'reading_Q6_ECCV_schedule_overcorrects': below[8],
        'reading_Q5_ECCV_schedule_worse_than_registered': worse[4], 'reading_Q6_ECCV_schedule_worse_than_registered': worse[8],
        'claim_P0_better_than_M0_VGG16_T4_replicated': R('Q1'), 'claim_P0_better_than_M0_VGG16_T8_replicated': R('Q2'),
        'claim_P0_better_than_registered_FTBC_VGG16_T4_replicated': R('Q3'), 'claim_P0_better_than_registered_FTBC_VGG16_T8_replicated': R('Q4'),
        'claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T4_replicated': R('Q5') and not below[4] and not worse[4],
        'claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T8_replicated': R('Q6') and not below[8] and not worse[8],
        'claim_P0_advantage_carries_to_second_model_at_T4_and_T8': all(R(k) for k in SECOND_MODEL)}


def check(which):
    avg = lambda vals: sum(vals) / len(vals)
    c = json.loads((E / 'config' / f'{which}.json').read_text()); o = Path(c['output']); w = Path(c['workspace'])
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, p
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    res = json.loads((o / 'result.json').read_text()); assert json.loads((o / 'config.json').read_text()) == c, 'run config differs from the frozen config'
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['twin_unchanged']
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
    arms = expected_arms(); assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    # Source identity: every network the run built has exactly the renamed VGG-16 Q16 checkpoint parameters.
    assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
    src = R18.renamed_source_state(c['source_checkpoint']); sdig = state_dict_digest(src)
    assert res['teacher_digest'] == sdig and set(res['proxy_digests']) == {f'T{T}' for T in TS} == set(res['snn_digests'])
    assert all(v == sdig for v in list(res['proxy_digests'].values()) + list(res['snn_digests'].values()))
    names = res['interfaces']; assert len(names) == c['interfaces'] and set(res['channels']) == set(names)
    assert names == [k[:-len('.thresh')] for k in src if k.endswith('.thresh')], 'interfaces differ from the checkpoint thresholds'
    # Data identity (section 7 item 3).
    roles = json.loads(Path(c['dataset_manifest']).read_text()); assert sha(c['dataset_manifest']) == c['dataset_manifest_sha256']
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
    rows = {'fresh': json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
    old = [r for v in roles.values() for r in v] + [r for v in json.loads(Path(c['e5_roles']).read_text()).values() for r in v]
    old += [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
    old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
    cal = json.loads(Path(c['calibration_manifest']).read_text()); allc = [r for k in CAL for r in cal[k]]
    for group in [rows['fresh'], allc]:
        assert not {int(r['index']) for r in group} & {int(r['index']) for r in old} and not {r['sha256'] for r in group} & {r['sha256'] for r in old}
    assert not {int(r['index']) for r in allc} & {int(r['index']) for r in rows['fresh']} and len({int(r['index']) for r in allc}) == len(allc)
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; small = c['small_n']; ne = c['eccv_n']
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    assert ne == c['eccv_batch'] * c['eccv_batches'] < c['draw_n'] and c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2
    rowsets = {**draws, **{f'{d}s': v[:small] for d, v in draws.items()}, **{f'{d}e': v[:ne] for d, v in draws.items()}}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    # Deployed states, biases, solver records and FTBC (items 4-6).
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    states = {}                                            # one saved non-bias state per T, referenced by every arm of that T
    for T in TS:
        st = tload(o / f'state-T{T}.pt'); assert st['T'] == T and st['source_checkpoint_sha256'] == c['source_checkpoint_sha256'] and st['state_digest'] == sdig
        assert set(st['state_dict']) == set(src) and all(torch.equal(st['state_dict'][k], src[k]) for k in src), T      # non-bias state = source
        states[T] = sha(o / f'state-T{T}.pt'); del st
    for arm in [a for a in arms if a != 'teacher' and not a.startswith('twin')]:
        f = tload(o / f'bias-{arm}.pt'); assert recs[arm]['strict_reload'] and f['source_checkpoint_sha256'] == c['source_checkpoint_sha256']
        Tx = f['T']; assert Tx == int(arm.split('-T')[1].split('-')[0]) and list(f['runtime_biases']) == names and 'state_dict' not in f
        assert f['state_file'] == f'state-T{Tx}.pt' and f['state_sha256'] == states[Tx] and f['state_digest'] == sdig, arm
        assert all(len(f['runtime_biases'][n]) == Tx and all(b.shape == (res['channels'][n],) and torch.isfinite(b).all() for b in f['runtime_biases'][n]) for n in names)
        bias[arm] = f['runtime_biases']
        assert bias_hash(f['runtime_biases'], names, Tx) == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v); continue
        if arm.startswith(('F0', 'FE')):
            continue
        kind = arm.split('-T')[0]; key = arm.split('-')[-1]; s = solver[arm]
        assert s['full_network_replay_equal'] and s['fit_images'] == len(rowsets[key]) and s['T'] == Tx and s['objective'] == kind, arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == kind, arm
        assert list(s['layers']) == list(tfiles[tk]['means']) == names
        for n, info in s['layers'].items():
            th = info['theta']; assert torch.equal(info['target_mean'], tfiles[tk]['means'][n]), (arm, n)
            if kind == 'M0':
                beta = info['beta'].to(torch.float32); assert within(beta, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], torch.zeros_like(beta) - beta) for t in range(Tx)), (arm, n)
            else:
                b = info['b']; assert b.shape[0] == Tx and within(b, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    recipes = {'F0': ('registered', c['ftbc_alpha'], c['ftbc_batch'], 3, 's'), 'FE': ('eccv-schedule', c['eccv_alpha'], c['eccv_batch'], c['eccv_batches'], 'e')}
    checks = {f'{k}-T{T}-{d}{r[4]}': ([int(x['index']) for x in rowsets[f'{d}{r[4]}']], T, r) for d in CAL for T in TS for k, r in recipes.items()}
    assert sorted(ftbc) == sorted(checks)
    L = len(names)
    for arm, (images, T, (recipe, alpha, bs, nb, _)) in checks.items():
        v = ftbc[arm]; calls = v['calls']; zh = bias_hash(bias[f'S0-T{T}'], names, T)
        assert v['recipe'] == recipe and v['batches'] == nb and v['alpha'] == alpha and v['batch_size'] == bs and v['init_bias_sha256'] == zh and v['T'] == T, arm
        assert calls['ann'] == calls['snn'] == L * nb and sorted(calls['updates']) == sorted(names) and all(x == nb for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == sdig and v['ann_level'] == [c['source_level']] and v['images'] == images and len(images) == bs * nb, arm
    want = {f'{d}{x}-teacher' for d in CAL for x in ['', 's', 'e']}; assert set(res['targets_sha256']) == want
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; base = key[:-len('-teacher')]
        assert f['ids'].tolist() == [int(x['index']) for x in rowsets[base]] and f['target'] == 'teacher' and f['network_digest'] == sdig and f['level'] == [c['source_level']], key
        assert list(f['means']) == names and hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    M = w / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-E; not confirmation or method claim', 'metrics': metrics, 'decisions': {}, 'pairs': {}}
    for k, (x, y) in QUESTIONS.items():
        d = {dd: decide(pr('fresh', x.format(dd), y.format(dd)), ce(x.format(dd)), ce(y.format(dd))) for dd in CAL}
        out['decisions'][k] = {'per_draw': d, 'status': status([v['reading'] for v in d.values()]), 'candidate': x, 'control': y}
    for role in c['score_roles']:
        out['pairs'][role] = {f'{x.format(d)}-vs-{y.format(d)}': pr(role, x.format(d), y.format(d)) for d in CAL for x, y in
                              list(QUESTIONS.values()) + [('P0-T4-{}', 'P0-T4-{}s'), ('P0-T8-{}', 'P0-T8-{}s'), ('M0-T4-{}', 'F0-T4-{}s'), ('M0-T8-{}', 'F0-T8-{}s'),
                                                          ('F0-T4-{}s', 'S0-T4'), ('F0-T8-{}s', 'S0-T8'), ('P0-T8-{}', 'P0-T4-{}'),
                                                          ('P0-T4-{}', 'FE-T4-{}e'), ('P0-T8-{}', 'FE-T8-{}e'), ('P0-T4-{}s', 'FE-T4-{}e'), ('P0-T8-{}s', 'FE-T8-{}e'),
                                                          ('FE-T4-{}e', 'F0-T4-{}s'), ('FE-T8-{}e', 'F0-T8-{}s'), ('FE-T4-{}e', 'S0-T4'), ('FE-T8-{}e', 'S0-T8'),
                                                          ('P0-T4-{}', 'P0-T4-{}e'), ('P0-T8-{}', 'P0-T8-{}e')]}
        for x, y in [('twin-T4', 'teacher'), ('twin-T8', 'teacher'), ('S0-T8', 'S0-T4')]:
            out['pairs'][role][f'{x}-vs-{y}'] = pr(role, x, y)
    pat = {f'{k}-T{T}': f'{k}-T{T}-{{}}' for T in TS for k in ['M0', 'P0']} | {f'{k}-T{T}-96': f'{k}-T{T}-{{}}s' for T in TS for k in ['P0', 'F0']}
    pat |= {f'{k}-T{T}-100': f'{k}-T{T}-{{}}e' for T in TS for k in ['P0', 'FE']}
    out['draw_spread'] = {s: (lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)})([metrics[p.format(d)]['fresh']['correct'] for d in CAL]) for s, p in pat.items()}
    out['prefix_consistency_gpu'] = {f'P0-{d}{x}': all(torch.equal(bias[f'P0-T4-{d}{x}'][n][t], bias[f'P0-T8-{d}{x}'][n][t]) for n in names for t in range(4))
                                     for d in CAL for x in ['', 's', 'e']}
    out['fe_below_s0_by_draw'] = {f'T{T}': {d: metrics[f'FE-T{T}-{d}e']['fresh']['correct'] < metrics[f'S0-T{T}']['fresh']['correct'] for d in CAL} for T in TS}
    out['fe_below_f0_by_draw'] = {f'T{T}': {d: metrics[f'FE-T{T}-{d}e']['fresh']['correct'] < metrics[f'F0-T{T}-{d}s']['fresh']['correct'] for d in CAL} for T in TS}
    for d in CAL:                                      # same images in the same order across T within each recipe
        for k, x in [('F0', 's'), ('FE', 'e')]:
            assert ftbc[f'{k}-T4-{d}{x}']['inputs_sha256'] == ftbc[f'{k}-T8-{d}{x}']['inputs_sha256'], (k, d)
    theta = {n: float(src[n + '.thresh']) for n in names}
    out['ftbc'] = {a: {k: v[k] for k in ['recipe', 'ann', 'ann_level', 'batches', 'alpha', 'batch_size', 'T', 'max_abs_bias_over_theta', 'inputs_sha256']} |
                   {'mean_abs_b_over_theta_by_layer': {n: avg([float(b.abs().mean()) for b in bias[a][n]]) / theta[n] for n in names}} for a, v in ftbc.items()}
    summary = {}
    for arm, s in solver.items():
        Ls = s['layers']; Tx = s['T']
        if s['objective'] == 'P0':
            summary[arm] = {'kind': 'P0', 'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'], v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in Ls.values()) for t in range(Tx)],
                            'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'above_range_channels_by_step': [sum(int(v['steps'][t]['above_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'below_range_channels_by_step': [sum(int(v['steps'][t]['below_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in Ls.values()]) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_layer': {n: float(v['b'].abs().mean()) / v['theta'] for n, v in Ls.items()},
                            'solve_seconds': sum(v['seconds'] for v in Ls.values())}
        else:
            summary[arm] = {'kind': 'M0', 'in_range_channels_beyond_one_quantum': sum(beyond(v['normalized_residual'], v['observations_per_channel'], v['above_range'], v['below_range']) for v in Ls.values()),
                            'at_bound_channels': sum(int(v['at_bound'].sum()) for v in Ls.values()),
                            'mean_abs_beta_over_theta': avg([float(v['beta'].abs().mean()) / v['theta'] for v in Ls.values()]), 'solve_seconds': sum(v['seconds'] for v in Ls.values()),
                            'mean_abs_beta_over_theta_by_layer': {n: float(v['beta'].abs().mean()) / v['theta'] for n, v in Ls.items()}}
    out.update(claims={}, solver_summary=summary, resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    out['claims'] = claims(out)
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: {kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk == 'per_draw' else vv) for kk, vv in v.items()}
                           for k, v in out['decisions'].items()}, 'claims': out['claims']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
