"""E14 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e14_accept.py {sanity,main,extra}."""
import hashlib, json, math, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E14 acceptance uses assert; do not run with -O')
ALPHA = 0.05


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def tload(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def upper_tail(k, n):
    """P(X >= k) for X ~ Binomial(n, 1/2); 1.0 when n == 0."""
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def paired(lab, x, y):
    a, b = x == lab, y == lab
    u, v = int((a & ~b).sum()), int((b & ~a).sum())
    return {'x_only_correct': u, 'y_only_correct': v, 'net': u - v, 'prediction_disagreements': int((x != y).sum()),
            'one_sided_p_x_better': upper_tail(u, u + v), 'one_sided_p_y_better': upper_tail(v, u + v)}


def bias_hash(biases, names, T):
    return hashlib.sha256(b''.join(biases[n][t].to(torch.float32).contiguous().numpy().tobytes() for n in names for t in range(T))).hexdigest()


def within(x, limit):
    """|x| <= limit compared in float32, the dtype the solvers compare in."""
    return bool((x.to(torch.float32).abs() <= torch.tensor(limit, dtype=torch.float32)).all())


def beyond(res, n_obs, above, below):
    """Channels inside the reachable range whose |normalized residual| exceeds one count quantum (1/observations); ties can cause this."""
    return int(((res.abs() > (1.0 / n_obs) * (1 + 1e-9)) & ~above & ~below).sum())


def decide(pair, ce_x, ce_y, sign_check=None):
    better = pair['one_sided_p_x_better'] < ALPHA and ce_x < ce_y and (sign_check is None or sign_check > 0)
    worse = pair['one_sided_p_y_better'] < ALPHA
    return {'reading': 'candidate better' if better else ('candidate worse' if worse else 'no difference detected'),
            'net': pair['net'], 'one_sided_p_candidate_better': pair['one_sided_p_x_better'], 'one_sided_p_candidate_worse': pair['one_sided_p_y_better'],
            'ce_candidate': ce_x, 'ce_control': ce_y, 'perturbed_net_sign': sign_check, 'pair': pair}


def expected_arms(c):
    L, S = c['fit_count'], c['fit_small']; arms = []
    if 'main' in c['stages']:
        arms += ['teacher', 'twin-T4', 'S0', f'M0-{L}', f'M0-{L}@b{c["perturb_batch"]}', f'P0-{L}', f'P0-{L}@b{c["perturb_batch"]}',
                 f'M0-{S}', f'P0-{S}'] + [a for a, _, _ in c['ftbc_arms']]
    if 'extra' in c['stages']:
        arms += [a for x in c['extra_T'] for a in [f'twin-T{x}', f'S0-T{x}', f'M0-{L}-T{x}', f'P0-{L}-T{x}']]
    return arms


def check(which):
    c = json.loads((E / 'config' / f'{which}.json').read_text()); o = Path(c['output']); L, S = c['fit_count'], c['fit_small']; pb = c['perturb_batch']
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    res = json.loads((o / 'result.json').read_text())
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['twin_unchanged']
    assert res['stages'] == c['stages'] and res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
    arms = expected_arms(c); assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    Ts = ([4] if 'main' in c['stages'] else []) + (c['extra_T'] if 'extra' in c['stages'] else [])
    assert set(res['proxy_digests']) >= {str(x) for x in Ts}
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
    rows = {'fresh': json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
    used = {int(r['index']) for v in roles.values() for r in v}; assert not {int(r['index']) for r in rows['fresh']} & used
    assert not {r['sha256'] for r in rows['fresh']} & {r['sha256'] for v in roles.values() for r in v}, 'fresh content overlaps E0 roles'
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']; w = Path(c['workspace'])
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():                       # what the Slurm TaskProlog verified equals the freeze (staged: resolved path in workspace)
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    # Biases and solver records (section 7 items 5-8).
    deploy_sha = sha(c['plain_deployment']); solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text())
    bias, names = {}, None
    snn = [a for a in arms if '@' not in a and a not in ('teacher',) and not a.startswith('twin')]
    for arm in snn:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == deploy_sha and recs[arm]['strict_reload']
        Tx = f['T']; assert len(f['runtime_biases']) == 33 and all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        h = bias_hash(f['runtime_biases'], names, Tx); assert h == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v)
        elif arm.startswith('M0'):
            s = solver[arm]; assert s['full_network_replay_equal'] and s['fit_images'] == (S if arm == f'M0-{S}' else L)
            for n, info in s['layers'].items():
                th = info['theta']; beta = info['beta'].to(torch.float32); assert within(beta, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], torch.zeros_like(beta) - beta) for t in range(Tx)), (arm, n)
        elif arm.startswith('P0'):
            s = solver[arm]; assert s['full_network_replay_equal'] and s['per_step'] and s['fit_images'] == (S if arm == f'P0-{S}' else L)
            for n, info in s['layers'].items():
                th = info['theta']; b = info['b']; assert b.shape[0] == Tx and within(b, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)), (arm, n)
                assert len(info['steps']) == Tx and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    for arm in [a for a in arms if '@' in a]:
        base, b = arm.split('@b'); assert recs[arm]['perturbation_of'] == base and recs[arm]['eval_batch_size'] == int(b) == pb
        assert recs[arm]['bias_sha256'] == res['bias_sha256'][base]
    if 'main' in c['stages']:
        zh = bias_hash(bias['S0'], names, 4)
        assert c['ftbc_arms'][0] == ['F0-n3', 3, c['ftbc_alpha']] and sorted(ftbc) == sorted(a for a, _, _ in c['ftbc_arms'])
        for arm, nb, alpha in c['ftbc_arms']:
            v = ftbc[arm]; calls = v['calls']
            assert v['batches'] == nb and v['alpha'] == alpha and v['batch_size'] == c['ftbc_batch'] and v['init_bias_sha256'] == zh
            assert calls['teacher'] == calls['snn'] == 33 * nb and len(calls['updates']) == 33 and all(x == nb for x in calls['updates'].values())
            assert v['images'] == [int(r['index']) for r in roles['fit'][:c['ftbc_batch'] * nb]]
        assert res['diagnostic_ids'] == [int(r['index']) for r in roles['diagnostic'][:c['diag_count']]]
    for n_fit in ([L, S] if 'main' in c['stages'] else [L]):
        tt = tload(o / f'teacher-targets-{n_fit}.pt'); assert tt['ids'].tolist() == [int(r['index']) for r in roles['fit'][:n_fit]]
        assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in tt['means'].values())).hexdigest() == res['teacher_targets_sha256'][str(n_fit)]
    M = Path(c['workspace']) / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Analysis on the fresh development (primary) and dev512 (continuity).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen; fresh2048 primary, dev512 continuity; not confirmation or method claim',
           'stages': c['stages'], 'metrics': metrics}
    if 'main' in c['stages']:
        mL, pL, mS, pS = f'M0-{L}', f'P0-{L}', f'M0-{S}', f'P0-{S}'
        ce = lambda a, role='fresh': metrics[a][role]['ce']
        sign8 = pr('fresh', f'{pL}@b{pb}', f'{mL}@b{pb}')['net']
        decisions = {'Q1': decide(pr('fresh', pL, mL), ce(pL), ce(mL), sign8), 'Q2': decide(pr('fresh', pS, 'F0-n3'), ce(pS), ce('F0-n3')), 'alpha': ALPHA}
        pairs = {}
        for role in c['score_roles']:
            pairs[role] = {f'{x}-vs-{y}': pr(role, x, y) for x, y in [(pL, mL), (pS, 'F0-n3'), ('F0-g1', 'F0-n3'), ('F0-n6-g1.2', 'F0-n3'), ('F0-n3-g2.4', 'F0-n3'), (pS, mS), (mL, mS), (pL, pS),
                                                                       (mL, 'F0-n3'), (pL, 'F0-n3'), (mS, 'F0-n3'), (f'{pL}@b{pb}', f'{mL}@b{pb}'),
                                                                       (f'{mL}@b{pb}', mL), (f'{pL}@b{pb}', pL)] + [(a, 'S0') for a in [mL, pL, mS, pS] + [f for f, _, _ in c['ftbc_arms']]]}
        diag = json.loads((o / 'diagnostics.json').read_text()); step = json.loads((o / 'step-objective.json').read_text())
        layers = list(diag['S0'])
        deep = lambda n: n.startswith('conv4') or n.startswith('conv5')
        avg = lambda vals: sum(vals) / len(vals)
        mech = {}
        for arm in diag:
            mech[arm] = {k: avg([diag[arm][n][k] for n in layers[1:]]) for k in ['over_fraction', 'under_fraction', 'mean_abs_local', 'mean_abs_vs_twin']}
            mech[arm].update({f'deep_{k}': avg([diag[arm][n][k] for n in layers[1:] if deep(n)]) for k in ['over_fraction', 'under_fraction', 'mean_abs_vs_twin']})
            mech[arm].update({'residual_inner_over_fraction': avg([diag[arm][n]['over_fraction'] for n in layers[1:] if 'residual_function' in n]),
                              'merge_over_fraction': avg([diag[arm][n]['over_fraction'] for n in layers[1:] if n.endswith('.act')]),
                              'step_objective_residual': avg([avg(step[arm][n]['step_abs_residual_over_theta']) for n in layers[1:]]),
                              'total_objective_residual': avg([step[arm][n]['total_abs_residual_over_theta'] for n in layers[1:]]),
                              'spike_share_by_step': [avg([diag[arm][n]['spike_share_by_step'][t] for n in layers[1:]]) for t in range(4)]})
        mpred = {'a_P0_lower_step_objective_residual': mech[pL]['step_objective_residual'] < mech[mL]['step_objective_residual'],
                 'b_P0_more_overfiring_risk': mech[pL]['over_fraction'] > mech[mL]['over_fraction'],
                 'c_P0_lower_deep_twin_deviation': mech[pL]['deep_mean_abs_vs_twin'] < mech[mL]['deep_mean_abs_vs_twin']}
        out.update(decisions=decisions, pairs=pairs, mechanism=mech, mechanism_predictions=mpred, diagnostics_per_layer=diag, step_objective_per_layer=step,
                   ftbc={a: {k: v[k] for k in ['batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta']} for a, v in ftbc.items()},
                   anchors_e0_descriptive=res['anchors_e0'], anchors_e13_descriptive=res['anchors_e13'])
    else:
        out['pairs'] = {role: {f'{a}-T{x}-vs-{b}-T{x}': pr(role, f'{a}-T{x}', f'{b}-T{x}') for x in c['extra_T']
                               for a, b in [(f'P0-{L}', f'M0-{L}'), (f'P0-{L}', 'S0'), (f'M0-{L}', 'S0')]} for role in c['score_roles']}
    summary = {}
    for arm, s in solver.items():
        Ls = s['layers']
        if s.get('per_step'):
            Tx = len(next(iter(Ls.values()))['steps'])
            summary[arm] = {'per_step': True, 'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'],
                                                                                                     v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in Ls.values()) for t in range(Tx)],
                            'mean_abs_normalized_residual_by_step': [avg([float(v['steps'][t]['normalized_residual'].abs().mean()) for v in Ls.values()]) for t in range(Tx)],
                            'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'above_range_channels_by_step': [sum(int(v['steps'][t]['above_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'below_range_channels_by_step': [sum(int(v['steps'][t]['below_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in Ls.values()]) for t in range(Tx)],
                            'max_abs_b_over_theta': max(float(v['b'].abs().max()) / v['theta'] for v in Ls.values()),
                            'solve_seconds': sum(v['seconds'] for v in Ls.values())}
        else:
            summary[arm] = {'per_step': False, 'in_range_channels_beyond_one_quantum': sum(beyond(v['normalized_residual'], v['observations_per_channel'], v['above_range'], v['below_range'])
                                                                                          for v in Ls.values()),
                            'mean_abs_normalized_residual': avg([float(v['normalized_residual'].abs().mean()) for v in Ls.values()]),
                            'at_bound_channels': sum(int(v['at_bound'].sum()) for v in Ls.values()),
                            'below_range_channels': sum(int(v['below_range'].sum()) for v in Ls.values()),
                            'above_range_channels': sum(int(v['above_range'].sum()) for v in Ls.values()),
                            'mean_abs_beta_over_theta': avg([float(v['beta'].abs().mean()) / v['theta'] for v in Ls.values()]),
                            'max_abs_beta_over_theta': max(float(v['beta'].abs().max()) / v['theta'] for v in Ls.values()),
                            'solve_seconds': sum(v['seconds'] for v in Ls.values())}
    out.update(solver_summary=summary, resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()}}
    if 'decisions' in out:
        brief['decisions'] = {k: ({kk: vv for kk, vv in v.items() if kk != 'pair'} if isinstance(v, dict) else v) for k, v in out['decisions'].items()}
        brief['mechanism_predictions'] = out['mechanism_predictions']
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
