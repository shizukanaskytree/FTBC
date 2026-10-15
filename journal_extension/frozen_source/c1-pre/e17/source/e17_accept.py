"""E17 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e17_accept.py {sanity,main}.
Statistics (exact one-sided McNemar, decide rule) come unchanged from the frozen E14 acceptance module; adoption() applies section 5."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E17 acceptance uses assert; do not run with -O')
_spec = importlib.util.spec_from_file_location('e17_e14_accept', str(E.parent / 'e14/source/e14_accept.py'))
A14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A14)
_spec = importlib.util.spec_from_file_location('e17_e16_accept', str(E.parent / 'e16/source/e16_accept.py'))
A16 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A16)
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL, TS, SOLVERS = ['C1', 'C2', 'C3'], [2, 4], ['P0', 'M0', 'CE', 'CE-A']


def expected_arms():
    arms = ['teacher'] + [f'twin-T{T}' for T in TS] + [f'S0-T{T}' for T in TS]
    arms += [f'{k}-T{T}-{d}' for d in CAL for T in TS for k in SOLVERS] + [f'{k}-T{T}-{d}s' for d in CAL for T in TS for k in ['CE-A', 'F0']]
    return arms


def adoption(main):
    """Section 5: CE-A replaces P0 as the serial solver only if it beats M0 at T=2 in all three draws (Q1) and, at T=4 against P0 (Q2), no draw
    reads 'candidate worse' and the three net differences sum to at least 0 (no consistent sub-threshold loss of the replicated P0 result)."""
    q = main['decisions']; R = lambda k: q[k]['status'] == 'replicated'
    q2_worse = any(v['reading'] == 'candidate worse' for v in q['Q2']['per_draw'].values())
    q2_net = sum(v['net'] for v in q['Q2']['per_draw'].values())
    return {'Q1': q['Q1']['status'], 'Q2': q['Q2']['status'], 'Q2_any_draw_worse': q2_worse, 'Q2_net_sum': q2_net, 'Q3': q['Q3']['status'], 'Q4': q['Q4']['status'],
            'Q5_T2': q['Q5-T2']['status'], 'Q5_T4': q['Q5-T4']['status'],
            'adopt_CE_A_as_serial_solver': R('Q1') and not q2_worse and q2_net >= 0,
            'claim_CE_A_better_than_M0_at_T2_replicated': R('Q1'),
            'claim_CE_A_better_than_P0_at_T4_replicated': R('Q2'),
            'claim_CE_A_not_worse_than_P0_at_T4': not q2_worse and q2_net >= 0,
            'claim_CE_better_than_M0_at_T2_replicated': R('Q3'), 'claim_CE_better_than_P0_at_T4_replicated': R('Q4'),
            'claim_CE_A_better_than_registered_FTBC_same_images_T2_replicated': R('Q5-T2'),
            'claim_CE_A_better_than_registered_FTBC_same_images_T4_replicated': R('Q5-T4')}


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
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    rowsets = {**draws, **{f'{d}s': v[:small] for d, v in draws.items()}}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce,
                                  'mean_max_prob': float(torch.softmax(q['logits'].double(), 1).max(1).values.mean()), 'logit_std': float(q['logits'].double().std(1).mean())}
    # Deployment state, biases and solver records (items 4-6).
    deploy = tload(c['plain_deployment'])['state_dict']
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    near_ties = {}
    for arm in [a for a in arms if a != 'teacher' and not a.startswith('twin')]:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == dep and recs[arm]['strict_reload']
        assert recs[arm]['detail']['state'] == 'plain' and recs[arm]['detail']['deployment_sha256'] == dep, arm
        assert set(f['state_dict']) == set(deploy) and all(torch.equal(f['state_dict'][k], deploy[k]) for k in deploy if 'time_based_bias' not in k), arm
        Tx = f['T']; assert Tx == int(arm.split('-T')[1].split('-')[0]) and len(f['runtime_biases']) == 33
        assert all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        assert bias_hash(f['runtime_biases'], names, Tx) == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v)
            continue
        if arm.startswith('F0'):
            continue
        kind = arm.split('-T')[0]; key = arm.split('-')[-1]; s = solver[arm]
        assert s['full_network_replay_equal'] and s['fit_images'] == len(rowsets[key]) and s['state'] == 'plain' and s['T'] == Tx, arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == kind, arm
        assert list(s['layers']) == list(tfiles[tk]['means']) == names
        for n, info in s['layers'].items():
            th = info['theta']
            assert torch.equal(info['target_mean'], tfiles[tk]['means'][n]), (arm, n)      # the solver used the declared target file
            if kind == 'M0':
                beta = info['beta'].to(torch.float32); assert within(beta, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], torch.zeros_like(beta) - beta) for t in range(Tx)), (arm, n)
            elif kind == 'P0':
                b = info['b']; assert s['per_step'] and b.shape[0] == Tx and within(b, c['bound_ratio'] * th)
                assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
            else:
                assert s['objective'] == kind and s['reference'] == 'teacher' and s['reference_digest'] == fdig, arm
                b = info['b']; assert b.shape[0] == Tx and within(b, c['bound_ratio'] * th) and len(info['steps']) == Tx
                assert all(torch.equal(f['runtime_biases'][n][t], b[t]) and torch.equal(info['steps'][t]['b'], b[t]) for t in range(Tx)), (arm, n)
                rules = [st['rule'] for st in info['steps']]
                if kind == 'CE-A':                        # the anchor target is T*target minus the realized means of the first T-1 steps (same FP64 order)
                    cum = torch.zeros_like(info['target_mean'])
                    for st in info['steps'][:-1]: cum += st['mean_after']
                    assert torch.equal(info['steps'][-1]['anchor_target'], Tx * info['target_mean'] - cum), (arm, n)
                assert rules == ['cumulative-error'] * Tx if kind == 'CE' else rules == ['cumulative-error'] * (Tx - 1) + ['mean-anchor'], (arm, n, rules)
                # the per-unit reference is the declared teacher output on the declared rows: its channel means equal the target file (FP64 sums)
                assert torch.allclose(info['reference_mean'], tfiles[tk]['means'][n], rtol=1e-9, atol=1e-12), (arm, n)
                near_ties[arm] = near_ties.get(arm, 0) + sum(st['near_tie_mismatches'] for st in info['steps'])
    checks = {f'F0-T{T}-{d}s': ([int(x['index']) for x in draws[d][:small]], T) for d in CAL for T in TS}
    assert sorted(ftbc) == sorted(checks)
    for arm, (images, T) in checks.items():
        v = ftbc[arm]; calls = v['calls']; zh = bias_hash(bias[f'S0-T{T}'], names, T)
        assert v['recipe'] == 'registered' and v['batches'] == 3 and v['alpha'] == c['ftbc_alpha'] and v['batch_size'] == c['ftbc_batch'] and v['init_bias_sha256'] == zh
        assert calls['ann'] == calls['snn'] == 33 * 3 and len(calls['updates']) == 33 and all(x == 3 for x in calls['updates'].values())
        assert v['ann'] == 'teacher' and v['ann_digest'] == fdig and v['ann_level'] == [8] and v['state'] == 'plain' and v['images'] == images, arm
    # prefix consistency on the GPU (section 6; reported, exact arithmetic predicts identity for P0 and CE, not for CE-A)
    prefix = {f'{kind}-{d}': all(torch.equal(bias[f'{kind}-T2-{d}'][n][t], bias[f'{kind}-T4-{d}'][n][t]) for n in names for t in range(2))
              for kind in SOLVERS if kind != 'M0' for d in CAL}
    want = {f'{d}-teacher' for d in CAL} | {f'{d}s-teacher' for d in CAL}
    assert set(res['targets_sha256']) == want, sorted(set(res['targets_sha256']) ^ want)
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; base = key.split('-')[0]
        assert f['ids'].tolist() == [int(x['index']) for x in rowsets[base]] and f['target'] == 'teacher' and f['state'] is None, key
        assert f['network_digest'] == fdig and f['level'] == [8] and list(f['means']) == names, key
        assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    M = w / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-D; not confirmation or method claim',
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    def q(x, y):
        d = {k: decide(pr('fresh', x.format(k), y.format(k)), ce(x.format(k)), ce(y.format(k))) for k in CAL}
        return {'per_draw': d, 'status': status([v['reading'] for v in d.values()])}
    out['decisions'] = {'Q1': q('CE-A-T2-{}', 'M0-T2-{}'), 'Q2': q('CE-A-T4-{}', 'P0-T4-{}'), 'Q3': q('CE-T2-{}', 'M0-T2-{}'), 'Q4': q('CE-T4-{}', 'P0-T4-{}'),
                        'Q5-T2': q('CE-A-T2-{}s', 'F0-T2-{}s'), 'Q5-T4': q('CE-A-T4-{}s', 'F0-T4-{}s')}
    for role in c['score_roles']:
        out['pairs'][role] = {f'{x.format(d)}-vs-{y.format(d)}': pr(role, x.format(d), y.format(d)) for d in CAL for x, y in
                              [('CE-A-T2-{}', 'M0-T2-{}'), ('CE-A-T4-{}', 'P0-T4-{}'), ('CE-T2-{}', 'M0-T2-{}'), ('CE-T4-{}', 'P0-T4-{}'),
                               ('CE-A-T2-{}s', 'F0-T2-{}s'), ('CE-A-T4-{}s', 'F0-T4-{}s'), ('P0-T2-{}', 'M0-T2-{}'), ('P0-T4-{}', 'M0-T4-{}'),
                               ('CE-A-T2-{}', 'CE-T2-{}'), ('CE-A-T4-{}', 'CE-T4-{}'), ('CE-A-T2-{}', 'P0-T2-{}'), ('CE-A-T4-{}', 'M0-T4-{}'),
                               ('CE-T2-{}', 'P0-T2-{}'), ('CE-T4-{}', 'M0-T4-{}'), ('CE-A-T2-{}', 'CE-A-T2-{}s'), ('CE-A-T4-{}', 'CE-A-T4-{}s'),
                               ('F0-T2-{}s', 'S0-T2'), ('F0-T4-{}s', 'S0-T4'), ('F0-T2-{}s', 'M0-T2-{}'), ('F0-T4-{}s', 'P0-T4-{}')]}
    pat = {f'{k}-T{T}': f'{k}-T{T}-{{}}' for T in TS for k in SOLVERS} | {f'{k}-T{T}-96': f'{k}-T{T}-{{}}s' for T in TS for k in ['CE-A', 'F0']}
    out['draw_spread'] = {s: (lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)})([metrics[p.format(d)]['fresh']['correct'] for d in CAL]) for s, p in pat.items()}
    out['ftbc'] = {a: {k: v[k] for k in ['recipe', 'ann', 'ann_level', 'batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta', 'inputs_sha256']} for a, v in ftbc.items()}
    summary = {}
    for arm, s in solver.items():
        Ls = s['layers']; Tx = s['T']; kind = s.get('objective') or ('P0' if s.get('per_step') else 'M0')
        if kind == 'P0':
            summary[arm] = {'kind': 'P0', 'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'], v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in Ls.values()) for t in range(Tx)],
                            'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in Ls.values()]) for t in range(Tx)],
                            'solve_seconds': sum(v['seconds'] for v in Ls.values())}
        elif kind == 'M0':
            summary[arm] = {'kind': 'M0', 'in_range_channels_beyond_one_quantum': sum(beyond(v['normalized_residual'], v['observations_per_channel'], v['above_range'], v['below_range']) for v in Ls.values()),
                            'at_bound_channels': sum(int(v['at_bound'].sum()) for v in Ls.values()),
                            'mean_abs_beta_over_theta': avg([float(v['beta'].abs().mean()) / v['theta'] for v in Ls.values()]), 'solve_seconds': sum(v['seconds'] for v in Ls.values())}
        else:
            last = [v['steps'][-1] for v in Ls.values()]
            summary[arm] = {'kind': kind, 'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'cut_at_range_end_channels_by_step': [(sum(int(v['steps'][t]['cut_at_range_end'].sum()) for v in Ls.values()) if 'cut_at_range_end' in next(iter(Ls.values()))['steps'][t] else None) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in Ls.values()]) for t in range(Tx)],
                            'near_tie_mismatches': near_ties.get(arm, 0), 'solve_seconds': sum(v['seconds'] for v in Ls.values()),
                            'mean_rate_minus_target_over_theta': avg([float(((sum(st['mean_after'] for st in v['steps']) / Tx) - v['target_mean']).abs().mean()) / v['theta'] for v in Ls.values()])}
            if kind == 'CE-A':                            # anchor step: in-range channels beyond one count; out-of-range channels and their rate deviation
                summary[arm]['anchor_in_range_channels_beyond_one_quantum'] = sum(beyond(st['normalized_residual'], v['observations_per_channel_per_step'], st['above_range'], st['below_range']) for st, v in zip(last, Ls.values()))
                gaps = {n: ((sum(st['mean_after'] for st in v['steps']) / Tx) - v['target_mean']).abs() / v['theta'] for n, v in Ls.items()}
                outr = {n: v['steps'][-1]['above_range'] | v['steps'][-1]['below_range'] for n, v in Ls.items()}
                summary[arm]['anchor_out_of_range_channels'] = sum(int(o.sum()) for o in outr.values())
                summary[arm]['anchor_out_of_range_max_rate_deviation_over_theta'] = max([float(gaps[n][outr[n]].max()) for n in Ls if outr[n].any()], default=0.0)
                summary[arm]['anchor_in_range_max_rate_deviation_quanta'] = max([float((gaps[n][~outr[n]] * Tx * v['observations_per_channel_per_step']).max()) for n, v in Ls.items() if (~outr[n]).any()], default=0.0)
    out.update(adoption={}, solver_summary=summary, prefix_consistency_gpu=prefix, resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    out['adoption'] = adoption(out)
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: {kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk == 'per_draw' else vv) for kk, vv in v.items()}
                           for k, v in out['decisions'].items()}, 'adoption': out['adoption']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
