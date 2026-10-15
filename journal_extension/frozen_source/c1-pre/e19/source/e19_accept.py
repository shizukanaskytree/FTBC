"""E19 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 4-7). No CUDA. Usage: e19_accept.py {sanity,main}.
Statistics (exact one-sided McNemar, decide rule) come unchanged from the frozen E14 acceptance module; status and state_dict_digest from E16."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E19 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e19_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e19_e16_accept', E.parent / 'e16/source/e16_accept.py')
sha, tload, paired, decide, bias_hash, within = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL, TS = ['C1', 'C2', 'C3'], [4, 8]
QUESTIONS = {'D1': ('FA-T4-{}s', 'F0-T4-{}s'), 'D2': ('FD-T4-{}s', 'FA-T4-{}s'), 'D3': ('P0-T4-{}s', 'FD-T4-{}s'),
             'D4': ('P0-T4-{}s', 'FD10-T4-{}s'), 'D5': ('P0-T8-{}s', 'F0-T8-{}s'), 'D6': ('P0-T4-{}e', 'FE-T4-{}e')}
PRIMARY = ['D1', 'D2', 'D3']
# arm prefix -> (variant, recipe, passes); registered recipe on Cks, ECCV schedule on Cke
FTBC_ARMS = {'F0-T4': ('released', 'registered', 1), 'FA-T4': ('FA', 'registered', 1), 'FD-T4': ('FD', 'registered', 1), 'FD10-T4': ('FD', 'registered', 10),
             'F0-T8': ('released', 'registered', 1), 'FE-T4': ('released', 'eccv-schedule', 1)}


def expected_arms():
    arms = ['teacher'] + [f'twin-T{T}' for T in TS] + [f'S0-T{T}' for T in TS]
    for d in CAL:
        arms += [f'{k}-{d}s' for k in ['F0-T4', 'FA-T4', 'FD-T4', 'FD10-T4', 'P0-T4', 'F0-T8', 'P0-T8']] + [f'FE-T4-{d}e', f'P0-T4-{d}e']
    return arms


def claims(main):
    """Section 5: every question reads on its own; the order-dependent decomposition sentence needs D1-D3 replicated; D4 is read against
    ten alpha=0.4 passes only; D6 is the ECCV-schedule comparison and is not a method statement when FE falls below S0 in any draw."""
    q = main['decisions']; R = lambda k: q[k]['status'] == 'replicated'
    nodiff = all(v['reading'] == 'no difference detected' for v in q['D4']['per_draw'].values())
    return {k: q[k]['status'] for k in QUESTIONS} | {
        'claim_including_own_correction_in_replay_helps_replicated': R('D1'),
        'claim_deployed_advancement_helps_over_residual_injection_replicated': R('D2'),
        'claim_exact_solve_helps_over_one_alpha_pass_replicated': R('D3'),
        'claim_three_part_decomposition_all_replicated': all(R(k) for k in PRIMARY),
        'claim_exact_solve_helps_over_ten_alpha_passes_replicated': R('D4'),
        'reading_D4_all_draws_no_difference': nodiff,
        'claim_P0_better_than_registered_FTBC_R34_T8_replicated': R('D5'),
        'FE_below_S0_by_draw': main['fe_below_s0_by_draw'], 'FE_below_S0_any_draw': any(main['fe_below_s0_by_draw'].values()),
        'reading_D6_ECCV_schedule_overcorrects': any(main['fe_below_s0_by_draw'].values()),
        'claim_P0_better_than_ECCV_schedule_FTBC_R34_T4_replicated': R('D6') and not any(main['fe_below_s0_by_draw'].values())}


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
    assert res['release_update_restored'] and res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
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
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; small, ne = c['small_n'], c['eccv_n']
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()} and res['held_out'] == [int(x['index']) for x in rows['fresh'][:c['diag_n']]]
    assert small == 3 * c['ftbc_batch'] and ne == c['eccv_batch'] * c['eccv_batches'] and c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2 and c['ftbc_alpha'] == 0.4
    rowsets = {**{f'{d}s': v[:small] for d, v in draws.items()}, **{f'{d}e': v[:ne] for d, v in draws.items()}, 'H': rows['fresh'][:c['diag_n']]}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    # Deployment state, biases, solver records and FTBC calls (items 4-6).
    deploy = tload(c['plain_deployment'])['state_dict']
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    for arm in [a for a in arms if a != 'teacher' and not a.startswith('twin')]:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == dep and recs[arm]['strict_reload']
        assert recs[arm]['detail']['state'] == 'plain' and recs[arm]['detail']['deployment_sha256'] == dep, arm
        assert set(f['state_dict']) == set(deploy) and all(torch.equal(f['state_dict'][k], deploy[k]) for k in deploy if 'time_based_bias' not in k), arm
        Tx = f['T']; assert Tx == int(arm.split('-T')[1].split('-')[0]) and len(f['runtime_biases']) == 33
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
    checks = {f'{p}-{d}{"e" if p == "FE-T4" else "s"}': spec for p, spec in FTBC_ARMS.items() for d in CAL}
    assert sorted(ftbc) == sorted(checks)
    for arm, (variant, recipe, passes) in checks.items():
        v = ftbc[arm]; calls = v['calls']; T = int(arm.split('-T')[1].split('-')[0]); key = arm.split('-')[-1]; zh = bias_hash(bias[f'S0-T{T}'], names, T)
        nb, alpha, bs = (3, c['ftbc_alpha'], c['ftbc_batch']) if recipe == 'registered' else (c['eccv_batches'], c['eccv_alpha'], c['eccv_batch'])
        assert v['variant'] == variant and v['recipe'] == recipe and v['passes'] == passes and v['batches_per_pass'] == nb and v['batches'] == nb * passes, arm
        assert v['alpha'] == alpha and v['batch_size'] == bs and v['init_bias_sha256'] == zh and v['images'] == [int(x['index']) for x in rowsets[key]], arm
        assert calls['ann'] == calls['snn'] == 33 * nb * passes and len(calls['updates']) == 33 and all(x == nb * passes for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == fdig and v['ann_level'] == [8] and v['state'] == 'plain', arm
    want = {f'{d}{x}-teacher' for d in CAL for x in ['s', 'e']} | {'H-teacher'}
    assert set(res['targets_sha256']) == want, sorted(set(res['targets_sha256']) ^ want)
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; base = key[:-len('-teacher')]
        assert f['ids'].tolist() == [int(x['index']) for x in rowsets[base]] and f['target'] == 'teacher' and f['state'] is None, key
        assert f['network_digest'] == fdig and f['level'] == [8] and list(f['means']) == names, key
        assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    # Section 4 readout: every calibrated arm on its own rows and on H, S0 on H; declared target keys (item 7).
    diag = json.loads((o / 'diagnostics.json').read_text())
    assert sorted(diag) == sorted([a for a in arms if a[:2] not in ('te', 'tw', 'S0')] + ['S0-T4', 'S0-T8'])
    for arm, v in diag.items():
        if arm.startswith('S0'):
            assert list(v) == ['held_out']
        else:
            assert v['calibration']['target_key'] == f'{arm.split("-")[-1]}-teacher' and v['calibration']['rows'] == len(rowsets[arm.split('-')[-1]]), arm
        assert v['held_out']['target_key'] == 'H-teacher' and v['held_out']['rows'] == c['diag_n'], arm
        T = int(arm.split('-T')[1].split('-')[0])
        for part in v.values():
            assert list(part['layers']) == names and all(len(x['mean_by_step']) == T for x in part['layers'].values()), arm
            assert part['bias_sha256'] == res['bias_sha256'][arm] == recs[arm]['bias_sha256'], arm          # the scored, strict-reloaded biases
        if arm.startswith('P0'):                         # P0 readout on its own rows equals the solver's per-channel residuals (replay_check: same counts)
            for n, info in solver[arm]['layers'].items():
                ref = [float(st['normalized_residual'].abs().mean()) for st in info['steps']]
                assert all(abs(x - y) < 1e-9 for x, y in zip(v['calibration']['layers'][n]['mean_by_step'], ref)), (arm, n)
    M = w / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-F; mechanism ablation, not confirmation or method claim',
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    for k, (x, y) in QUESTIONS.items():
        d = {dd: decide(pr('fresh', x.format(dd), y.format(dd)), ce(x.format(dd)), ce(y.format(dd))) for dd in CAL}
        out['decisions'][k] = {'per_draw': d, 'status': status([v['reading'] for v in d.values()]), 'candidate': x, 'control': y}
    extra = [('F0-T4-{}s', 'S0-T4'), ('FE-T4-{}e', 'S0-T4'), ('F0-T8-{}s', 'S0-T8'), ('FD10-T4-{}s', 'FD-T4-{}s'), ('P0-T4-{}s', 'P0-T4-{}e'),
             ('P0-T4-{}s', 'F0-T4-{}s'), ('FD-T4-{}s', 'F0-T4-{}s'), ('FE-T4-{}e', 'F0-T4-{}s'), ('P0-T8-{}s', 'P0-T4-{}s')]
    for role in c['score_roles']:
        out['pairs'][role] = {f'{x.format(d)}-vs-{y.format(d)}': pr(role, x.format(d), y.format(d)) for d in CAL for x, y in list(QUESTIONS.values()) + extra}
        for x, y in [('twin-T4', 'teacher'), ('twin-T8', 'teacher'), ('S0-T8', 'S0-T4')]:
            out['pairs'][role][f'{x}-vs-{y}'] = pr(role, x, y)
    out['fe_below_s0_by_draw'] = {d: metrics[f'FE-T4-{d}e']['fresh']['correct'] < metrics['S0-T4']['fresh']['correct'] for d in CAL}
    pat = {k: f'{k}-{{}}s' for k in ['F0-T4', 'FA-T4', 'FD-T4', 'FD10-T4', 'P0-T4', 'F0-T8', 'P0-T8']} | {k + 'e': f'{k}-{{}}e' for k in ['FE-T4', 'P0-T4']}
    out['draw_spread'] = {s: (lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)})([metrics[p.format(d)]['fresh']['correct'] for d in CAL]) for s, p in pat.items()}
    out['prefix_consistency_gpu'] = {f'P0-{d}s': all(torch.equal(bias[f'P0-T4-{d}s'][n][t], bias[f'P0-T8-{d}s'][n][t]) for n in names for t in range(4)) for d in CAL}
    for d in CAL:                                      # same images in the same order for the single-pass registered arms
        assert len({ftbc[f'{k}-{d}s']['inputs_sha256'] for k in ['F0-T4', 'FA-T4', 'FD-T4', 'F0-T8']}) == 1, d
    theta = {n: float(tload(o / 'bias-S0-T4.pt')['state_dict'][n + '.thresh']) for n in names}
    out['ftbc'] = {a: {k: v[k] for k in ['variant', 'recipe', 'passes', 'batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta', 'inputs_sha256']} |
                   {'mean_abs_b_over_theta_by_layer': {n: avg([float(b.abs().mean()) for b in bias[a][n]]) / theta[n] for n in names}} for a, v in ftbc.items()}
    out['mechanism'] = {a: {part: {'mean': v[part]['mean'], 'max': v[part]['max']} for part in v} for a, v in diag.items()}
    out['solver_summary'] = {arm: {'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in s['layers'].values()) for t in range(s['T'])],
                                   'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in s['layers'].values()]) for t in range(s['T'])],
                                   'solve_seconds': sum(v['seconds'] for v in s['layers'].values())} for arm, s in solver.items()}
    out.update(resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    out['claims'] = claims(out)
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: {kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk == 'per_draw' else vv) for kk, vv in v.items()}
                           for k, v in out['decisions'].items()}, 'mechanism': out['mechanism'], 'claims': out['claims']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
