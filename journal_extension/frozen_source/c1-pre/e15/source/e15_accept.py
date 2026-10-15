"""E15 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e15_accept.py {sanity,main,release,extra}.
Statistics (exact one-sided McNemar, decide rule) come unchanged from the frozen E14 acceptance module. combine() applies the
cross-job decision rule of section 5 (Q3 claim needs Q1 and Q2 replicated in main and all three release pairs better)."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E15 acceptance uses assert; do not run with -O')
_spec = importlib.util.spec_from_file_location('e15_e14_accept', str(E.parent / 'e14/source/e14_accept.py'))
A14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A14)
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
DRAWS, RSETS = ['D1', 'D2', 'D3'], ['R1', 'R2', 'R3']


def expected_arms(c, release_done):
    arms = []
    if 'main' in c['stages']:
        arms += ['teacher', 'twin-T4', 'S0'] + [f'{s}-{d}' for d in DRAWS for s in ['M0', 'P0']] + [f'{s}-{d}s' for d in DRAWS for s in ['P0', 'F0']]
    if 'release' in c['stages']:
        arms += ([] if 'main' in c['stages'] else ['S0']) + [f'P0-{k}' for k in RSETS] + [f'F0-release-{k}' for k in RSETS if release_done[k]]
    if 'extra' in c['stages']:
        arms += [a for x in c['extra_T'] for a in [f'twin-T{x}', f'S0-T{x}'] + [f'{s}-{d}-T{x}' for d in DRAWS for s in ['M0', 'P0']]]
    return arms


def status(readings):
    if all(r == 'candidate better' for r in readings): return 'replicated'
    if any(r == 'candidate worse' for r in readings): return 'not replicated'
    return 'partially replicated'


def combine(main, release):
    """Section 5 decision across jobs. main/release: the acceptance dicts (None if absent)."""
    q1 = main['decisions']['Q1']['status'] if main else 'absent'; q2 = main['decisions']['Q2']['status'] if main else 'absent'
    q3 = release['decisions']['Q3'] if release else {'pairs_pass': False, 'status': 'absent'}
    return {'Q1': q1, 'Q2': q2, 'Q3_pairs_pass': q3['pairs_pass'],
            'claim_P0_better_than_M0_replicated': q1 == 'replicated',
            'claim_P0_better_than_registered_FTBC_replicated': q1 == 'replicated' and q2 == 'replicated',
            'claim_better_than_release_default_FTBC_allowed': q1 == 'replicated' and q2 == 'replicated' and q3['pairs_pass']}


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
    res = json.loads((o / 'result.json').read_text())
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['twin_unchanged']
    assert res['stages'] == c['stages'] and res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
    release_done = {k: res['release'].get(k, {}).get('status') == 'completed' for k in RSETS}
    if 'release' in c['stages']:
        assert sorted(res['release']) == RSETS and all(v['status'] in ('completed', 'out_of_memory', 'failed') for v in res['release'].values())
        assert all(v['seed'] == c['release_sets'][k][2] for k, v in res['release'].items())
    arms = expected_arms(c, release_done)
    assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    Ts = ([4] if {'main', 'release'} & set(c['stages']) else []) + (c['extra_T'] if 'extra' in c['stages'] else [])
    assert set(res['proxy_digests']) >= {str(x) for x in Ts}
    # Data identity (section 7 item 3).
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
    rows = {'fresh': json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
    old = [r for v in roles.values() for r in v] + json.loads(Path(c['e14_fresh_manifest']).read_text())['development']
    assert not {int(r['index']) for r in rows['fresh']} & {int(r['index']) for r in old} and not {r['sha256'] for r in rows['fresh']} & {r['sha256'] for r in old}
    fit = roles['fit']; draws = {d: fit[lo:hi] for d, (lo, hi) in c['fit_draws'].items()}; rsets = {k: fit[v[0]:v[1]] for k, v in c['release_sets'].items()}
    spans = sorted([tuple(v) for v in c['fit_draws'].values()] + [tuple(v[:2]) for v in c['release_sets'].values()])
    assert sorted(draws) == DRAWS and sorted(rsets) == RSETS and spans[0][0] >= c['exclude_fit_prefix'] and all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()} and res['release_sets'] == {k: [int(x['index']) for x in v] for k, v in rsets.items()}
    small = c['small_n']; rowsets = {**draws, **{f'{d}s': v[:small] for d, v in draws.items()}, **rsets}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    # Biases and solver records (items 5-7).
    deploy_sha = sha(c['plain_deployment']); solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None
    for arm in [a for a in arms if a not in ('teacher',) and not a.startswith('twin')]:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == deploy_sha and recs[arm]['strict_reload']
        Tx = f['T']; assert len(f['runtime_biases']) == 33 and all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        assert bias_hash(f['runtime_biases'], names, Tx) == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v)
        elif arm.startswith(('M0', 'P0')):
            key = arm.split('-')[1]; s = solver[arm]; assert s['full_network_replay_equal'] and s['fit_images'] == len(rowsets[key]), arm
            for n, info in s['layers'].items():
                if arm.startswith('M0'):
                    beta = info['beta'].to(torch.float32); assert within(beta, c['bound_ratio'] * info['theta'])
                    assert all(torch.equal(f['runtime_biases'][n][t], torch.zeros_like(beta) - beta) for t in range(Tx)), (arm, n)
                else:
                    b = info['b']; assert s['per_step'] and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta'])
                    assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    s0_key = 'S0' if 'S0' in bias else None
    checks = {}
    if 'main' in c['stages']:
        checks.update({f'F0-{d}s': ('registered', c['ftbc_batch'], [int(x['index']) for x in draws[d][:small]], None) for d in DRAWS})
    if 'release' in c['stages']:
        checks.update({f'F0-release-{k}': ('release-default', c['release_batch'], None, k) for k in RSETS if release_done[k]})
    assert sorted(ftbc) == sorted(checks)
    if checks:
        zh = bias_hash(bias[s0_key], names, 4)
        for arm, (recipe, bs, images, k) in checks.items():
            v = ftbc[arm]; calls = v['calls']
            assert v['recipe'] == recipe and v['batches'] == 3 and v['alpha'] == c['ftbc_alpha'] and v['batch_size'] == bs and v['init_bias_sha256'] == zh
            assert calls['teacher'] == calls['snn'] == 33 * 3 and len(calls['updates']) == 33 and all(x == 3 for x in calls['updates'].values())
            if images is not None:
                assert v['images'] == images
            else:
                assert sorted(v['images']) == sorted(int(x['index']) for x in rsets[k]) and len(v['images']) == len(rsets[k]) and len(v['inputs_sha256']) == 64
                assert v['seed'] == c['release_sets'][k][2] and set(v['versions']) == {'torch', 'torchvision', 'PIL'}
    want = set()
    if 'main' in c['stages']: want |= set(DRAWS) | {f'{d}s' for d in DRAWS}
    if 'release' in c['stages']: want |= set(RSETS)
    if 'extra' in c['stages']: want |= set(DRAWS)
    assert set(res['teacher_targets_sha256']) == want, sorted(set(res['teacher_targets_sha256']) ^ want)
    for key, tt in res['teacher_targets_sha256'].items():
        f = tload(o / f'teacher-targets-{key}.pt'); assert f['ids'].tolist() == [int(x['index']) for x in rowsets[key]]
        assert hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    M = w / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development replication on fresh-B; not confirmation or method claim', 'stages': c['stages'],
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    if 'main' in c['stages']:
        q1 = {d: decide(pr('fresh', f'P0-{d}', f'M0-{d}'), ce(f'P0-{d}'), ce(f'M0-{d}')) for d in DRAWS}
        q2 = {d: decide(pr('fresh', f'P0-{d}s', f'F0-{d}s'), ce(f'P0-{d}s'), ce(f'F0-{d}s')) for d in DRAWS}
        out['decisions']['Q1'] = {'per_draw': q1, 'status': status([v['reading'] for v in q1.values()])}
        out['decisions']['Q2'] = {'per_draw': q2, 'status': status([v['reading'] for v in q2.values()])}
        arm_of = {'M0': 'M0-{}', 'P0': 'P0-{}', 'P0s': 'P0-{}s', 'F0': 'F0-{}s'}
        spread = {s: [metrics[pattern.format(d)]['fresh']['correct'] for d in DRAWS] for s, pattern in arm_of.items()}
        out['draw_spread'] = {s: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)} for s, v in spread.items()}
        for role in c['score_roles']:
            out['pairs'][role] = {f'{x}-vs-{y}': pr(role, x, y) for x, y in [(f'P0-{d}', f'M0-{d}') for d in DRAWS] + [(f'P0-{d}s', f'F0-{d}s') for d in DRAWS] +
                                  [(f'P0-{d}', f'P0-{d}s') for d in DRAWS] + [(f'P0-{d}s', f'M0-{d}') for d in DRAWS] + [(f'M0-{d}', 'S0') for d in DRAWS]}
    if 'release' in c['stages']:
        done = [k for k in RSETS if release_done[k]]
        q3 = {k: decide(pr('fresh', f'P0-{k}', f'F0-release-{k}'), ce(f'P0-{k}'), ce(f'F0-release-{k}')) for k in done}
        out['decisions']['Q3'] = {'pairs': q3, 'completed_sets': done, 'status': {k: res['release'][k]['status'] for k in RSETS},
                                  'exception_types': {k: res['release'][k].get('exception_type') for k in RSETS if not release_done[k]},
                                  'pairs_pass': len(done) == 3 and all(v['reading'] == 'candidate better' for v in q3.values())}
        for role in c['score_roles']:
            out['pairs'].setdefault(role, {}).update({f'P0-{k}-vs-F0-release-{k}': pr(role, f'P0-{k}', f'F0-release-{k}') for k in done})
            if 'main' in c['stages']:
                out['pairs'][role].update({f'P0-{d}s-vs-F0-release-{k}': pr(role, f'P0-{d}s', f'F0-release-{k}') for d in DRAWS for k in done})
        out['release'] = res['release']
    if 'extra' in c['stages']:
        q4 = {x: {d: decide(pr('fresh', f'P0-{d}-T{x}', f'M0-{d}-T{x}'), ce(f'P0-{d}-T{x}'), ce(f'M0-{d}-T{x}')) for d in DRAWS} for x in c['extra_T']}
        out['decisions']['Q4'] = {str(x): {'per_draw': v, 'worse_count': sum(r['reading'] == 'candidate worse' for r in v.values()),
                                           'replicated_T2_deficit': all(r['reading'] == 'candidate worse' for r in v.values())} for x, v in q4.items()}
        for role in c['score_roles']:
            out['pairs'].setdefault(role, {}).update({f'P0-{d}-T{x}-vs-M0-{d}-T{x}': pr(role, f'P0-{d}-T{x}', f'M0-{d}-T{x}') for x in c['extra_T'] for d in DRAWS})
    out['ftbc'] = {a: {k: v[k] for k in ['recipe', 'batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta', 'inputs_sha256']} | {k: v[k] for k in ['seed', 'versions'] if k in v}
                   for a, v in ftbc.items()}
    summary = {}
    for arm, s in solver.items():
        Ls = s['layers']
        if s.get('per_step'):
            Tx = len(next(iter(Ls.values()))['steps'])
            summary[arm] = {'per_step': True, 'in_range_channels_beyond_one_quantum_by_step': [sum(beyond(v['steps'][t]['normalized_residual'], v['observations_per_channel_per_step'], v['steps'][t]['above_range'], v['steps'][t]['below_range']) for v in Ls.values()) for t in range(Tx)],
                            'at_bound_channels_by_step': [sum(int(v['steps'][t]['at_bound'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'above_range_channels_by_step': [sum(int(v['steps'][t]['above_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'below_range_channels_by_step': [sum(int(v['steps'][t]['below_range'].sum()) for v in Ls.values()) for t in range(Tx)],
                            'mean_abs_b_over_theta_by_step': [avg([float(v['b'][t].abs().mean()) / v['theta'] for v in Ls.values()]) for t in range(Tx)],
                            'max_abs_b_over_theta': max(float(v['b'].abs().max()) / v['theta'] for v in Ls.values()), 'solve_seconds': sum(v['seconds'] for v in Ls.values())}
        else:
            summary[arm] = {'per_step': False, 'in_range_channels_beyond_one_quantum': sum(beyond(v['normalized_residual'], v['observations_per_channel'], v['above_range'], v['below_range']) for v in Ls.values()),
                            'at_bound_channels': sum(int(v['at_bound'].sum()) for v in Ls.values()), 'below_range_channels': sum(int(v['below_range'].sum()) for v in Ls.values()),
                            'above_range_channels': sum(int(v['above_range'].sum()) for v in Ls.values()),
                            'mean_abs_beta_over_theta': avg([float(v['beta'].abs().mean()) / v['theta'] for v in Ls.values()]),
                            'max_abs_beta_over_theta': max(float(v['beta'].abs().max()) / v['theta'] for v in Ls.values()), 'solve_seconds': sum(v['seconds'] for v in Ls.values())}
    out.update(solver_summary=summary, resources={k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated', 'alloc_conf']} |
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    if 'main' in c['stages'] and 'release' in c['stages']:
        out['combined'] = combine(out, out)
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: ({kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk in ('per_draw', 'pairs') else vv) for kk, vv in v.items()}
                               if k != 'Q4' else {x: {'worse_count': vv['worse_count'], 'readings': {d: r['reading'] for d, r in vv['per_draw'].items()}} for x, vv in v.items()})
                           for k, v in out['decisions'].items()}}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
