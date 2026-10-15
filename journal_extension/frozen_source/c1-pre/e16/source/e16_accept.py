"""E16 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e16_accept.py {sanity,main,release}.
Statistics (exact one-sided McNemar, decide rule) come unchanged from the frozen E14 acceptance module; combine() applies the
cross-job decision rule of section 5."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E16 acceptance uses assert; do not run with -O')
_spec = importlib.util.spec_from_file_location('e16_e14_accept', str(E.parent / 'e14/source/e14_accept.py'))
A14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A14)
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
CAL, REL, ANN = ['C1', 'C2', 'C3'], ['Q1', 'Q2', 'Q3'], ['tw', 'te']
ANN_TAG = {'tw': 'twin-spatial', 'te': 'teacher'}           # the released FTBC update's ANN argument per arm tag


def state_dict_digest(sd):
    """run_probe.state_digest (run_probe.py:59-69) applied to a saved state_dict: sorted keys; name, dtype, shape, bytes."""
    h = hashlib.sha256()
    for name, value in sorted(sd.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def expected_arms(c, release_done):
    arms = []
    if 'main' in c['stages']:
        arms += ['teacher', 'twin-pl', 'twin-sp', 'a8-sp', 'S0-pl', 'S0-sp']
        arms += [f'{a}-{d}' for d in CAL for a in ['P0-pl-te', 'P0-pl-tw', 'P0-sp-te', 'P0-sp-a8', 'P0-sp-tw', 'M0-sp-tw']]
        arms += [f'{a}-{d}s' for d in CAL for a in ['P0-sp-tw', 'F0-sp-tw', 'F0-sp-te']]
    if 'release' in c['stages']:
        arms += ([] if 'main' in c['stages'] else ['S0-sp']) + [f'P0-sp-tw-{k}' for k in REL]
        arms += [f'F0-sp-rel-{a}-{k}' for k in REL for a in ANN if release_done[f'{k}-{a}']]
    return arms


def status(readings):
    if all(r == 'candidate better' for r in readings): return 'replicated'
    if any(r == 'candidate worse' for r in readings): return 'not replicated'
    return 'partially replicated'


def combine(main, release):
    """Section 5 decision across jobs. main/release: the acceptance dicts (None if absent). Same-target claims (FTBC with the spatial
    twin as its ANN) need Q2 / Q3; the wording without the same-target qualifier also needs the source-teacher references Q2b / Q3b."""
    st = lambda q: main['decisions'][q]['status'] if main else 'absent'
    R = lambda q: st(q) == 'replicated'
    q3 = {q: (release['decisions'][q]['pairs_pass'] if release else False) for q in ['Q3', 'Q3b']}
    reg_same = R('Q1') and R('Q2'); reg_both = reg_same and R('Q2b')
    return {'Q1': st('Q1'), 'Q2': st('Q2'), 'Q2b': st('Q2b'), 'Q4': st('Q4'), 'Q3_pairs_pass': q3['Q3'], 'Q3b_pairs_pass': q3['Q3b'],
            'claim_two_stage_better_than_P0_alone_replicated': R('Q1'),
            'claim_two_stage_better_than_BN_then_registered_FTBC_same_target_replicated': reg_same,
            'claim_two_stage_better_than_BN_then_registered_FTBC_both_references_replicated': reg_both,
            'claim_two_stage_better_than_BN_then_release_default_FTBC_same_target_allowed': reg_same and q3['Q3'],
            'claim_two_stage_better_than_BN_then_release_default_FTBC_both_references_allowed': reg_both and q3['Q3'] and q3['Q3b'],
            # pipeline-level statements against the source-teacher reference only; the qualifier "Q8 source teacher as ANN" is mandatory
            'claim_pipeline_better_than_BN_then_registered_FTBC_teacher_reference_replicated': R('Q1') and R('Q2b'),
            'claim_pipeline_better_than_BN_then_release_default_FTBC_teacher_reference_allowed': R('Q1') and R('Q2b') and q3['Q3b'],
            'claim_twin_target_better_than_teacher_target_on_BN_state_replicated': R('Q4'),
            'Q1_wording_must_disclose_label_free_target_gain': (main['label_free_twin_target_better_draws'] > 0) if main else None}


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
    rkeys = [f'{k}-{a}' for k in REL for a in ANN]
    release_done = {rk: res['release'].get(rk, {}).get('status') == 'completed' for rk in rkeys}
    if 'release' in c['stages']:
        assert sorted(res['release']) == sorted(rkeys) and all(v['status'] in ('completed', 'out_of_memory', 'failed') for v in res['release'].values())
        assert all(v['seed'] == c['release_seeds'][rk.split('-')[0]] and v['ann'] == ANN_TAG[rk.split('-')[1]] for rk, v in res['release'].items())
    else:
        assert res['release'] == {}
    arms = expected_arms(c, release_done)
    assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    # Frozen network files and the networks the run actually built (section 7 item 4).
    art = json.loads(Path(c['e0_artifacts']).read_text())
    for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt'), ('spatial_deployment', 'deployment-spatial.pt'), ('spatial_proxy', 'proxy-spatial.pt')]:
        assert Path(c[key]).name == name and Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
    fdig = {'plain': state_dict_digest(tload(c['plain_proxy'])['state_dict']), 'spatial': state_dict_digest(tload(c['spatial_proxy'])['state_dict'])}
    assert fdig['plain'] != fdig['spatial'] and res['teacher_digest'] == fdig['plain']      # the Q8 teacher has exactly the plain proxy parameters: A8(phi0)
    want_states = ({'plain-T4', 'spatial-T4'} if 'main' in c['stages'] else set()) | ({'spatial-T4'} if 'release' in c['stages'] else set())
    assert set(res['proxy_digests']) == want_states and all(res['proxy_digests'][k] == fdig[k.split('-')[0]] for k in want_states)
    assert set(res['grid_proxy_digests']) == ({'spatial-L8'} if 'main' in c['stages'] else set()) and all(v == fdig['spatial'] for v in res['grid_proxy_digests'].values())
    dep = {s: sha(c[f'{s}_deployment']) for s in ['plain', 'spatial']}; assert res['deployment_sha256'] == dep
    # Data identity (section 7 item 3).
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
    rows = {'fresh': json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']], 'development': roles['development'][:c['dev_count']]}
    e5 = json.loads(Path(c['e5_roles']).read_text())
    old = [r for v in roles.values() for r in v] + [r for v in e5.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
    cal = json.loads(Path(c['calibration_manifest']).read_text()); allc = [r for k in CAL + REL for r in cal[k]]
    for group in [rows['fresh'], allc]:
        assert not {int(r['index']) for r in group} & {int(r['index']) for r in old} and not {r['sha256'] for r in group} & {r['sha256'] for r in old}
    assert not {int(r['index']) for r in allc} & {int(r['index']) for r in rows['fresh']} and len({int(r['index']) for r in allc}) == len(allc)
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; rsets = {k: cal[k][:c['release_n'] * c['release_batch']] for k in REL}; small = c['small_n']
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()} and res['release_sets'] == {k: [int(x['index']) for x in v] for k, v in rsets.items()}
    rowsets = {**draws, **{f'{d}s': v[:small] for d, v in draws.items()}, **rsets}
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text()); metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in c['score_roles']:
            q = pred[arm][role]; ids = torch.tensor([int(r['index']) for r in rows[role]]); lab = torch.tensor([int(r['label']) for r in rows[role]])
            assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], lab) and torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == lab).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), lab))
            r = recs[arm]['metrics'][role]; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
            metrics[arm][role] = {'correct': correct, 'n': len(lab), 'top1': 100 * correct / len(lab), 'ce': ce}
    # Deployment states, biases and solver records (items 4-7).
    deploy = {'plain': tload(c['plain_deployment'])['state_dict'], 'spatial': tload(c['spatial_deployment'])['state_dict']}
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None
    tfiles = {key: tload(o / f'targets-{key}.pt') for key in res['targets_sha256']}
    for arm in [a for a in arms if a not in ('teacher', 'a8-sp') and not a.startswith('twin')]:
        # plain_deployment_sha256 is the inherited E14 field and names the plain file for every arm; the declared state's file is deployment_sha256
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == dep['plain'] and recs[arm]['strict_reload']
        state = recs[arm]['detail']['state']; assert state == ('plain' if '-pl' in arm else 'spatial'), (arm, state)
        assert recs[arm]['detail']['deployment_sha256'] == dep[state], arm
        assert set(f['state_dict']) == set(deploy[state]) and all(torch.equal(f['state_dict'][k], deploy[state][k]) for k in deploy[state] if 'time_based_bias' not in k), arm
        Tx = f['T']; assert Tx == 4 and len(f['runtime_biases']) == 33 and all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        assert bias_hash(f['runtime_biases'], names, Tx) == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
        if arm.startswith('S0'):
            assert all(not torch.count_nonzero(b) for v in f['runtime_biases'].values() for b in v)
        elif arm.startswith(('M0', 'P0')):
            key = arm.split('-')[-1]; s = solver[arm]; target = arm.split('-')[2]
            assert s['full_network_replay_equal'] and s['fit_images'] == len(rowsets[key]) and s['state'] == state, arm
            tk = f'{key}-teacher' if target == 'te' else f"{key}-{ {'tw': 'twin', 'a8': 'a8'}[target] }-{state}"
            assert s['target_key'] == tk == recs[arm]['detail']['target_key'], (arm, s['target_key'])
            assert list(s['layers']) == list(tfiles[tk]['means']) == names
            for n, info in s['layers'].items():
                assert torch.equal(info['target_mean'], tfiles[tk]['means'][n]), (arm, n)      # the solver used the declared target file
                if arm.startswith('M0'):
                    beta = info['beta'].to(torch.float32); assert within(beta, c['bound_ratio'] * info['theta'])
                    assert all(torch.equal(f['runtime_biases'][n][t], torch.zeros_like(beta) - beta) for t in range(Tx)), (arm, n)
                else:
                    b = info['b']; assert s['per_step'] and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta'])
                    assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    checks = {}
    if 'main' in c['stages']:
        checks.update({f'F0-sp-{a}-{d}s': ('registered', c['ftbc_batch'], [int(x['index']) for x in draws[d][:small]], None, a) for d in CAL for a in ANN})
    if 'release' in c['stages']:
        checks.update({f'F0-sp-rel-{a}-{k}': ('release-default', c['release_batch'], None, k, a) for k in REL for a in ANN if release_done[f'{k}-{a}']})
    assert sorted(ftbc) == sorted(checks)
    if checks:
        zh = bias_hash(bias['S0-sp'], names, 4)
        for arm, (recipe, bs, images, k, a) in checks.items():
            v = ftbc[arm]; calls = v['calls']
            assert v['recipe'] == recipe and v['batches'] == 3 and v['alpha'] == c['ftbc_alpha'] and v['batch_size'] == bs and v['init_bias_sha256'] == zh
            assert calls['ann'] == calls['snn'] == 33 * 3 and len(calls['updates']) == 33 and all(x == 3 for x in calls['updates'].values())
            assert v['ann'] == ANN_TAG[a] and v['ann_digest'] == fdig['spatial' if a == 'tw' else 'plain'] and v['state'] == 'spatial', arm
            assert v['ann_level'] == ([4] if a == 'tw' else [8]), arm       # the digest does not contain L: the twin is L=4, the teacher L=8
            if images is not None:
                assert v['images'] == images
            else:
                assert sorted(v['images']) == sorted(int(x['index']) for x in rsets[k]) and len(v['images']) == len(rsets[k]) and len(v['inputs_sha256']) == 64
                assert v['seed'] == c['release_seeds'][k] and set(v['versions']) == {'torch', 'torchvision', 'PIL'}
        # the two ANN references of one draw or realization saw identical inputs in the same order
        same = ([(f'F0-sp-tw-{d}s', f'F0-sp-te-{d}s') for d in CAL] if 'main' in c['stages'] else []) + [(f'F0-sp-rel-tw-{k}', f'F0-sp-rel-te-{k}') for k in REL
                                                                                                          if 'release' in c['stages'] and release_done[f'{k}-tw'] and release_done[f'{k}-te']]
        for x, y in same:
            assert ftbc[x]['inputs_sha256'] == ftbc[y]['inputs_sha256'] and ftbc[x]['images'] == ftbc[y]['images'], (x, y)
    want = set()
    if 'main' in c['stages']:
        want |= {f'{d}-teacher' for d in CAL} | {f'{d}-twin-{s}' for d in CAL for s in ['plain', 'spatial']} | {f'{d}-a8-spatial' for d in CAL} | {f'{d}s-twin-spatial' for d in CAL}
    if 'release' in c['stages']: want |= {f'{k}-twin-spatial' for k in REL}
    assert set(res['targets_sha256']) == want, sorted(set(res['targets_sha256']) ^ want)
    for key, tt in res['targets_sha256'].items():
        f = tfiles[key]; parts = key.split('-'); base, kind = parts[0], parts[1]; st = None if kind == 'teacher' else parts[2]
        assert f['ids'].tolist() == [int(x['index']) for x in rowsets[base]] and f['target'] == kind and f['state'] == st, key
        assert f['network_digest'] == fdig['plain' if kind == 'teacher' else st] and f['level'] == [4 if kind == 'twin' else 8], key
        assert list(f['means']) == names and hashlib.sha256(b''.join(v.numpy().tobytes() for v in f['means'].values())).hexdigest() == tt
    if 'main' in c['stages']:                       # the four targets of a draw are distinct (state and grid really changed)
        for d in CAL:
            m = {k: tfiles[f'{d}-{k}']['means'] for k in ['teacher', 'twin-plain', 'twin-spatial', 'a8-spatial']}
            for x, y in [('twin-plain', 'twin-spatial'), ('teacher', 'a8-spatial'), ('a8-spatial', 'twin-spatial'), ('teacher', 'twin-plain')]:
                assert any(not torch.equal(m[x][n], m[y][n]) for n in names), (d, x, y)
    M = w / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Pre-registered analysis (section 5).
    P = {role: {a: pred[a][role]['predictions'] for a in arms} for role in c['score_roles']}
    lab = {role: torch.tensor([int(r['label']) for r in rows[role]]) for role in c['score_roles']}
    pr = lambda role, x, y: paired(lab[role], P[role][x], P[role][y])
    ce = lambda a: metrics[a]['fresh']['ce']
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-C; not confirmation or method claim', 'stages': c['stages'],
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    def q(x, y, keys):
        d = {k: decide(pr('fresh', x.format(k), y.format(k)), ce(x.format(k)), ce(y.format(k))) for k in keys}
        return {'per_draw': d, 'status': status([v['reading'] for v in d.values()])}
    if 'main' in c['stages']:
        out['decisions'].update(Q1=q('P0-sp-tw-{}', 'P0-pl-te-{}', CAL), Q2=q('P0-sp-tw-{}s', 'F0-sp-tw-{}s', CAL), Q2b=q('P0-sp-tw-{}s', 'F0-sp-te-{}s', CAL),
                                Q4=q('P0-sp-tw-{}', 'P0-sp-te-{}', CAL))
        lf = q('P0-pl-tw-{}', 'P0-pl-te-{}', CAL)    # wording rule of section 5 (descriptive; not a decision)
        out['label_free_twin_target'] = lf; out['label_free_twin_target_better_draws'] = sum(v['reading'] == 'candidate better' for v in lf['per_draw'].values())
        for role in c['score_roles']:
            out['pairs'][role] = {f'{x.format(d)}-vs-{y.format(d)}': pr(role, x.format(d), y.format(d)) for d in CAL for x, y in
                                  [('P0-sp-tw-{}', 'P0-pl-te-{}'), ('P0-sp-tw-{}s', 'F0-sp-tw-{}s'), ('P0-sp-tw-{}s', 'F0-sp-te-{}s'), ('P0-sp-tw-{}', 'P0-sp-te-{}'),
                                   ('P0-pl-tw-{}', 'P0-pl-te-{}'), ('P0-sp-tw-{}', 'P0-pl-tw-{}'), ('P0-sp-tw-{}', 'S0-sp'), ('P0-sp-a8-{}', 'P0-sp-te-{}'),
                                   ('P0-sp-tw-{}', 'P0-sp-a8-{}'), ('P0-sp-tw-{}', 'M0-sp-tw-{}'), ('P0-sp-te-{}', 'P0-pl-te-{}'), ('P0-sp-tw-{}', 'P0-sp-tw-{}s'),
                                   ('F0-sp-tw-{}s', 'F0-sp-te-{}s'), ('F0-sp-tw-{}s', 'S0-sp'), ('F0-sp-te-{}s', 'S0-sp')]}
            for x, y in [('S0-sp', 'S0-pl'), ('twin-sp', 'twin-pl'), ('a8-sp', 'teacher'), ('twin-sp', 'a8-sp')]:
                out['pairs'][role][f'{x}-vs-{y}'] = pr(role, x, y)
        pat = {'P0-pl-te': 'P0-pl-te-{}', 'P0-pl-tw': 'P0-pl-tw-{}', 'P0-sp-te': 'P0-sp-te-{}', 'P0-sp-a8': 'P0-sp-a8-{}', 'P0-sp-tw': 'P0-sp-tw-{}', 'M0-sp-tw': 'M0-sp-tw-{}',
               'P0-sp-tw-96': 'P0-sp-tw-{}s', 'F0-sp-tw-96': 'F0-sp-tw-{}s', 'F0-sp-te-96': 'F0-sp-te-{}s'}
        out['draw_spread'] = {s: (lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)})([metrics[p.format(d)]['fresh']['correct'] for d in CAL]) for s, p in pat.items()}
    if 'release' in c['stages']:
        for name, a in [('Q3', 'tw'), ('Q3b', 'te')]:
            done = [k for k in REL if release_done[f'{k}-{a}']]
            q3 = {k: decide(pr('fresh', f'P0-sp-tw-{k}', f'F0-sp-rel-{a}-{k}'), ce(f'P0-sp-tw-{k}'), ce(f'F0-sp-rel-{a}-{k}')) for k in done}
            out['decisions'][name] = {'pairs': q3, 'ann': ANN_TAG[a], 'completed_sets': done, 'status': {k: res['release'][f'{k}-{a}']['status'] for k in REL},
                                      'exception_types': {k: res['release'][f'{k}-{a}'].get('exception_type') for k in REL if not release_done[f'{k}-{a}']},
                                      'pairs_pass': len(done) == 3 and all(v['reading'] == 'candidate better' for v in q3.values())}
            for role in c['score_roles']:
                out['pairs'].setdefault(role, {}).update({f'P0-sp-tw-{k}-vs-F0-sp-rel-{a}-{k}': pr(role, f'P0-sp-tw-{k}', f'F0-sp-rel-{a}-{k}') for k in done})
        for role in c['score_roles']:
            out['pairs'][role].update({f'F0-sp-rel-tw-{k}-vs-F0-sp-rel-te-{k}': pr(role, f'F0-sp-rel-tw-{k}', f'F0-sp-rel-te-{k}') for k in REL
                                       if release_done[f'{k}-tw'] and release_done[f'{k}-te']})
        out['release'] = res['release']
    out['ftbc'] = {a: {k: v[k] for k in ['recipe', 'ann', 'ann_level', 'batches', 'alpha', 'batch_size', 'max_abs_bias_over_theta', 'inputs_sha256']} | {k: v[k] for k in ['seed', 'versions'] if k in v}
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
               {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    if 'main' in c['stages'] and 'release' in c['stages']:
        out['combined'] = combine(out, out)
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: {kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk in ('per_draw', 'pairs') else vv) for kk, vv in v.items()}
                           for k, v in out['decisions'].items()}}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
