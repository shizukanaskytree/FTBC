"""E20 VGG-16 child: CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA. Usage: e20_vgg_accept.py {sanity,main}.
Adapted from the frozen E18 acceptance (same checks for the shared machinery); statistics (exact one-sided McNemar, decide rule) come unchanged
from the frozen E14 acceptance module; status and state_dict_digest from E16. Questions and claims are the R34 child's (e20_r34_accept)."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E20 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e20v_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e20v_e16_accept', E.parent / 'e16/source/e16_accept.py')
R18 = _load('e20v_e18_runner_helpers', E.parent / 'e18/source/e18_runner.py'); RA = _load('e20v_r34_accept', E / 'source/e20_r34_accept.py')
sha, tload, paired, decide, bias_hash, within, beyond = A14.sha, A14.tload, A14.paired, A14.decide, A14.bias_hash, A14.within, A14.beyond
status, state_dict_digest = A16.status, A16.state_dict_digest
CAL, TS, MODEL = ['C1', 'C2', 'C3'], [4, 8], 'vgg'
questions, expected_arms, claims, reading_e20, READINGS = RA.questions, RA.expected_arms, RA.claims, RA.reading_e20, RA.READINGS


def check(which):
    avg = lambda vals: sum(vals) / len(vals)
    c = json.loads((E / 'config' / f'{MODEL}-{which}.json').read_text()); o = Path(c['output']); w = Path(c['workspace']); sizes = c['sizes']; QUESTIONS = questions(sizes)
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, p
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    res = json.loads((o / 'result.json').read_text()); assert json.loads((o / 'config.json').read_text()) == c, 'run config differs from the frozen config'
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed'] and res['twin_unchanged']
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower() and res['sizes'] == sizes
    arms = expected_arms(sizes); assert sorted(res['arms']) == sorted(arms), sorted(set(res['arms']) ^ set(arms))
    # Source identity: every network the run built has exactly the renamed VGG-16 Q16 checkpoint parameters.
    assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
    src = R18.renamed_source_state(c['source_checkpoint']); sdig = state_dict_digest(src)
    assert res['teacher_digest'] == sdig and set(res['proxy_digests']) == {f'T{T}' for T in TS} == set(res['snn_digests'])
    assert all(v == sdig for v in list(res['proxy_digests'].values()) + list(res['snn_digests'].values()))
    names = res['interfaces']; L = len(names); assert L == c['interfaces'] and set(res['channels']) == set(names)
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
    assert not {int(r['index']) for r in allc} & {int(r['index']) for r in rows['fresh']} and not {r['sha256'] for r in allc} & {r['sha256'] for r in rows['fresh']}
    assert len({int(r['index']) for r in allc}) == len(allc) == len({r['sha256'] for r in allc})
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; small = c['small_n']
    assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    assert small == 3 * c['ftbc_batch'] == sizes[-1] < c['draw_n'] and sizes == sorted(set(sizes)) and len(sizes) == 4 and c['ftbc_alpha'] == 0.4
    rowsets = {f'{d}-{n}': v[:n] for d, v in draws.items() for n in sizes}
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
        if arm.startswith('F0'):
            continue
        key = '-'.join(arm.split('-')[2:]); s = solver[arm]
        assert s['full_network_replay_equal'] and s['fit_images'] == len(rowsets[key]) and s['T'] == Tx and s['objective'] == 'P0', arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == 'P0', arm
        assert list(s['layers']) == list(tfiles[tk]['means']) == names
        for n, info in s['layers'].items():
            th = info['theta']; assert torch.equal(info['target_mean'], tfiles[tk]['means'][n]), (arm, n)
            b = info['b']; assert b.shape[0] == Tx and within(b, c['bound_ratio'] * th)
            assert all(torch.equal(f['runtime_biases'][n][t], b[t]) for t in range(Tx)) and all(torch.equal(st['b'], b[i]) for i, st in enumerate(info['steps']))
    assert sorted(solver) == sorted(a for a in arms if a.startswith('P0'))
    checks = {f'F0-T{T}-{d}': ([int(x['index']) for x in rowsets[f'{d}-{small}']], T) for d in CAL for T in TS}
    assert sorted(ftbc) == sorted(checks)
    for arm, (images, T) in checks.items():
        v = ftbc[arm]; calls = v['calls']; zh = bias_hash(bias[f'S0-T{T}'], names, T)
        assert v['recipe'] == 'registered' and v['batches'] == 3 and v['alpha'] == c['ftbc_alpha'] and v['batch_size'] == c['ftbc_batch'] and v['init_bias_sha256'] == zh and v['T'] == T, arm
        assert calls['ann'] == calls['snn'] == L * 3 and sorted(calls['updates']) == sorted(names) and all(x == 3 for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == sdig and v['ann_level'] == [c['source_level']] and v['images'] == images and len(images) == small, arm
    for d in CAL:                                      # same images in the same order at both T
        assert ftbc[f'F0-T4-{d}']['inputs_sha256'] == ftbc[f'F0-T8-{d}']['inputs_sha256'], d
    want = {f'{d}-{n}-teacher' for d in CAL for n in sizes}; assert set(res['targets_sha256']) == want, sorted(set(res['targets_sha256']) ^ want)
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
    out = {'passed': True, 'model': MODEL, 'sizes': sizes, 'scope': 'sanity only' if c['sanity'] else 'development screen on fresh-G; calibration budget, not confirmation or method claim',
           'metrics': metrics, 'decisions': {}, 'pairs': {}}
    for k, (x, y) in QUESTIONS.items():
        d = {dd: decide(pr('fresh', x.format(dd), y.format(dd)), ce(x.format(dd)), ce(y.format(dd))) for dd in CAL}
        rd = [v['reading'] for v in d.values()]
        out['decisions'][k] = {'per_draw': d, 'status': status(rd), 'counts': {r: rd.count(r) for r in READINGS}, 'reading_e20': reading_e20(rd), 'candidate': x, 'control': y}
    extra = [(f'P0-T{T}-{{}}-{sizes[-1]}', f'F0-T{T}-{{}}') for T in TS] + [(f'F0-T{T}-{{}}', f'S0-T{T}') for T in TS]
    extra += [(f'P0-T{T}-{{}}-{n}', f'P0-T{T}-{{}}-{sizes[-1]}') for T in TS for n in sizes[:2]] + [(f'P0-T8-{{}}-{n}', f'P0-T4-{{}}-{n}') for n in sizes]
    for role in c['score_roles']:
        out['pairs'][role] = {f'{x.format(d)}-vs-{y.format(d)}': pr(role, x.format(d), y.format(d)) for d in CAL for x, y in list(QUESTIONS.values()) + extra}
        for x, y in [('twin-T4', 'teacher'), ('twin-T8', 'teacher'), ('S0-T8', 'S0-T4')]:
            out['pairs'][role][f'{x}-vs-{y}'] = pr(role, x, y)
    pat = {f'P0-T{T}-{n}': f'P0-T{T}-{{}}-{n}' for T in TS for n in sizes} | {f'F0-T{T}': f'F0-T{T}-{{}}' for T in TS}
    out['draw_spread'] = {s: (lambda v: {'fresh_correct': v, 'min': min(v), 'max': max(v), 'mean': avg(v)})([metrics[p.format(d)]['fresh']['correct'] for d in CAL]) for s, p in pat.items()}
    out['prefix_consistency_gpu'] = {f'P0-{d}-{n}': all(torch.equal(bias[f'P0-T4-{d}-{n}'][m][t], bias[f'P0-T8-{d}-{n}'][m][t]) for m in names for t in range(4))
                                     for d in CAL for n in sizes}
    theta = {n: float(src[n + '.thresh']) for n in names}
    out['ftbc'] = {a: {k: v[k] for k in ['recipe', 'ann', 'ann_level', 'batches', 'alpha', 'batch_size', 'T', 'max_abs_bias_over_theta', 'inputs_sha256']} |
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
               manuscript_git_unchanged=True, not_method_claim=True, cuda_initialized=torch.cuda.is_initialized())
    out['claims'] = claims(out, sizes)
    assert not out['cuda_initialized']
    (E / f'{MODEL}-{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'model': MODEL, 'metrics': {a: {r: (m[r]['correct'], round(m[r]['ce'], 4)) for r in m} for a, m in metrics.items()},
             'decisions': {k: {kk: ({d: {x: y for x, y in vv2.items() if x != 'pair'} for d, vv2 in vv.items()} if kk == 'per_draw' else vv) for kk, vv in v.items()}
                           for k, v in out['decisions'].items()}, 'claims': out['claims']}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
