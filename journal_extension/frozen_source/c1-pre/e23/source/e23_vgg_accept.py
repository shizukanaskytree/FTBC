"""E23 VGG-16 child: CPU acceptance and the cost table (EXPERIMENT_PLAN.md sections 4-6). No CUDA. Usage: e23_vgg_accept.py {sanity,main}.
Checks adapted from the frozen E20/E22 VGG-16 acceptances; the manuscript rule, the expected arms, the per-arm exclusivity reading, the timing
and memory sanity checks and the cost table come from the R34 child's acceptance (e23_r34_accept). The release-default recipe is not run on
VGG-16 (section 3)."""
import hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E23 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e23v_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e23v_e16_accept', E.parent / 'e16/source/e16_accept.py')
R18 = _load('e23v_e18_runner_helpers', E.parent / 'e18/source/e18_runner.py'); RA = _load('e23v_r34_accept', E / 'source/e23_r34_accept.py')
sha, tload, bias_hash, within = A14.sha, A14.tload, A14.bias_hash, A14.within
state_dict_digest = A16.state_dict_digest
CAL, TS, MODEL = ['C1', 'C2', 'C3'], [2, 4, 8], 'vgg'
manuscript_check, expected_cost_arms, exclusivity, cost_table, timing_sane = RA.manuscript_check, RA.expected_cost_arms, RA.exclusivity, RA.cost_table, RA.timing_sane


def check(which):
    c = json.loads((E / 'config' / f'{MODEL}-{which}.json').read_text()); o = Path(c['output']); w = Path(c['workspace'])
    frozen = json.loads((E / 'protocol-freeze.json').read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, p
    ssm = json.loads((E / 'slurm-source-manifest.json').read_text())['files_sha256']; fz = E / 'protocol-freeze.json'
    if fz.is_relative_to(w): assert ssm[str(fz.relative_to(w))] == sha(fz)
    for p, h in frozen.items():
        if Path(p).is_relative_to(w) and Path(p).resolve().is_relative_to(w): assert ssm[str(Path(p).relative_to(w))] == h, p
    res = json.loads((o / 'result.json').read_text()); assert json.loads((o / 'config.json').read_text()) == c, 'run config differs from the frozen config'
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed']
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
    sizes = c['sizes']; small = c['small_n']; assert sizes[0] < sizes[1] == small == 3 * c['ftbc_batch'] < c['draw_n'] and c['ftbc_alpha'] == 0.4
    cost = json.loads((o / 'cost.json').read_text()); want = expected_cost_arms(sizes, False); assert sorted(cost) == sorted(want), sorted(set(cost) ^ set(want))
    assert all(cost[a].get('status', 'completed') == 'completed' for a in want)
    assert sorted(res['arms']) == sorted(want) and sorted(res['cost_arms']) == sorted(want)
    assert res['allocator_conf'] == 'expandable_segments:True', res.get('allocator_conf')     # N6: the reserved peaks depend on the allocator mode
    timing_sane(cost, want, res['hardware']['total_memory'])
    # Source identity: every network the run built has exactly the renamed VGG-16 Q16 checkpoint parameters (as E20/E22).
    assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
    src = R18.renamed_source_state(c['source_checkpoint']); sdig = state_dict_digest(src)
    assert res['teacher_digest'] == sdig and set(res['proxy_digests']) == {f'T{T}' for T in TS} == set(res['snn_digests'])
    assert all(v == sdig for v in list(res['proxy_digests'].values()) + list(res['snn_digests'].values()))
    names = res['interfaces']; L = len(names); assert L == c['interfaces'] and set(res['channels']) == set(names)
    assert names == [k[:-len('.thresh')] for k in src if k.endswith('.thresh')], 'interfaces differ from the checkpoint thresholds'
    # Data identity (section 6 item 3).
    roles = json.loads(Path(c['dataset_manifest']).read_text()); assert sha(c['dataset_manifest']) == c['dataset_manifest_sha256']
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
    fresh = json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']]
    old = [r for v in roles.values() for r in v] + [r for v in json.loads(Path(c['e5_roles']).read_text()).values() for r in v]
    old += [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
    old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
    cal = json.loads(Path(c['calibration_manifest']).read_text()); allc = [r for k in CAL for r in cal[k]]
    for g in [fresh, allc]:
        assert len({int(r['index']) for r in g}) == len(g) == len({r['sha256'] for r in g})
        assert not {int(r['index']) for r in g} & {int(r['index']) for r in old} and not {r['sha256'] for r in g} & {r['sha256'] for r in old}
    assert not {int(r['index']) for r in allc} & {int(r['index']) for r in fresh} and not {r['sha256'] for r in allc} & {r['sha256'] for r in fresh}
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    rowsets = {f'C1-{n}': draws['C1'][:n] for n in sizes}
    # Deployed states, biases, strict reload, solver records and FTBC (as E20/E22).
    recs = json.loads((o / 'metrics.json').read_text()); solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text())
    states = {}
    for T in TS:
        st = tload(o / f'state-T{T}.pt'); assert st['T'] == T and st['source_checkpoint_sha256'] == c['source_checkpoint_sha256'] and st['state_digest'] == sdig
        assert set(st['state_dict']) == set(src) and all(torch.equal(st['state_dict'][k], src[k]) for k in src), T
        states[T] = sha(o / f'state-T{T}.pt'); del st
    bias, hashes = {}, {}
    for arm in want:
        f = tload(o / f'bias-{arm}.pt'); assert recs[arm]['strict_reload'] and f['source_checkpoint_sha256'] == c['source_checkpoint_sha256']
        Tx = f['T']; assert Tx == cost[arm]['T'] == int(arm.split('-T')[1].split('-')[0]) and list(f['runtime_biases']) == names and 'state_dict' not in f
        assert f['state_file'] == f'state-T{Tx}.pt' and f['state_sha256'] == states[Tx] and f['state_digest'] == sdig, arm
        assert all(len(f['runtime_biases'][n]) == Tx and all(b.shape == (res['channels'][n],) and torch.isfinite(b).all() for b in f['runtime_biases'][n]) for n in names)
        bias[arm] = f['runtime_biases']; hashes[arm] = bias_hash(f['runtime_biases'], names, Tx)
        assert hashes[arm] == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
    tfiles = {f'C1-{n}-teacher': tload(o / f'targets-C1-{n}-teacher.pt') for n in sizes}
    for arm in [a for a in want if a.startswith('P0')]:
        n = int(arm.split('-')[1]); key = f'C1-{n}'; s = solver[arm]; Tx = cost[arm]['T']
        assert s['full_network_replay_equal'] and s['fit_images'] == n == cost[arm]['fit_images'] and s['T'] == Tx and s['objective'] == 'P0', arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == 'P0', arm
        assert tfiles[tk]['ids'].tolist() == [int(x['index']) for x in rowsets[key]] and tfiles[tk]['target'] == 'teacher' and tfiles[tk]['network_digest'] == sdig
        assert list(s['layers']) == list(tfiles[tk]['means']) == names and cost[arm]['captures'] == L
        assert abs(cost[arm]['layer_seconds_sum'] - sum(v['seconds'] for v in s['layers'].values())) < 1e-6 and cost[arm]['layer_seconds_sum'] <= cost[arm]['seconds']
        for nm, info in s['layers'].items():
            b = info['b']; assert torch.equal(info['target_mean'], tfiles[tk]['means'][nm]) and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta']), (arm, nm)
            assert all(torch.equal(bias[arm][nm][t], b[t]) for t in range(Tx))
    assert sorted(solver) == sorted(a for a in want if a.startswith('P0')) and sorted(ftbc) == sorted(a for a in want if a.startswith('F0'))
    for arm in [a for a in want if a.startswith('F0')]:
        v = ftbc[arm]; calls = v['calls']; Tx = cost[arm]['T']
        zh = bias_hash({nm: [torch.zeros_like(b) for b in bias[arm][nm]] for nm in names}, names, Tx)
        passes = int(arm[len('F0reg')]); assert passes in (1, 3), arm; nb = 3 * passes
        assert v['recipe'] == ('registered' if passes == 1 else 'registered-3pass') and v['batches'] == nb and v['alpha'] == c['ftbc_alpha'], arm
        assert v['batch_size'] == c['ftbc_batch'] and v['init_bias_sha256'] == zh and v['T'] == Tx, arm
        assert calls['ann'] == calls['snn'] == L * nb and sorted(calls['updates']) == sorted(names) and all(x == nb for x in calls['updates'].values()), arm
        assert v['ann'] == 'teacher' and v['ann_digest'] == sdig and v['ann_level'] == [c['source_level']] and v['images'] == [int(x['index']) for x in draws['C1'][:small]], arm
        assert cost[arm]['calls'] == calls, arm
    bases = sorted({a.rsplit('-r', 1)[0] for a in want})
    determinism = {b: hashes[f'{b}-r1'] == hashes[f'{b}-r2'] for b in bases}
    manuscript = manuscript_check(w / 'manuscript', json.loads((E / 'preservation-before.json').read_text()))
    out = {'passed': True, 'model': MODEL, 'scope': 'sanity only' if c['sanity'] else 'fitting-cost measurement; no accuracy reported',
           'cost_table': cost_table(cost, want, c['solver_batch']), 'repeat_bitwise_equal': determinism, 'not_completed': {},
           'arms_with_allocator_ooms': {a: cost[a]['num_ooms_increase'] for a in want if cost[a]['num_ooms_increase']},   # follow-up 1 B5: completed after a caught failure
           'exclusivity': exclusivity(res['hardware']['slurm_job_id'], res['hardware']['pid'], cost, E / 'slurm'),
           'resources': {k: res[k] for k in ['seconds_total', 'peak_reserved_since_last_reset', 'peak_allocated_since_last_reset', 'allocator_conf']} |
                        {'max_arm_peak_reserved_bytes': max(cost[a]['peak_reserved_bytes'] for a in want), 'max_arm_peak_allocated_bytes': max(cost[a]['peak_allocated_bytes'] for a in want),
                         'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
           'manuscript': manuscript, 'not_method_claim': True, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not out['cuda_initialized']
    (E / f'{MODEL}-{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'passed': True, 'model': MODEL, 'repeat_bitwise_equal': determinism,
                      'arms_possibly_affected': out['exclusivity']['arms_possibly_affected'], 'arms_unknown': out['exclusivity']['arms_unknown'],
                      'wall_seconds_r2': {b: v.get('r2', {}).get('wall_seconds') for b, v in out['cost_table'].items()}}, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
