"""E23 ResNet-34 child: CPU acceptance and the cost table (EXPERIMENT_PLAN.md sections 4-6). No CUDA. Usage: e23_r34_accept.py {sanity,main}.
Checks adapted from the frozen E20/E22 R34 acceptances (freeze, data, state, biases, strict reload, solver records, FTBC calls, manuscript);
statistics are not computed (no accuracy is reported). New here: the release rows, the repeat-determinism record, the per-arm exclusivity
reading (initial review B4), the out-of-memory rule for the release-default arms (B3), the timing and memory sanity checks (N6, N10) and the
cost table (B1, B2, N1)."""
import hashlib, importlib.util, json, statistics, subprocess, sys, time
from datetime import datetime
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
if not __debug__:
    raise SystemExit('E23 acceptance uses assert; do not run with -O')
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A14 = _load('e23r_e14_accept', E.parent / 'e14/source/e14_accept.py'); A16 = _load('e23r_e16_accept', E.parent / 'e16/source/e16_accept.py')
sha, tload, bias_hash, within = A14.sha, A14.tload, A14.bias_hash, A14.within
state_dict_digest = A16.state_dict_digest
CAL, TS, REPEATS, MODEL, L = ['C1', 'C2', 'C3'], [2, 4, 8], [1, 2], 'r34', 33
RELEASE_TS = [2, 4, 8]                                 # equal to the runner's (CPU check); T=8 is attempted under the out-of-memory rule
GIB = 2 ** 30
GAP = 3.0                                              # section 6 item 6: the largest gap (s) between the runner's own process rows still counted as covered
LIB_OOM = ('unable to find an engine', 'CUDNN_STATUS_ALLOC_FAILED', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_INTERNAL_ERROR')   # equal to the runner's (CPU check)


def git_out(M, *a):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(M), *a], text=True)    # never refreshes or rewrites the index


def manuscript_check(M, before):
    """Section 6 item 7: compares HEAD, the index entries and the staged changes with the snapshot and records the status lines and the index
    file. It only records (the E24 follow-up 1 R1, adopted for E23 before its review); verify_preservation.py asserts."""
    cur = {'HEAD': git_out(M, 'rev-parse', 'HEAD'), 'index_entries_sha256': hashlib.sha256(git_out(M, 'ls-files', '--stage').encode()).hexdigest(),
           'staged': git_out(M, 'diff', '--cached', '--name-status'), 'status': git_out(M, 'status', '--porcelain=v1', '--untracked-files=all'),
           'index_sha256': sha(M / '.git/index')}
    same = all(cur[k] == before[k] for k in ['HEAD', 'index_entries_sha256', 'staged'])
    a, b = before['status'].splitlines(), cur['status'].splitlines()
    return {'head_index_entries_staged_unchanged': same, 'decided_here': False, 'status_lines_added': sorted(set(b) - set(a)), 'status_lines_removed': sorted(set(a) - set(b)),
            'index_file_unchanged': cur['index_sha256'] == before['index_sha256']}


def expected_cost_arms(sizes, release):
    arms = []
    for T in TS:
        for k in REPEATS:
            arms += [f'P0-{n}-T{T}-r{k}' for n in sizes] + [f'F0reg1-T{T}-r{k}', f'F0reg3-T{T}-r{k}'] + ([f'F0rel-T{T}-r{k}'] if release and T in RELEASE_TS else [])
    return arms


def parse_ts(text):
    """nvidia-smi prints local time without a zone; the samplers and the acceptance run on the same node (the offset is recorded)."""
    return datetime.strptime(text.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()


def read_rows(path, ncol):
    rows = []
    for line in path.read_text(errors='replace').splitlines() if path.exists() else []:
        f = [x.strip() for x in line.split(',')]
        if len(f) == ncol:
            try:
                rows.append([parse_ts(f[0])] + f[1:])
            except ValueError:                         # a line cut when the sampler was stopped, or an nvidia-smi message
                pass
    return rows


def windows(cost, clock='unix'):
    """Window of every arm: the timed call; the release-default arm from the start of its data step; an arm that ran out of memory over its
    attempt. clock='unix' aligns with the nvidia-smi rows (B4); clock='perf' is monotonic and is the one the overlap check uses (follow-up 1 c)."""
    out = {}
    for arm, v in cost.items():
        if v.get('status', 'completed') == 'completed':
            out[arm] = (v.get(f'attempt_{clock}_start', v[f'{clock}_start']), v[f'{clock}_end'])
        else:
            out[arm] = (v[f'attempt_{clock}_start'], v[f'attempt_{clock}_end'])
    return out


def device_summary(rows):
    def num(i):
        out = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except ValueError:                         # '[N/A]'
                pass
        return out
    used, clock, power, temp = num(2), num(4), num(5), num(6)
    return {'samples': len(rows), 'used_mib_max': max(used, default=None), 'sm_clock_mhz_median': statistics.median(clock) if clock else None,
            'sm_clock_mhz_min': min(clock, default=None), 'power_w_max': max(power, default=None), 'temperature_c_max': max(temp, default=None)}


def exclusivity(job, runner_pid, cost, slurm):
    """Section 6 item 6 (initial review B4). nvidia-smi lists every compute process on the card once a second (gpuapps); the runner's own rows,
    present while it holds its CUDA context, show that the sampler covered a window. An arm is covered only if the runner's rows within GAP s of
    its window include one at or before its start and one at or after its end, with no gap above GAP s. A row of a process that is none of this
    job's runners (memstats PIDs), within the window +-1 s, marks the arm possibly_affected; an uncovered arm without such a row reads unknown,
    never not affected. A process that lived for less than a sampling period between two rows can be missed. gpumem (every 0.5 s) gives the
    device conditions in each window (used memory, SM clock, power, temperature; initial review N7), descriptive only."""
    f = Path(slurm) / f'gpuapps-{job}.csv'; mem = Path(slurm) / f'memstats-{job}.jsonl'; win = windows(cost)
    if not f.exists():
        return {'sampler': 'gpuapps log not found', 'arms': {a: {'reading': 'unknown'} for a in win}, 'arms_possibly_affected': [], 'arms_unknown': sorted(win)}
    job_pids = {json.loads(l)['pid'] for l in mem.read_text().splitlines() if l.strip()} if mem.exists() else set()
    apps = [(t, int(pid)) for t, _, pid, used in read_rows(f, 4) if pid.isdigit()]     # used memory may read [N/A] (follow-up 1 a): PID only
    own = sorted(t for t, pid in apps if pid == runner_pid)
    other = [(t, pid) for t, pid in apps if pid != runner_pid and pid not in job_pids]
    dev = read_rows(Path(slurm) / f'gpumem-{job}.csv', 7); arms = {}
    for arm, (s0, e0) in win.items():
        S = [t for t in own if s0 - GAP <= t <= e0 + GAP]
        covered = bool(S) and S[0] <= s0 and S[-1] >= e0 and all(b - a <= GAP for a, b in zip(S, S[1:]))
        seen = sorted({pid for t, pid in other if s0 - 1 <= t <= e0 + 1})
        arms[arm] = {'window_unix': [s0, e0], 'covered': covered, 'own_rows': len(S), 'other_pids': seen,
                     'reading': 'possibly_affected' if seen else ('no_other_process_seen' if covered else 'unknown'),
                     'device': device_summary([x for x in dev if s0 <= x[0] <= e0])}
    return {'sampler': 'gpuapps every 1 s, gpumem every 0.5 s', 'utc_offset_seconds': time.localtime().tm_gmtoff, 'rows': len(apps), 'runner_pid': runner_pid,
            'job_runner_pids': sorted(job_pids), 'other_compute_pids': sorted({p for _, p in other}), 'arms': arms,
            'arms_possibly_affected': sorted(a for a, v in arms.items() if v['reading'] == 'possibly_affected'),
            'arms_unknown': sorted(a for a, v in arms.items() if v['reading'] == 'unknown')}


def timing_sane(cost, done, total):
    """Initial review N10: every completed arm has a positive time, nonnegative parts that add up with fit_seconds to it, 0 < peak allocated <=
    peak reserved <= the card, nonnegative increments over the baseline and nonnegative allocator counter increases; the arms' windows on the
    monotonic clock do not overlap (one process, one arm at a time; follow-up 1 c: unix time can step under NTP, so it is not checked)."""
    for a in done:
        v = cost[a]
        assert v['seconds'] > 0 and all(x >= 0 for x in v['parts_seconds'].values()) and v['fit_seconds'] >= -1e-6, a
        assert abs(v['seconds'] - v['fit_seconds'] - sum(v['parts_seconds'].values())) < 1e-6, a
        assert 0 < v['peak_allocated_bytes'] <= v['peak_reserved_bytes'] <= total, a
        assert v['peak_allocated_increment_bytes'] >= 0 and v['peak_reserved_increment_bytes'] >= 0, a
        assert v['num_ooms_increase'] >= 0 and v['num_alloc_retries_increase'] >= 0, a
        assert v['perf_start'] < v['perf_end'] and v.get('attempt_perf_start', v['perf_start']) <= v['perf_start'], a
    w = sorted(windows(cost, 'perf').values())
    assert all(a[1] <= b[0] for a, b in zip(w, w[1:])), 'arm windows overlap'


def cost_table(cost, arms, solver_batch):
    """Section 5 (initial review B1, B2, N1). Both repeats; the table value is the second repeat. Time: wall (the release-default arm: its
    separately timed data step plus the timed update), wall minus data, and fit_seconds (wall of the timed call minus every wrapped part).
    Memory: the peaks and their increments over the resident baseline (networks, teacher, test images); the ratios use the increments.
    Forwards: P0 = teacher forwards (one per capture batch) + prefix capture forwards (one per interface and capture batch, each only through
    the earlier layers) + full-network replay forwards (one per capture batch); FTBC = the counted full ANN and SNN forwards. Ratios to F0reg1 of
    the same T are descriptive."""
    table = {}
    for arm in arms:
        base, k = arm.rsplit('-r', 1); v = cost[arm]
        entry = table.setdefault(base, {'T': v['T'], 'method': v['method']})
        if v.get('status', 'completed') != 'completed':
            entry[f'r{k}'] = {kk: v.get(kk) for kk in ['status', 'stage', 'oom_basis', 'exception_type', 'num_ooms_increase', 'device_free_bytes', 'peak_allocated_bytes',
                                                       'peak_reserved_bytes', 'card_total_bytes', 'data_seconds']}
            continue
        parts = dict(v['parts_seconds']); sep = v.get('data_seconds') or 0.0     # sep: the release-default data step, timed outside the update
        e = {'wall_seconds': v['seconds'] + sep, 'wall_minus_data_seconds': v['seconds'] - parts.get('data', 0.0), 'fit_seconds': v['fit_seconds'],
             'data_seconds': parts.get('data', 0.0) + sep, 'parts_seconds': parts, 'timed_call_seconds': v['seconds']}
        e |= {kk: v[kk] for kk in ['peak_allocated_bytes', 'peak_reserved_bytes', 'peak_allocated_increment_bytes', 'peak_reserved_increment_bytes',
                                     'baseline_allocated_bytes', 'baseline_reserved_bytes', 'fit_images', 'num_ooms_increase', 'num_alloc_retries_increase']}
        if v['method'] == 'P0':
            e |= {kk: v[kk] for kk in ['layer_seconds_sum', 'captures', 'capture_batches', 'cache_bytes_max']}
            e['forwards'] = {'teacher': v['capture_batches'], 'prefix_capture': v['captures'] * v['capture_batches'], 'full_replay': v['capture_batches'], 'batch_size': solver_batch}
        else:
            e |= {kk: v[kk] for kk in ['batches', 'batch_size']}
            e['forwards'] = {'ann_full': v['calls']['ann'], 'snn_full': v['calls']['snn'], 'batch_size': v['batch_size']}
        entry[f'r{k}'] = e
    ratio = lambda a, b: a / b if b else None
    for base, entry in table.items():
        ref = table.get(f"F0reg1-T{entry['T']}", {}).get('r2'); own = entry.get('r2'); r1 = entry.get('r1')
        if ref and own and 'wall_seconds' in ref and 'wall_seconds' in own:
            entry['ratio_to_F0reg1_r2'] = {kk: ratio(own[kk], ref[kk]) for kk in ['wall_seconds', 'wall_minus_data_seconds', 'fit_seconds',
                                                                                  'peak_allocated_increment_bytes', 'peak_reserved_increment_bytes']}
        if r1 and own and 'wall_seconds' in r1 and 'wall_seconds' in own:
            entry['r1_minus_r2'] = {'wall_seconds': r1['wall_seconds'] - own['wall_seconds'], 'fit_seconds': r1['fit_seconds'] - own['fit_seconds'],
                                    'parts_seconds': {p: r1['parts_seconds'].get(p, 0.0) - own['parts_seconds'].get(p, 0.0)
                                                      for p in sorted(set(r1['parts_seconds']) | set(own['parts_seconds']))}}
    return table


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
    cost = json.loads((o / 'cost.json').read_text()); want = expected_cost_arms(sizes, True); assert sorted(cost) == sorted(want), sorted(set(cost) ^ set(want))
    done = [a for a in want if cost[a].get('status', 'completed') == 'completed']
    assert all(a in done for a in want if not a.startswith('F0rel')), 'only release-default arms may be recorded as not completed (section 7)'
    for a in want:
        if a not in done:                              # initial review B3, follow-up 1 B5 and b: out of memory only, with its basis, stage and peaks
            v = cost[a]; assert v['status'] == 'out_of_memory' and v['stage'] in ('data', 'update') and v['card_total_bytes'] == res['hardware']['total_memory'], a
            assert v['oom_basis'] in ('exception', 'library') and v['num_ooms_increase'] >= 0 and v['attempt_perf_start'] < v['attempt_perf_end'], a
            if v['oom_basis'] == 'library':            # a pre-registered library signature backed by allocator evidence
                assert any(sig.lower() in v['exception_message'].lower() for sig in LIB_OOM) and (v['num_ooms_increase'] > 0 or v['device_free_bytes'] < GIB), a
            if v['stage'] == 'update':
                assert 0 <= v['peak_allocated_bytes'] <= v['peak_reserved_bytes'], a
            else:
                assert v['peak_allocated_bytes'] is None and v['peak_reserved_bytes'] is None, a
    assert sorted(res['arms']) == sorted(done) and sorted(res['cost_arms']) == sorted(want)
    assert res['allocator_conf'] == 'expandable_segments:True', res.get('allocator_conf')     # N6: the reserved peaks depend on the allocator mode
    timing_sane(cost, done, res['hardware']['total_memory'])
    # Frozen network files and the networks the run built (as E20/E22).
    art = json.loads(Path(c['e0_artifacts']).read_text())
    for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt')]:
        assert Path(c[key]).name == name and Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
    fdig = state_dict_digest(tload(c['plain_proxy'])['state_dict'])
    assert res['teacher_digest'] == fdig and set(res['proxy_digests']) == {f'plain-T{T}' for T in TS} and all(v == fdig for v in res['proxy_digests'].values())
    dep = sha(c['plain_deployment']); assert res['deployment_sha256'] == {'plain': dep}
    # Data identity (section 6 item 3).
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256']
    fresh = json.loads(Path(c['fresh_manifest']).read_text())['development'][:c['fresh_count']]
    e5 = json.loads(Path(c['e5_roles']).read_text())
    old = [r for v in roles.values() for r in v] + [r for v in e5.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
    old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
    cal = json.loads(Path(c['calibration_manifest']).read_text()); allc = [r for k in CAL for r in cal[k]]
    groups = [fresh, allc]
    for i, g in enumerate(groups):
        assert len({int(r['index']) for r in g}) == len(g) == len({r['sha256'] for r in g})
        assert not {int(r['index']) for r in g} & {int(r['index']) for r in old} and not {r['sha256'] for r in g} & {r['sha256'] for r in old}
        for h in groups[i + 1:]:
            assert not {int(r['index']) for r in g} & {int(r['index']) for r in h} and not {r['sha256'] for r in g} & {r['sha256'] for r in h}
    draws = {k: cal[k][:c['draw_n']] for k in CAL}; assert res['draws'] == {d: [int(x['index']) for x in v] for d, v in draws.items()}
    rel_rows = cal['C1'][:c['release_n'] * c['release_batch']]; assert res['release_rows'] == [int(x['index']) for x in rel_rows]   # the pool C1 (section 2)
    rowsets = {f'C1-{n}': draws['C1'][:n] for n in sizes}
    # Biases, strict reload, solver records and FTBC calls.
    deploy = tload(c['plain_deployment'])['state_dict']; recs = json.loads((o / 'metrics.json').read_text())
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text()); bias = {}; names = None; hashes = {}
    for arm in done:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == dep and recs[arm]['strict_reload']
        assert recs[arm]['detail']['state'] == 'plain' and recs[arm]['detail']['deployment_sha256'] == dep, arm
        assert set(f['state_dict']) == set(deploy) and all(torch.equal(f['state_dict'][k], deploy[k]) for k in deploy if 'time_based_bias' not in k), arm
        Tx = f['T']; assert Tx == cost[arm]['T'] == int(arm.split('-T')[1].split('-')[0]) and len(f['runtime_biases']) == L
        assert all(len(v) == Tx and all(torch.isfinite(b).all() for b in v) for v in f['runtime_biases'].values())
        names = list(f['runtime_biases']); bias[arm] = f['runtime_biases']
        hashes[arm] = bias_hash(f['runtime_biases'], names, Tx); assert hashes[arm] == recs[arm]['bias_sha256'] == res['bias_sha256'][arm], arm
    tfiles = {f'C1-{n}-teacher': tload(o / f'targets-C1-{n}-teacher.pt') for n in sizes}
    for arm in [a for a in done if a.startswith('P0')]:
        n = int(arm.split('-')[1]); key = f'C1-{n}'; s = solver[arm]; Tx = cost[arm]['T']
        assert s['full_network_replay_equal'] and s['per_step'] and s['fit_images'] == n == cost[arm]['fit_images'] and s['state'] == 'plain' and s['T'] == Tx, arm
        tk = f'{key}-teacher'; assert s['target_key'] == tk == recs[arm]['detail']['target_key'] and recs[arm]['detail']['solver'] == 'P0', arm
        assert tfiles[tk]['ids'].tolist() == [int(x['index']) for x in rowsets[key]] and tfiles[tk]['target'] == 'teacher' and tfiles[tk]['network_digest'] == fdig
        assert list(s['layers']) == list(tfiles[tk]['means']) == names and cost[arm]['captures'] == L
        assert abs(cost[arm]['layer_seconds_sum'] - sum(v['seconds'] for v in s['layers'].values())) < 1e-6 and cost[arm]['layer_seconds_sum'] <= cost[arm]['seconds']
        for nm, info in s['layers'].items():
            b = info['b']; assert torch.equal(info['target_mean'], tfiles[tk]['means'][nm]) and b.shape[0] == Tx and within(b, c['bound_ratio'] * info['theta']), (arm, nm)
            assert all(torch.equal(bias[arm][nm][t], b[t]) for t in range(Tx))
    assert sorted(solver) == sorted(a for a in done if a.startswith('P0'))
    assert sorted(ftbc) == sorted(a for a in done if a.startswith('F0'))
    for arm in [a for a in done if a.startswith('F0')]:
        v = ftbc[arm]; calls = v['calls']; Tx = cost[arm]['T']
        zh = bias_hash({nm: [torch.zeros_like(b) for b in bias[arm][nm]] for nm in names}, names, Tx)
        assert v['alpha'] == c['ftbc_alpha'] and v['init_bias_sha256'] == zh and v['ann'] == 'teacher' and v['ann_digest'] == fdig and v['ann_level'] == [8], arm
        assert calls['ann'] == calls['snn'] == L * v['batches'] and len(calls['updates']) == L and all(x == v['batches'] for x in calls['updates'].values()), arm
        assert cost[arm]['calls'] == calls, arm
        if arm.startswith('F0reg'):
            passes = int(arm[len('F0reg')]); assert passes in (1, 3), arm
            assert v['recipe'] == ('registered' if passes == 1 else 'registered-3pass') and v['batches'] == 3 * passes and v['passes'] == passes, arm
            assert v['batches_per_pass'] == 3 and v['batch_size'] == c['ftbc_batch'] and v['images'] == [int(x['index']) for x in draws['C1'][:small]], arm
        else:
            assert v['recipe'] == 'release-default' and v['batches'] == c['release_n'] and v['batch_size'] == c['release_batch'] and v['seed'] == c['release_seed'], arm
            assert sorted(v['images']) == sorted(int(x['index']) for x in rel_rows) and cost[arm]['data_seconds'] > 0, arm
    # Section 6 item 4: determinism between the two repeats (recorded; a difference is flagged, not fatal).
    bases = sorted({a.rsplit('-r', 1)[0] for a in done})
    determinism = {b: (hashes[f'{b}-r1'] == hashes[f'{b}-r2']) if f'{b}-r1' in hashes and f'{b}-r2' in hashes else None for b in bases}
    manuscript = manuscript_check(w / 'manuscript', json.loads((E / 'preservation-before.json').read_text()))
    out = {'passed': True, 'model': MODEL, 'scope': 'sanity only' if c['sanity'] else 'fitting-cost measurement; no accuracy reported',
           'cost_table': cost_table(cost, want, c['solver_batch']), 'repeat_bitwise_equal': determinism,
           'not_completed': {a: {k: cost[a].get(k) for k in ['status', 'stage', 'oom_basis', 'exception_type', 'num_ooms_increase', 'device_free_bytes', 'peak_reserved_bytes',
                                                             'leftover_hooks_dropped']} for a in want if a not in done},
           'arms_with_allocator_ooms': {a: cost[a]['num_ooms_increase'] for a in done if cost[a]['num_ooms_increase']},   # follow-up 1 B5: completed after a caught failure
           'exclusivity': exclusivity(res['hardware']['slurm_job_id'], res['hardware']['pid'], cost, E / 'slurm'),
           'resources': {k: res[k] for k in ['seconds_total', 'peak_reserved_since_last_reset', 'peak_allocated_since_last_reset', 'allocator_conf']} |
                        {'max_arm_peak_reserved_bytes': max(cost[a]['peak_reserved_bytes'] for a in done), 'max_arm_peak_allocated_bytes': max(cost[a]['peak_allocated_bytes'] for a in done),
                         'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id'], 'total_memory': res['hardware']['total_memory']},
           'manuscript': manuscript, 'not_method_claim': True, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not out['cuda_initialized']
    (E / f'{MODEL}-{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {'passed': True, 'model': MODEL, 'repeat_bitwise_equal': determinism, 'not_completed': out['not_completed'],
             'arms_possibly_affected': out['exclusivity']['arms_possibly_affected'], 'arms_unknown': out['exclusivity']['arms_unknown'],
             'wall_seconds_r2': {b: v.get('r2', {}).get('wall_seconds') for b, v in out['cost_table'].items()}}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
