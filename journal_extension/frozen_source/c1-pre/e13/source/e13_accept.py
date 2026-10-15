"""E13 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-7). No CUDA."""
import hashlib, json, math, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import serial_solver as ss
if not __debug__:
    raise SystemExit('E13 acceptance uses assert; do not run with -O')

Q1_MARGIN, Q2_MARGIN = 26, 10


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def tload(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def sign_p(b, c):
    n, k = b + c, min(b, c)
    return None if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(lab, x, y):
    a, b = x == lab, y == lab
    u, v = int((a & ~b).sum()), int((b & ~a).sum())
    return {'x_only_correct': u, 'y_only_correct': v, 'net': u - v, 'prediction_disagreements': int((x != y).sum()), 'exact_two_sided_sign_p_descriptive': sign_p(u, v)}


def check(which):
    c = json.loads((E / 'config' / f'{which}.json').read_text()); o = Path(c['output']); T = 4
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    res = json.loads((o / 'result.json').read_text())
    assert res['status'] == 'completed' and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed']
    assert res['twin_unchanged'] and set(res['proxy_digests']) == {str(x) for x in [4] + c['extra_T']}
    assert res['hardware']['gpu_uuid'] in c['allowed_uuids'] and 'slurm' in res['hardware']['cgroup'].lower()
    ftbc_arms = [f'F0-n{n}' for n in c['ftbc_ns']] + [f"F1-n{c['f1_n']}"]
    snn4 = ['S0', 'S1', 'S2', 'S3', 'M0', 'M1', 'L1'] + ftbc_arms
    extra = [f'{a}-T{x}' for x in c['extra_T'] for a in ['S0', 'S1']]
    ann = ['teacher', 'twin-T4'] + [f'twin-T{x}' for x in c['extra_T']]
    assert set(res['arms']) == set(snn4 + extra + ann), sorted(set(res['arms']) ^ set(snn4 + extra + ann))
    parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
    dev = roles['development'][:c['dev_count']]; ids = torch.tensor([r['index'] for r in dev]); labels = torch.tensor([r['label'] for r in dev])
    pred = tload(o / 'predictions.pt'); recs = json.loads((o / 'metrics.json').read_text())
    metrics = {}
    for arm in res['arms']:
        q = pred[arm]['development']
        assert torch.equal(q['ids'], ids) and torch.equal(q['labels'], labels) and torch.isfinite(q['logits']).all()
        assert torch.equal(q['logits'].argmax(1), q['predictions'])
        correct = int((q['predictions'] == labels).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), labels))
        r = recs[arm]['metrics']['development']; assert r['correct'] == correct and abs(r['cross_entropy'] - ce) < 1e-12
        metrics[arm] = {'correct': correct, 'n': len(labels), 'top1': 100 * correct / len(labels), 'ce': ce}
    # Real biases, schedules and solver records.
    deploy_sha = sha(c['plain_deployment']); thr = {}
    deploy = tload(c['plain_deployment'])
    for k, v in deploy['state_dict'].items():
        if k.endswith('.thresh'): thr[k[:-len('.thresh')]] = v.reshape(-1)[0]
    solver = tload(o / 'solver.pt'); ftbc = json.loads((o / 'ftbc.json').read_text())
    def expect_schedule(off, Tx):
        b = ss.biases_from_offsets(torch.as_tensor(off, dtype=torch.float64), Tx)
        return {n: [(b[t] * float(th)).to(torch.float32) for t in range(Tx)] for n, th in thr.items()}
    bias = {}
    for arm in snn4 + extra:
        f = tload(o / f'bias-{arm}.pt'); assert f['plain_deployment_sha256'] == deploy_sha and recs[arm]['strict_reload']
        Tx = f['T']; assert len(f['runtime_biases']) == 33
        for n, vals in f['runtime_biases'].items():
            assert len(vals) == Tx and all(torch.isfinite(v).all() for v in vals)
        bias[arm] = f['runtime_biases']
        sched = None
        if arm in ['S0'] or arm.startswith('S0-T'):
            assert all(not torch.count_nonzero(v) for vals in f['runtime_biases'].values() for v in vals)
        elif arm in ['S1', 'S2', 'S3'] or arm.startswith('S1-T'):
            off = {'S1': ss.deferral(4, 1.0), 'S2': ss.deferral(4, 0.5), 'S3': torch.tensor(c['s3_offsets_theta'], dtype=torch.float64)}.get(arm, ss.deferral(Tx, 1.0))
            exp = expect_schedule(off, Tx)
            for n, vals in f['runtime_biases'].items():
                assert all(torch.equal(v, e.expand_as(v)) for v, e in zip(vals, exp[n])), (arm, n)
    for n, info in solver['L1']['layers'].items():
        off = info['schedule']['history'][-1]['offsets']
        assert len(off) == 4 and off[3] == 0 and all(float(v * 32).is_integer() and 0 <= v * 32 <= 16 for v in off), (n, off)
    assert res['diagnostic_ids'] == [r['index'] for r in roles['diagnostic'][:c['diag_count']]]
    s1h = hashlib.sha256(b''.join(v.to(torch.float32).contiguous().numpy().tobytes() for vals in bias['S1'].values() for v in vals)).hexdigest()
    zh = hashlib.sha256(b''.join(v.to(torch.float32).contiguous().numpy().tobytes() for vals in bias['S0'].values() for v in vals)).hexdigest()
    assert ftbc[f"F1-n{c['f1_n']}"]['init_bias_sha256'] == s1h and all(ftbc[f'F0-n{n}']['init_bias_sha256'] == zh for n in c['ftbc_ns'])
    for arm in ['M0', 'M1', 'L1']:
        s = solver[arm]; assert s['full_network_replay_equal']
        for n, info in s['layers'].items():
            th = info['theta']; beta = info['beta'].to(torch.float32)
            assert float(beta.abs().max()) <= c['bound_ratio'] * th
            off = (torch.tensor(info['schedule']['history'][-1]['offsets'], dtype=torch.float64) if arm == 'L1'
                   else (ss.deferral(4, 1.0) if arm == 'M1' else torch.zeros(4, dtype=torch.float64)))
            b0 = (ss.biases_from_offsets(off, 4) * th).to(torch.float32).view(4, 1).expand(4, beta.numel())
            b = b0 - beta.view(1, -1)
            assert all(torch.equal(bias[arm][n][t], b[t]) for t in range(4)), (arm, n)
    for arm in ftbc_arms:
        nb = ftbc[arm]['batches']; calls = ftbc[arm]['calls']
        assert calls['teacher'] == calls['snn'] == 33 * nb and all(v == nb for v in calls['updates'].values()) and len(calls['updates']) == 33
        assert ftbc[arm]['images'] == [r['index'] for r in roles['fit'][:c['ftbc_batch'] * nb]]
    tt = tload(o / 'teacher-targets.pt'); assert tt['ids'].tolist() == [r['index'] for r in roles['fit'][:c['fit_count']]]
    # Manuscript Git (section 7 item 8).
    M = Path(c['workspace']) / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index'); assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Analysis.
    P = {a: pred[a]['development']['predictions'] for a in res['arms']}
    vs_s0 = {a: paired(labels, P[a], P['S0']) for a in snn4 if a != 'S0'}
    f0_best = max([f'F0-n{n}' for n in c['ftbc_ns']], key=lambda a: (metrics[a]['correct'], -metrics[a]['ce']))
    controls = [f0_best, f"F1-n{c['f1_n']}", 'M0', 'M1', 'S1']
    best_c = max(metrics[a]['correct'] for a in controls); low_ce = min(metrics[a]['ce'] for a in controls)
    vs_f0 = {a: paired(labels, P[a], P[f0_best]) for a in snn4 if a != f0_best}
    q1 = metrics['S1']['correct'] - metrics['S0']['correct']
    decisions = {'Q1': {'S1_minus_S0': q1, 'margin': Q1_MARGIN, 'reading': 'helps' if q1 >= Q1_MARGIN else ('hurts' if q1 <= -Q1_MARGIN else 'unclear')},
                 'Q2': {'controls': controls, 'best_control_correct': best_c, 'lowest_control_ce': low_ce, 'L1_correct': metrics['L1']['correct'],
                        'L1_ce': metrics['L1']['ce'], 'margin': Q2_MARGIN,
                        'exploratory_signal': metrics['L1']['correct'] >= best_c + Q2_MARGIN and metrics['L1']['ce'] <= low_ce,
                        'pairs': {a: paired(labels, P['L1'], P[a]) for a in controls}},
                 'Q3': {'M1_minus_M0': metrics['M1']['correct'] - metrics['M0']['correct'], 'L1_minus_M1': metrics['L1']['correct'] - metrics['M1']['correct'],
                        f"F1_minus_F0-n{c['f1_n']}": metrics[f"F1-n{c['f1_n']}"]['correct'] - metrics[f"F0-n{c['f1_n']}"]['correct'] if f"F0-n{c['f1_n']}" in metrics else None,
                        'S2_minus_S0': metrics['S2']['correct'] - metrics['S0']['correct'], 'S3_minus_S0': metrics['S3']['correct'] - metrics['S0']['correct'],
                        **{f'S1_minus_S0_T{x}': metrics[f'S1-T{x}']['correct'] - metrics[f'S0-T{x}']['correct'] for x in c['extra_T']},
                        'pairs': {'M1-vs-M0': paired(labels, P['M1'], P['M0']), 'L1-vs-M1': paired(labels, P['L1'], P['M1'])}}}
    diag = json.loads((o / 'diagnostics.json').read_text())
    layers = list(diag['S0'])
    def layer_avg(arm, key, sel):
        v = [diag[arm][n][key] for n in layers[1:] if sel(n)]; return sum(v) / len(v)
    deep = lambda n: n.startswith('conv4') or n.startswith('conv5')
    mech = {arm: {k: layer_avg(arm, k, lambda n: True) for k in ['over_fraction', 'under_fraction', 'mean_abs_local', 'mean_abs_vs_twin']} |
                 {f'deep_{k}': layer_avg(arm, k, deep) for k in ['over_fraction', 'under_fraction', 'mean_abs_vs_twin']} for arm in diag}
    predictions_mech = {'a_S1_less_overfiring_layers2_33': mech['S1']['over_fraction'] < mech['S0']['over_fraction'],
                        'b_S1_more_deep_underfiring': mech['S1']['deep_under_fraction'] > mech['S0']['deep_under_fraction'],
                        'c_S1_less_deep_twin_deviation': mech['S1']['deep_mean_abs_vs_twin'] < mech['S0']['deep_mean_abs_vs_twin']}
    solver_summary = {}
    for arm in ['M0', 'M1', 'L1']:
        L = solver[arm]['layers']
        solver_summary[arm] = {'mean_abs_normalized_residual': sum(float(v['normalized_residual'].abs().mean()) for v in L.values()) / len(L),
                               'at_bound_channels': sum(int(v['at_bound'].sum()) for v in L.values()),
                               'below_range_channels': sum(int(v['below_range'].sum()) for v in L.values()),
                               'above_range_channels': sum(int(v['above_range'].sum()) for v in L.values()),
                               'mean_abs_beta_over_theta': sum(float(v['beta'].abs().mean()) / v['theta'] for v in L.values()) / len(L),
                               'solve_seconds': sum(v['seconds'] for v in L.values())}
    schedules = {n: {'offsets': v['schedule']['history'][-1]['offsets'], 'objective_init_deferral': v['schedule']['history'][0]['objective'],
                     'objective_final': v['schedule']['history'][-1]['objective'], 'objective_IF': v['schedule']['baseline_IF_objective']}
                 for n, v in solver['L1']['layers'].items()}
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'development screen on exposed E0 serial dev512; not confirmation or method claim',
           'metrics': metrics, 'decisions': decisions, 'paired_vs_S0': vs_s0, 'paired_vs_best_FTBC': vs_f0, 'mechanism': mech, 'mechanism_predictions': predictions_mech,
           'diagnostics_per_layer': diag, 'solver_summary': solver_summary, 'L1_schedules': schedules,
           'ftbc': {a: {'batches': v['batches'], 'max_abs_bias_over_theta': v['max_abs_bias_over_theta']} for a, v in ftbc.items()},
           'anchors_e0_descriptive': res['anchors'], 'f0_best_on_dev': f0_best,
           'resources': {k: res[k] for k in ['seconds_total', 'max_memory_reserved', 'max_memory_allocated']} | {'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id']},
           'manuscript_git_unchanged': True, 'not_method_innovation_yet': True, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'passed': True, 'dev': {a: (m['correct'], round(m['ce'], 4)) for a, m in metrics.items()}, 'decisions': {k: {kk: vv for kk, vv in v.items() if kk != 'pairs'} for k, v in decisions.items()},
                      'mechanism_predictions': predictions_mech, 'solver': solver_summary}, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
