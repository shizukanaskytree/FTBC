"""E12 CPU acceptance and pre-registered analysis (EXPERIMENT_PLAN.md sections 5-10). No CUDA."""
import hashlib, json, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch
torch.set_num_threads(4)
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12_relative as rel

P1_STATES = ['bn-only', 'kl-e4', 'Cabs', 'Crel', 'Transfer']
P0_STATES = ['p0-zero', 'C0']
STATES = P1_STATES[:2] + P0_STATES + P1_STATES[2:]
SOLVED = {'C0': 'a', 'Cabs': 'a', 'Crel': 'm'}
E7_STATE = {'bn-only': 'bn-only', 'kl-e4': 'kl-e4', 'Cabs': 'finite-solve'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def tload(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def check(which):
    c = json.loads((E / 'config' / f'{which}.json').read_text()); o = Path(c['output'])
    for p, h in json.loads((E / 'protocol-freeze.json').read_text())['files_sha256'].items():
        assert sha(p) == h, p
    res = json.loads((o / 'result.json').read_text()); res0 = json.loads((o / 'P0/result.json').read_text())
    assert res['status'] == 'completed' and res0['status'] == 'completed' and res['arms'] == ['bn-only', 'kl-e4', 'p0-zero', 'C0', 'Cabs', 'Crel', 'Transfer']
    assert res['full_fit_replay_equal'] and res['teacher_unchanged'] and res['non_bias_unchanged'] and res['all_strict_reloads_passed']
    assert res0['teacher_unchanged'] and res0['non_bias_unchanged'] and res0['all_strict_reloads_passed']
    assert res['hardware']['gpu_uuid'] == c['allowed_uuid'] == res0['hardware']['gpu_uuid']
    for h in (res['hardware'], res0['hardware']):
        assert 'slurm' in h['cgroup'].lower() and h['slurm_job_id'] == res['hardware']['slurm_job_id']
    st = res['state_check']
    assert st['interfaces'] == 33 and st['bn_state_tensors'] == st['differing_tensors'] == 180 and st['p0_source_shared_tensors'] == 218
    assert all(st[k] for k in ['only_bn_differs', 'all_bn_differ', 'p0_equals_source_on_shared_keys', 'thresholds_equal', 'teacher_digest_equal', 'zero_biases'])
    # Section 10 item 12: manuscript Git unchanged since the E12 baseline.
    M = Path(c['workspace']) / 'manuscript'
    git = {k: subprocess.check_output(['git', '-C', str(M), *x], text=True) for k, x in [('HEAD', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain=v1', '--untracked-files=all'])]}
    git['index_sha256'] = sha(M / '.git/index')
    assert git == json.loads((E / 'preservation-before.json').read_text()), 'manuscript Git changed'
    # Section 10 item 9: independent log evidence that m was frozen before the first solve.
    job = json.loads(json.loads((E / f'slurm/{which}-submit.json').read_text())['stdout'])['job_id']
    assert str(job) == str(res['hardware']['slurm_job_id'])
    stages = []
    for line in (E / f'slurm/{which}-{job}.out').read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if isinstance(x, dict) and 'stage' in x:
            stages.append(x['stage'])
    assert stages.index('relative_target_frozen') < stages.index('fitted_layer') and stages.count('relative_target_frozen') == 1
    roles = json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text())
    roles = roles.get('roles', roles)
    ids = {r: torch.tensor([x['index'] for x in roles[k][:n]]) for r, k, n in [('fit', 'fit', c['fit_count']), ('development', 'development', c['dev_count'])]}
    labels = {r: torch.tensor([x['label'] for x in roles[k][:n]]) for r, k, n in [('fit', 'fit', c['fit_count']), ('development', 'development', c['dev_count'])]}

    pred = {**tload(o / 'predictions.pt'), **tload(o / 'P0/predictions.pt')}
    records = {r['id']: r for r in json.loads((o / 'metrics-progress.json').read_text()) + json.loads((o / 'P0/metrics-progress.json').read_text())}
    assert set(pred) == set(records) == set(STATES)
    meas = tload(o / 'measurements.pt'); fits = tload(o / 'fits.pt')
    assert meas['fit_batch'] == meas['dev_batch'] == 8 and set(meas['fit']) == set(meas['development']) == set(STATES)
    targets = tload(o / 'targets.pt'); dev_targets = tload(o / 'dev-targets.pt')
    ref = {k: Path(c['e7_reference']['output']) / v for k, v in c['e7_reference']['files'].items()}
    e7t, e7m, e7p, e7b = tload(ref['targets']), tload(ref['measurements']), tload(ref['predictions']), tload(ref['bias_finite_solve'])
    sha_p0, sha_p1 = sha(c['p0']['parent_deployment']), sha(c['p1']['parent_deployment'])
    names = list(targets['means']); assert len(names) == 33

    # Data identity, metrics recomputed from saved logits.
    metrics = {}
    for s in STATES:
        metrics[s] = {}
        for role in ['fit', 'development']:
            q = pred[s][role]
            assert torch.equal(q['ids'], ids[role]) and torch.equal(q['labels'], labels[role])
            assert torch.isfinite(q['logits']).all() and torch.equal(q['logits'].argmax(1), q['predictions'])
            correct = int((q['predictions'] == labels[role]).sum()); ce = float(torch.nn.functional.cross_entropy(q['logits'].double(), labels[role]))
            rec = records[s]['metrics'][role]; assert rec['correct'] == correct and abs(rec['cross_entropy'] - ce) < 1e-12
            metrics[s][role] = {'correct': correct, 'n': len(labels[role]), 'top1': 100 * correct / len(labels[role]), 'ce': ce}
        assert records[s]['strict_reload'] and records[s]['non_bias_frozen']
    assert torch.equal(dev_targets['ids'], ids['development']) and torch.equal(targets['ids'], ids['fit'])

    # Real biases.
    bias = {}
    thetas = tload(c['p1']['parent_deployment'])['interface_thresholds']
    assert thetas == tload(c['p0']['parent_deployment'])['interface_thresholds'] and list(thetas) == names
    for s in STATES:
        f = tload((o / 'P0' if s in P0_STATES else o) / f'bias-{s}.pt')
        assert f['parent_sha256'] == (sha_p0 if s in P0_STATES else sha_p1) and len(f['biases']) == 33 and f['thresholds'] == thetas
        for n, v in f['biases'].items():
            assert torch.isfinite(v).all() and torch.equal(v, v[:1].expand_as(v)) and float(v.abs().max()) <= .5 * thetas[n]
        bias[s] = f['biases']
    assert all(not torch.count_nonzero(bias[s][n]) for s in ['bn-only', 'p0-zero'] for n in names)
    assert all(torch.equal(bias['Transfer'][n], bias['C0'][n]) for n in names)
    assert all(torch.equal(bias['Cabs'][n], e7b['biases'][n]) for n in names)
    kl = tload(c['kl_bias']); assert all(torch.equal(bias['kl-e4'][n], kl['biases'][n]) for n in names)

    # B3 targets: E7 bitwise, frozen m and descriptors written before the relative solve.
    assert torch.equal(targets['ids'], e7t['ids']) and torch.equal(targets['labels'], e7t['labels'])
    assert all(torch.equal(targets['means'][n], e7t['means'][n]) for n in names)
    man = json.loads((o / 'target-manifest.json').read_text()); assert man['written_before_relative_solve']
    assert sha(o / 'relative-target.pt') == man['relative_target_sha256'] and sha(o / 'descriptors.json') == man['descriptors_sha256']
    rt = tload(o / 'relative-target.pt')
    p0 = rel.means_from_counts(meas['fit']['p0-zero']); p1 = rel.means_from_counts(meas['fit']['bn-only'])
    a = {n: targets['means'][n] for n in names}
    desc, m = rel.descriptors(a, p0, p1, thetas)
    assert all(torch.equal(rt['m'][n], m[n]) and torch.equal(rt['p0'][n], p0[n]) and torch.equal(rt['p1'][n], p1[n]) for n in names)
    assert json.loads((o / 'descriptors.json').read_text()) == json.loads(json.dumps(desc))
    own = {'a': a, 'm': m}

    # Solved arms: solver records equal full-network replay, saved real bias and frozen target.
    arms = {}
    for s, t in SOLVED.items():
        layers = {}
        for n in names:
            f = fits[s][n]; mm = meas['fit'][s][n]
            assert torch.equal(f['after_count'], mm['count']) and f['observations_per_channel'] == mm['observations_per_channel']
            assert torch.equal(f['delta'], bias[s][n][0].flatten()) and torch.equal(f['target_mean'], own[t][n])
            r = mm['count'].double() / mm['observations_per_channel'] - own[t][n] / thetas[n]
            assert torch.allclose(r, f['normalized_residual'], rtol=0, atol=1e-15)
            outside = f['below_range'] | f['above_range']
            jump = (~outside) & f['bracketed'] & f['float_adjacent'] & ((f['hi_count'] - f['lo_count']) > 1)
            other = ~(outside | jump)
            layers[n] = {'M': float(r.abs().mean()), 'within_quantum': int((r.abs() <= 1 / f['observations_per_channel'] + 1e-7).sum()),
                         'residual_partition': {k: {'channels': int(z.sum()), 'absolute_residual_sum': float(r[z].abs().sum())} for k, z in
                                                [('outside_bound_support', outside), ('bracketed_adjacent_count_jump', jump), ('other_unresolved', other)]},
                         'selected_zero': int((f['delta'] == 0).sum()), 'selected_boundary': int((f['delta'].abs() == .5 * f['theta']).sum()),
                         'support': {k: int(f[k].sum()) for k in ['below_range', 'above_range', 'at_lower_mean', 'at_upper_mean', 'bracketed', 'float_adjacent']},
                         'max_lo_hi_count_gap': int((f['hi_count'] - f['lo_count']).max()), 'seconds': f['seconds']}
        tot = lambda key: sum(v[key] for v in layers.values())
        part = {k: {'channels': sum(v['residual_partition'][k]['channels'] for v in layers.values()),
                    'absolute_residual_sum': sum(v['residual_partition'][k]['absolute_residual_sum'] for v in layers.values())}
                for k in ['outside_bound_support', 'bracketed_adjacent_count_jump', 'other_unresolved']}
        arms[s] = {'target': t, 'M_own_target_channel_equal': sum(v['M'] * fits[s][n]['target_mean'].numel() for n, v in layers.items()) / 7616,
                   'M_own_target_layer_equal': sum(v['M'] for v in layers.values()) / 33, 'within_quantum_fraction': tot('within_quantum') / 7616,
                   'selected_zero': tot('selected_zero'), 'selected_boundary': tot('selected_boundary'), 'residual_partition': part,
                   'support': {k: sum(v['support'][k] for v in layers.values()) for k in layers[names[0]]['support']},
                   'solve_seconds': tot('seconds'), 'layers': layers}

    # E7 bitwise anchors (same card; batch8 counts, batch4 logits).
    for s, t in E7_STATE.items():
        assert all(torch.equal(meas['fit'][s][n]['count'], e7m[t][n]['count']) for n in names), s
        for role in ['fit', 'development']:
            assert torch.equal(pred[s][role]['logits'], e7p[t][role]['logits']) and torch.equal(pred[s][role]['ids'], e7p[t][role]['ids'])
    replays = json.loads((o / 'reference-replays.json').read_text()); assert set(replays) == {'bn-only', 'kl-e4', 'Cabs'}

    # Analysis: B1 primary pair and stop rule; secondary pairs; R1; B2; fit identity check; geometry; bias magnitudes.
    dev = {s: {'correct': metrics[s]['development']['correct'], 'ce': metrics[s]['development']['ce']} for s in STATES}
    lab = labels['development']; P = {s: pred[s]['development']['predictions'] for s in STATES}
    decision = rel.decide(dev)
    pairs = {f'{x}-vs-{y}': rel.paired(lab, P[x], P[y]) for x, y in
             [('Crel', 'Transfer'), ('Transfer', 'bn-only'), ('Transfer', 'Cabs'), ('C0', 'p0-zero'), ('Cabs', 'bn-only'),
              ('Crel', 'Cabs'), ('Crel', 'bn-only'), ('kl-e4', 'bn-only'), ('Crel', 'kl-e4'), ('C0', 'bn-only'), ('p0-zero', 'bn-only')]}
    fit_pairs = {f'{s}-vs-bn-only': rel.paired(labels['fit'], pred[s]['fit']['predictions'], pred['bn-only']['fit']['predictions']) for s in STATES if s != 'bn-only'}
    geometry = {'Crel-vs-Transfer(delta_r vs delta_0)': rel.geometry(bias['Crel'], bias['C0'], thetas),
                'Cabs-vs-C0(delta_a vs delta_0)': rel.geometry(bias['Cabs'], bias['C0'], thetas),
                'Crel-vs-Cabs(delta_r vs delta_a)': rel.geometry(bias['Crel'], bias['Cabs'], thetas)}
    mu_dev = {s: rel.means_from_counts(meas['development'][s]) for s in STATES}
    mu_fit = {s: rel.means_from_counts(meas['fit'][s]) for s in STATES}
    b2 = rel.edev(mu_dev, thetas)
    fit_transfer = rel.edev(mu_fit, thetas, arms=('Transfer',))['Transfer']
    dev_teacher = {s: rel.teacher_residual(mu_dev[s], dev_targets['means'], thetas) for s in STATES}
    fit_teacher = {s: rel.teacher_residual(mu_fit[s], a, thetas) for s in STATES}
    ident = 0.
    for n in names:
        lhs = (mu_fit['Crel'][n] - mu_fit['C0'][n]) - (p1[n] - p0[n])
        rhs = thetas[n] * (fits['Crel'][n]['normalized_residual'] - fits['C0'][n]['normalized_residual'])
        ident = max(ident, float((lhs - rhs).abs().max()))
    assert ident <= 1e-12, ident
    out = {'passed': True, 'scope': 'sanity only' if c['sanity'] else 'single exposed-development diagnostic; ordinary combination control, not method novelty',
           'metrics': metrics, 'decision_B1': decision, 'rule_R1': rel.rule_r1(desc, dev), 'descriptors': desc, 'pairs_development': pairs,
           'pairs_fit_in_sample_vs_bn': fit_pairs, 'bias_geometry': geometry,
           'bias_summary': {s: rel.bias_summary(bias[s], thetas) for s in ['C0', 'Cabs', 'Crel', 'kl-e4']},
           'B2_E_dev': b2, 'E_fit_Transfer_descriptive': fit_transfer, 'teacher_mean_residual_dev': dev_teacher, 'teacher_mean_residual_fit': fit_teacher,
           'fit_identity_solver_execution_check_max_abs': ident, 'solved_arms': arms,
           'manuscript_git_unchanged': True, 'slurm_log_stage_order': {'relative_target_frozen_before_first_fitted_layer': True},
           'anchors': {'teacher_targets_bitwise_e7': True, 'bn-only_counts_logits_bitwise_e7': True, 'kl-e4_counts_logits_bitwise_e7': True,
                       'Cabs_bias_counts_logits_bitwise_e7': True, 'transfer_bias_equals_C0': True, 'm_and_descriptors_hash_before_solve': True},
           'resources': {'seconds_total': res['seconds_total'], 'max_memory_reserved': res['max_memory_reserved'],
                         'max_memory_allocated': res['max_memory_allocated'], 'p0_runtime_max_memory_reserved': res0['max_memory_reserved'],
                         'gpu': res['hardware']['name'], 'uuid': res['hardware']['gpu_uuid'], 'slurm_job_id': res['hardware']['slurm_job_id']},
           'not_method_innovation': True, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not out['cuda_initialized']
    (E / f'{which}-acceptance.json').write_text(json.dumps(out, indent=2) + '\n')
    brief = {k: out[k] for k in ['passed', 'scope', 'decision_B1', 'rule_R1', 'resources']}
    brief['dev'] = {s: metrics[s]['development'] for s in STATES}
    brief['global_descriptors'] = desc['global_theta_normalized']; brief['B2'] = {s: v['channel_equal'] for s, v in b2.items()}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    check(sys.argv[1])
