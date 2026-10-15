"""E12 frozen combination-control diagnostic: C0, Cabs, Crel, Transfer in one process.

GPU only inside the project's own Slurm allocation (gpu_guard). Reuses unchanged frozen code:
E4 common.prepare/Runtime (scoring, real bias save + strict reload) and E7 run_diagnostic
(teacher_targets, measure, fit_one with the original solver, verify_prediction_reference).
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E.parent / 'e7/source'))
import run_diagnostic as e7run          # inserts e4/source and imports the original E7 solver
from common import prepare
import torch
import e12_relative as rel

ARM_ORDER = ['bn-only', 'kl-e4', 'p0-zero', 'C0', 'Cabs', 'Crel', 'Transfer']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def derived(c, branch, output):
    """Config consumed by the frozen E4 prepare/Runtime for one network state."""
    keys = ['workspace', 'parent_config', 'parent_runner', 'prior_freeze', 'freeze_manifest', 'seed', 'allowed_uuid',
            'fit_count', 'dev_count', 'diagnostic_count', 'train_batch', 'bisection_steps', 'bound_ratio', 'sanity']
    d = {k: c[k] for k in keys}
    d.update(c[branch]); d['output'] = output; d['mode'] = c['mode'] + '-' + branch
    return d


def bn_state_keys(model):
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            keys.update(name + '.' + k for k in module.state_dict())
    return keys


@torch.no_grad()
def check_states(r0, r1):
    """P0/P1 differ only in BN tensors; P0 BN equals the source checkpoint; same names/thresholds/teacher."""
    assert r0.names == r1.names and len(r0.names) == 33
    assert r0.bn['interface_thresholds'] == r1.bn['interface_thresholds']
    assert r0.teacher_digest == r1.teacher_digest
    s0, s1 = r0.model.state_dict(), r1.model.state_dict()
    assert set(s0) == set(s1)
    bn = bn_state_keys(r1.model)
    assert bn == bn_state_keys(r0.model) and len(bn) == 180
    differ = {k for k in s0 if not torch.equal(s0[k], s1[k])}
    assert differ == bn, sorted(differ ^ bn)[:5]
    teacher = r1.teacher.state_dict()
    shared = set(s0) & set(teacher)
    assert bn <= shared and len(shared) == 218
    for k in shared:
        assert torch.equal(s0[k], teacher[k]), 'P0 differs from source checkpoint: ' + k
    for n in r0.names:
        assert not torch.count_nonzero(r0.bn['runtime_biases'][n]) and not torch.count_nonzero(r1.bn['runtime_biases'][n])
    return {'interfaces': 33, 'bn_state_tensors': len(bn), 'differing_tensors': len(differ), 'p0_source_shared_tensors': len(shared),
            'only_bn_differs': True, 'all_bn_differ': True, 'p0_equals_source_on_shared_keys': True, 'thresholds_equal': True,
            'teacher_digest_equal': True, 'zero_biases': True}


def assert_counts_equal(measured, reference, label):
    assert set(measured) == set(reference), label
    for n in measured:
        assert torch.equal(measured[n]['count'], reference[n]['count']), label + ' count differs at ' + n
        assert measured[n]['observations_per_channel'] == reference[n]['observations_per_channel'], label


def solve_all(r, batches, targets, c, label):
    fits = {}
    r.assign(r.bn['runtime_biases'])
    for i, name in enumerate(r.names):
        fits[name] = e7run.fit_one(r, batches, targets, name, c)
        assert torch.equal(fits[name]['target_mean'], targets[name].cpu()), 'solver target differs at ' + name
        r.e0.atomic_torch(r.output / f'fit-progress-{label}.pt', {'layers': fits, 'biases': r.snapshot(), 'completed_interfaces': i + 1})
        r.progress('fitted_layer', arm=label, index=i + 1, name=name,
                   mean_residual=float(fits[name]['normalized_residual'].abs().mean()), seconds=fits[name]['seconds'])
    final = r.snapshot()
    for n in r.names:
        assert torch.equal(final[n][0].flatten(), fits[n]['delta'])
        assert torch.equal(final[n], final[n][:1].expand_as(final[n]))
    return final, fits


def check_replay(measured, fits, label):
    for n, f in fits.items():
        assert torch.equal(measured[n]['count'], f['after_count']), label + ': full-network replay differs at ' + n
        assert measured[n]['observations_per_channel'] == f['observations_per_channel']


@torch.no_grad()
def run(c):
    start = time.perf_counter()
    out = Path(c['output'])
    ref = {k: Path(c['e7_reference']['output']) / v for k, v in c['e7_reference']['files'].items()}
    r1 = prepare(derived(c, 'p1', str(out)))
    r0 = prepare(derived(c, 'p0', str(out / 'P0')))
    r1.e0.atomic_json(out / 'e12-config.json', c)
    states = check_states(r0, r1)
    r1.e0.atomic_json(out / 'state-check.json', states)
    thetas = r1.bn['interface_thresholds']

    fit_batches = list(r1.e0.loader(r1.rows['fit'], r1.parent, c['train_batch']))
    dev_batches = list(r1.e0.loader(r1.rows['development'], r1.parent, c['measure_batch']))
    for batches, rows, size in [(fit_batches, r1.rows['fit'], c['train_batch']), (dev_batches, r1.rows['development'], c['measure_batch'])]:
        assert torch.equal(torch.cat([x[2] for x in batches]), torch.tensor([v['index'] for v in rows]))
        assert torch.equal(torch.cat([x[1] for x in batches]), torch.tensor([v['label'] for v in rows]))
        assert all(len(x[2]) == size for x in batches)

    # B3: teacher targets, bitwise E7 identity.
    a = e7run.teacher_targets(r1, fit_batches)
    e7t = torch.load(ref['targets'], map_location='cpu', weights_only=False)
    assert torch.equal(torch.cat([x[2] for x in fit_batches]), e7t['ids']) and torch.equal(torch.cat([x[1] for x in fit_batches]), e7t['labels'])
    for n in r1.names:
        assert torch.equal(a[n].cpu(), e7t['means'][n]), 'teacher target differs from E7 at ' + n
    r1.e0.atomic_torch(out / 'targets.pt', {'means': {n: x.cpu() for n, x in a.items()}, 'ids': e7t['ids'], 'labels': e7t['labels'],
                                            'teacher_digest': r1.teacher_digest, 'batch': c['train_batch']})
    a_dev = e7run.teacher_targets(r1, dev_batches)
    r1.e0.atomic_torch(out / 'dev-targets.pt', {'means': {n: x.cpu() for n, x in a_dev.items()},
                                                'ids': torch.cat([x[2] for x in dev_batches]), 'batch': c['measure_batch'],
                                                'use': 'B2 dev measurement only; never fitted or selected on'})
    r1.progress('teacher_targets', fit_bitwise_e7=True)

    e7m = torch.load(ref['measurements'], map_location='cpu', weights_only=False)
    fitm, devm, replays = {}, {}, {}

    def measure_both(r, ident, fit_target):
        fitm[ident] = e7run.measure(r, fit_batches, fit_target)
        devm[ident] = e7run.measure(r, dev_batches, a_dev)
        r1.e0.atomic_torch(out / 'measurements.pt', {'fit': fitm, 'development': devm, 'fit_batch': c['train_batch'], 'dev_batch': c['measure_batch']})

    # Strong ordinary controls on P1, bitwise against E7 (same card, batch4 scoring).
    r1.score('bn-only', r1.bn['runtime_biases'], {'state': 'P1 zero bias (E0 BN)'}, diagnostics=False)
    replays['bn-only'] = e7run.verify_prediction_reference(r1, 'bn-only', ref['predictions'], 'bn-only')
    measure_both(r1, 'bn-only', a)
    assert_counts_equal(fitm['bn-only'], e7m['bn-only'], 'P1 zero vs E7 bn-only')

    kl = torch.load(c['kl_bias'], map_location='cpu', weights_only=False)
    assert kl['parent_sha256'] == sha(c['p1']['parent_deployment']) and kl['thresholds'] == thetas
    e7kl = torch.load(ref['bias_kl'], map_location='cpu', weights_only=False)
    assert all(torch.equal(kl['biases'][n], e7kl['biases'][n]) for n in r1.names)
    r1.score('kl-e4', kl['biases'], {'source': c['kl_bias'], 'reference_only': True, 'historically_selected_on_E4_dev': True}, diagnostics=False)
    replays['kl-e4'] = e7run.verify_prediction_reference(r1, 'kl-e4', ref['predictions'], 'kl-e4')
    measure_both(r1, 'kl-e4', a)
    assert_counts_equal(fitm['kl-e4'], e7m['kl-e4'], 'KL vs E7 kl-e4')

    # Source network anchor.
    r0.score('p0-zero', r0.bn['runtime_biases'], {'state': 'P0 zero bias (E0 plain)'}, diagnostics=False)
    measure_both(r0, 'p0-zero', a)

    # Frozen relative target, saved with hashes before any Crel solve.
    p0 = rel.means_from_counts(fitm['p0-zero']); p1 = rel.means_from_counts(fitm['bn-only'])
    a64 = {n: a[n].cpu() for n in r1.names}
    desc, m = rel.descriptors(a64, p0, p1, thetas)
    delta, e, m_check = rel.relative_target(a64, p0, p1)
    assert all(torch.equal(m[n], m_check[n]) for n in r1.names)
    r1.e0.atomic_torch(out / 'relative-target.pt', {'m': m, 'a': a64, 'p0': p0, 'p1': p1, 'delta': delta, 'e': e,
                                                    'formula': 'm = a + (p1 - p0), FP64, unclipped', 'thetas': thetas})
    r1.e0.atomic_json(out / 'descriptors.json', desc)
    manifest = {'relative_target_sha256': sha(out / 'relative-target.pt'), 'descriptors_sha256': sha(out / 'descriptors.json'),
                'written_before_relative_solve': True, 'seconds': time.perf_counter() - start}
    r1.e0.atomic_json(out / 'target-manifest.json', manifest)
    r1.progress('relative_target_frozen', rho_g=desc['global_theta_normalized']['rho'], kappa_g=desc['global_theta_normalized']['kappa'],
                m_outside_fraction=desc['m_outside_0_theta_fraction'])

    fits = {}
    d0, fits['C0'] = solve_all(r0, fit_batches, a, c, 'C0')
    r0.score('C0', d0, {'solve': 'P0 to teacher a', 'bisection_steps': c['bisection_steps'], 'bound_ratio': c['bound_ratio']}, diagnostics=False)
    measure_both(r0, 'C0', a); check_replay(fitm['C0'], fits['C0'], 'C0')

    da, fits['Cabs'] = solve_all(r1, fit_batches, a, c, 'Cabs')
    e7b = torch.load(ref['bias_finite_solve'], map_location='cpu', weights_only=False)
    for n in r1.names:
        assert torch.equal(da[n], e7b['biases'][n]), 'Cabs bias differs from E7 finite-solve at ' + n
    r1.score('Cabs', da, {'solve': 'P1 to teacher a (E7 finite-solve reproduction)'}, diagnostics=False)
    replays['Cabs'] = e7run.verify_prediction_reference(r1, 'Cabs', ref['predictions'], 'finite-solve')
    measure_both(r1, 'Cabs', a); check_replay(fitm['Cabs'], fits['Cabs'], 'Cabs')
    assert_counts_equal(fitm['Cabs'], e7m['finite-solve'], 'Cabs vs E7 finite-solve')

    saved = torch.load(out / 'relative-target.pt', map_location='cpu', weights_only=False)
    assert sha(out / 'relative-target.pt') == manifest['relative_target_sha256'] and all(torch.equal(saved['m'][n], m[n]) for n in r1.names)
    m_gpu = {n: saved['m'][n].cuda() for n in r1.names}
    dr, fits['Crel'] = solve_all(r1, fit_batches, m_gpu, c, 'Crel')
    r1.score('Crel', dr, {'solve': 'P1 to frozen relative target m', 'target_sha256': manifest['relative_target_sha256']}, diagnostics=False)
    measure_both(r1, 'Crel', m_gpu); check_replay(fitm['Crel'], fits['Crel'], 'Crel')

    r1.score('Transfer', d0, {'source': 'C0 bias solved on P0, assigned to P1 without fitting', 'source_file': str(out / 'P0/bias-C0.pt')}, diagnostics=False)
    measure_both(r1, 'Transfer', a)

    r1.e0.atomic_torch(out / 'fits.pt', fits)
    r1.e0.atomic_json(out / 'reference-replays.json', replays)
    out0 = r0.finish()
    result = r1.finish()
    result.update(seconds_total=time.perf_counter() - start, arms=ARM_ORDER, full_fit_replay_equal=True, solved_arms=['C0', 'Cabs', 'Crel'],
                  e7_bitwise={'teacher_targets': True, 'bn-only': True, 'kl-e4': True, 'Cabs_bias': True, 'Cabs_counts': True, 'Cabs_logits': True},
                  p0_runtime=out0, state_check=states, finite_bisection_not_exact=True,
                  note='P0-runtime paired_vs_bn fields are self-referential and unused; acceptance recomputes pairs against P1 bn-only')
    r1.e0.atomic_json(out / 'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    try:
        print(json.dumps(run(c)))
    except Exception:
        Path(c['output']).mkdir(exist_ok=True, parents=True)
        (Path(c['output']) / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}))
        raise
