"""E29 acceptance (EXPERIMENT_PLAN.md sections 4-6): python3 e29_accept.py {r34,vgg} {sanity,main}. Reads the runner outputs on the CPU and
never initializes CUDA. Validity and results are separate (initial review B2):
  valid    every frozen file unchanged; configuration, records and A3 (section 6) hold; a PAIR trajectory that differs from REL without
           allocator evidence makes the run invalid (follow-up 1 R1);
  results  per T a reading pass, fail or not decided from the paired checks P1-P3 (P3 decided in main only; section 4.2), feasibility (CHS
           completed), and the descriptive comparisons (per-call CHS against REL and CHB by interface type with the flip scale, end-to-end CHS
           against REL, agreement);
  gate     (sanity) valid, at least one T reads pass and none reads fail; main runs only if it holds;
  reading  (main) 'pass' (a T passes, none fails), 'fail' (a T fails), 'not decided' or 'invalid' for this model (section 4.3); the
           cross-model reading is verify_preservation.py's.
Writes <model>-<phase>-acceptance.json. Exits nonzero only if the run is invalid or, in sanity, the gate fails."""
import hashlib, json, sys
from pathlib import Path
import torch
E = Path(__file__).resolve().parents[1]
TS, MODES = [2, 4, 8], ['REL', 'PAIR', 'CHS']
TOL = {'paired_max_over_theta': 1e-4, 'gross_max_over_theta': 0.1}
LIB_OOM = ('unable to find an engine', 'CUDNN_STATUS_ALLOC_FAILED', 'CUBLAS_STATUS_ALLOC_FAILED', 'CUDNN_STATUS_INTERNAL_ERROR')   # = the E23 runner's


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def corrections(out, arm):
    return torch.load(out / f'corrections-{arm}.pt', map_location='cpu', weights_only=False)['biases']


def diffs(A, B, theta, ifc):
    """End-to-end |b_A - b_B| / theta over every interface, timestep and channel: max (with interface and timestep) and mean, overall and per
    interface rank (4: conv, 2: vector)."""
    assert sorted(A) == sorted(B) == sorted(theta) == sorted(ifc)
    acc = {}
    for n in A:
        assert len(A[n]) == len(B[n])
        for key in ('all', f'rank{ifc[n]["rank"]}'):
            s = acc.setdefault(key, {'max': 0.0, 'sum': 0.0, 'elements': 0, 'argmax_interface': None, 'argmax_timestep': None})
            for t, (x, y) in enumerate(zip(A[n], B[n])):
                d = (x.double() - y.double()).abs() / theta[n]
                s['sum'] += float(d.sum()); s['elements'] += d.numel()
                if float(d.max()) > s['max']:
                    s.update(max=float(d.max()), argmax_interface=n, argmax_timestep=t)
    return {k: {'max_abs_over_theta': v['max'], 'mean_abs_over_theta': v['sum'] / v['elements'], 'argmax_interface': v['argmax_interface'],
                'argmax_timestep': v['argmax_timestep'], 'elements': v['elements']} for k, v in acc.items()}


def agreement(out, a, b):
    A = torch.load(out / f'agree-{a}.pt', map_location='cpu', weights_only=False); B = torch.load(out / f'agree-{b}.pt', map_location='cpu', weights_only=False)
    assert torch.equal(A['ids'], B['ids']) and torch.equal(A['labels'], B['labels'])
    ca, cb = int((A['predictions'] == A['labels']).sum()), int((B['predictions'] == B['labels']).sum())
    return {'n': int(A['ids'].numel()), 'agreement': float((A['predictions'] == B['predictions']).double().mean()), 'correct': {a: ca, b: cb}, 'correct_diff': ca - cb}


def oom_ok(v):
    if v['oom_basis'] == 'exception':
        return v['exception_type'] == 'OutOfMemoryError' or 'out of memory' in v['exception_message'].lower()
    return v['oom_basis'] == 'library' and any(s.lower() in v['exception_message'].lower() for s in LIB_OOM) and \
        (v['num_ooms_increase'] > 0 or (v['device_free_bytes'] is not None and v['device_free_bytes'] < 2 ** 30))


def pair_log_ok(log, theta, nb):
    """Every interface has one record per batch, with two fingerprinted captures (ANN call, SNN call) for REL and CHB, the comparisons
    CHB_vs_REL, CHS_vs_REL and CHS_vs_CHB, and the allocator's num_ooms before and after each implementation."""
    seen = {}
    for r in log:
        seen.setdefault(r['interface'], []).append(r['batch'])
        if not (sorted(r['fingerprints']) == ['CHB', 'REL'] and all(len(r['fingerprints'][k]) == 2 for k in ('CHB', 'REL'))
                and sorted(r['compare']) == ['CHB_vs_REL', 'CHS_vs_CHB', 'CHS_vs_REL'] and r['theta'] == theta[r['interface']]
                and sorted(r['num_ooms']) == ['CHB', 'CHS', 'REL'] and all(len(v) == 2 and v[1] >= v[0] for v in r['num_ooms'].values())):
            return False
    return sorted(seen) == sorted(theta) and all(v == list(range(nb)) for v in seen.values())


def trajectory_reading(equal, rel_ooms_increase, pair_ooms_increase):
    """Follow-up 1 R1, per T. PAIR keeps the released result of every call, so its final corrections equal those of the REL arm. When they
    differ and the allocator recorded an out-of-memory event in either arm at that T (cuDNN then retries another convolution plan, which
    changes the numerics), the pairing at that T is not decided; without that evidence the cause is unknown and the run is invalid. Never a
    fail."""
    if equal:
        return 'equal'
    return 'not decided' if (rel_ooms_increase > 0 or pair_ooms_increase > 0) else 'invalid'


def paired(log, ifc, alpha, B, main, trajectory='equal'):
    """Section 4.2 (follow-up 1 R1). The premise of P1 on a call is that the released hook's captures of REL and CHB are bitwise equal
    (fingerprints). A call fails P1 if its fingerprints are equal and max |b_CHB - b_REL| / theta > 1e-4, or if they differ and the difference
    exceeds the gross cap 0.1. P2 fails on a call at the first interface (no IF upstream) with max |b_CHS - b_CHB| / theta > 1e-4 (both
    accumulate in float64). P3 (main only) fails on a call with max |b_CHS - b_REL| / theta > 0.1. The T reads 'fail' if any of these fails;
    otherwise 'not decided' if some call's fingerprints differ (the forward was not reproducible) or the trajectory reading is 'not decided';
    otherwise 'pass'. Descriptive: per interface rank, the CHS_vs_REL and CHS_vs_CHB maxima and means, the single-flip change
    alpha / (B * spatial) and the maxima in its units; the calls whose fingerprints differ and which of them carry allocator evidence."""
    first = min(ifc, key=lambda n: ifc[n]['position'])
    fp_equal = [r['fingerprints']['REL'] == r['fingerprints']['CHB'] for r in log]
    chb = [r['compare']['CHB_vs_REL']['max'] for r in log]
    p1_fail_calls = [i for i, (e, x) in enumerate(zip(fp_equal, chb)) if x > (TOL['paired_max_over_theta'] if e else TOL['gross_max_over_theta'])]
    p2_max = max(r['compare']['CHS_vs_CHB']['max'] for r in log if r['interface'] == first)
    p3_max = max(r['compare']['CHS_vs_REL']['max'] for r in log)
    fails = {'P1': bool(p1_fail_calls), 'P2': p2_max > TOL['paired_max_over_theta'], 'P3': main and p3_max > TOL['gross_max_over_theta']}
    differ = [i for i, e in enumerate(fp_equal) if not e]
    evidence = [i for i in differ if any(v[1] > v[0] for v in log[i]['num_ooms'].values())]
    if any(fails.values()):
        reading, reason = 'fail', 'failed: ' + ', '.join(k for k, v in fails.items() if v)
    elif differ:
        reading, reason = 'not decided', f'forward not reproducible: fingerprints differ on {len(differ)} calls ({len(evidence)} with allocator evidence)'
    elif trajectory == 'not decided':
        reading, reason = 'not decided', 'the PAIR and REL trajectories differ, with allocator evidence'
    else:
        reading, reason = 'pass', 'P1 and P2' + (' and P3' if main else '') + ' hold on every call; fingerprints equal on every call'
    desc = {}
    for key in ('CHS_vs_REL', 'CHS_vs_CHB'):
        for r in log:
            i = ifc[r['interface']]; unit = alpha / (B * i['spatial'])
            s = desc.setdefault(key, {}).setdefault(f'rank{i["rank"]}', {'calls': 0, 'max': 0.0, 'mean_sum': 0.0, 'flip_unit_min': unit,
                                                                           'flip_unit_max': unit, 'max_in_flip_units': 0.0})
            s['calls'] += 1; s['max'] = max(s['max'], r['compare'][key]['max']); s['mean_sum'] += r['compare'][key]['mean']
            s['flip_unit_min'] = min(s['flip_unit_min'], unit); s['flip_unit_max'] = max(s['flip_unit_max'], unit)
            s['max_in_flip_units'] = max(s['max_in_flip_units'], r['compare'][key]['max'] / unit)
    for v in desc.values():
        for s in v.values():
            s['mean_of_call_means'] = s.pop('mean_sum') / s['calls']
    return {'decided': reading != 'not decided', 'reading': reading, 'reason': reason, 'calls': len(log), 'first_interface': first,
            'P1_failed_calls': len(p1_fail_calls), 'P1_max_CHB_vs_REL_fingerprints_equal': max([x for e, x in zip(fp_equal, chb) if e], default=None),
            'P1_max_CHB_vs_REL_fingerprints_differ': max([x for e, x in zip(fp_equal, chb) if not e], default=None),
            'fingerprints_differ_calls': len(differ), 'fingerprints_differ_with_allocator_evidence': len(evidence),
            'differing_calls': [{'interface': log[i]['interface'], 'batch': log[i]['batch'], 'num_ooms': log[i]['num_ooms']} for i in differ][:50],
            'P2_failed': fails['P2'], 'P2_max_CHS_vs_CHB_first_interface': p2_max,
            'P3_failed': fails['P3'] if main else None, 'P3_decided': main, 'P3_max_CHS_vs_REL': p3_max, 'trajectory': trajectory,
            'per_call_by_rank': desc}


def model_reading(valid, readings):
    """Section 4.3: 'invalid', 'fail' (some T fails), 'pass' (some T passes and none fails) or 'not decided'."""
    if not valid:
        return 'invalid'
    return 'fail' if 'fail' in readings.values() else 'pass' if 'pass' in readings.values() else 'not decided'


def check(model, phase):
    c = json.loads((E / f'config/{model}-{phase}.json').read_text()); main = phase == 'main'
    frozen = json.loads(Path(c['freeze_manifest']).read_text())['files_sha256']
    for p, h in frozen.items():
        assert sha(p) == h, 'frozen file changed: ' + p
    B, small, nb, alpha = c['release_batch'], c['chunk_small'], c['release_n'], c['ftbc_alpha']
    assert c['model'] == model and alpha == 0.4 and nb == 3
    assert (B, small) == ((200, 50) if main else (4, 1)), 'section 3 sizes'
    out = Path(c['output']); res = json.loads((out / 'result.json').read_text()); rec = json.loads((out / 'e29.json').read_text())
    assert res['status'] == 'completed' and res['model'] == model and res['all_strict_reloads_passed'] and res['teacher_unchanged'] and res['non_bias_unchanged']
    assert res['allocator_conf'] == 'expandable_segments:True'
    cal = json.loads(Path(c['calibration_manifest']).read_text()); rows = cal['C1'][:nb * B]
    assert rec['release_rows'] == [int(x['index']) for x in rows] and sorted(rec['order']) == sorted(rec['release_rows']), 'release rows (section 6 item 3)'
    assert rec['chunks'] == {'REL': None, 'PAIR': {'CHB': B, 'CHS': small}, 'CHS': small}
    metrics = json.loads((out / 'metrics.json').read_text())
    k = -(-B // small); per_call = {'REL': 1, 'CHS': k, 'PAIR': 2 + k}
    valid = rec['inputs_sha256'] == c['e23_release_inputs_sha256']    # B3: the E23 release-default batches (same pool, seed, transform)
    report = {'model': model, 'phase': phase, 'inputs_sha256': rec['inputs_sha256'], 'inputs_equal_e23_release_default': valid,
              'tolerances': TOL, 'per_T': {}}
    zero = {}
    for T in TS:
        theta, ifc = rec['thresholds'][f'T{T}'], rec['interfaces'][f'T{T}']; L = len(theta)
        arms = {m: rec['arms'][f'{m}-T{T}'] for m in MODES}; row = {'arms': {}}
        assert sorted(ifc) == sorted(theta)
        for m, v in arms.items():                                         # validity (A3)
            arm = f'{m}-T{T}'
            if v['status'] == 'completed':
                ok = v['inputs_sha256_after'] == rec['inputs_sha256'] and v['batches'] == nb and v['batch_size'] == B
                ok &= sorted(v['updates']) == sorted(theta) and all(x == nb for x in v['updates'].values())
                ok &= v['forwards']['ann'] == v['forwards']['snn'] == L * nb * per_call[m] and v['forwards']['per_call'] == per_call[m]
                ok &= arm in metrics and metrics[arm]['strict_reload']
                ok &= (out / f'corrections-{arm}.pt').exists() and (out / f'agree-{arm}.pt').exists()
                zero.setdefault(T, v['zero_sha256']); ok &= v['zero_sha256'] == zero[T]
                if m == 'PAIR':
                    log = json.loads((out / f'pair-T{T}.json').read_text())
                    ok &= len(log) == L * nb == v['paired_calls'] and pair_log_ok(log, theta, nb) and arms['REL']['status'] == 'completed'
                    row['trajectory'] = trajectory_reading(v['corrections_sha256'] == arms['REL']['corrections_sha256'],   # follow-up 1 R1
                                                           arms['REL']['num_ooms_increase'], v['num_ooms_increase'])
                    ok &= row['trajectory'] != 'invalid'
                row['arms'][m] = {'status': 'completed', 'seconds': v['seconds'], 'peak_allocated_bytes': v['peak_allocated_bytes'],
                                  'peak_reserved_bytes': v['peak_reserved_bytes'], 'baseline_allocated_bytes': v['baseline_allocated_bytes'],
                                  'baseline_reserved_bytes': v['baseline_reserved_bytes'], 'num_ooms_increase': v['num_ooms_increase'],
                                  'num_alloc_retries_increase': v['num_alloc_retries_increase'], 'forwards': v['forwards'],
                                  'max_abs_bias_over_theta': v['max_abs_bias_over_theta'], 'checks': ok}
            elif v['status'] == 'out_of_memory':
                ok = oom_ok(v) and v['stage'] == 'update' and v['export_restored']
                row['arms'][m] = {'status': 'out_of_memory', 'oom_basis': v['oom_basis'], 'peak_allocated_bytes': v['peak_allocated_bytes'],
                                  'peak_reserved_bytes': v['peak_reserved_bytes'], 'card_total_bytes': v['card_total_bytes'],
                                  'exception_message': v['exception_message'], 'leftover_hooks_dropped': v['leftover_hooks_dropped'],
                                  'paired_calls_completed': v['paired_calls_completed'], 'checks': ok}
            else:
                ok = m == 'PAIR' and v['status'] == 'not_run' and arms['REL']['status'] != 'completed'
                row['arms'][m] = {'status': v['status'], 'checks': ok}
            valid &= ok
        if arms['PAIR']['status'] == 'completed' and row['arms']['PAIR']['checks']:   # results (section 4)
            row['paired'] = paired(json.loads((out / f'pair-T{T}.json').read_text()), ifc, alpha, B, main, row['trajectory'])
        else:
            row['paired'] = {'decided': False, 'reading': 'not decided', 'reason': f'PAIR {arms["PAIR"]["status"]} (REL {arms["REL"]["status"]})'
                             + ('' if arms['PAIR']['status'] != 'completed' else '; its records failed the validity checks')}
        row['feasible_CHS'] = arms['CHS']['status'] == 'completed'
        if arms['REL']['status'] == 'completed' and arms['CHS']['status'] == 'completed':   # descriptive: floating-point sensitivity of the recipe
            row['end_to_end_CHS_vs_REL'] = diffs(corrections(out, f'CHS-T{T}'), corrections(out, f'REL-T{T}'), theta, ifc)
            row['agreement_CHS_vs_REL'] = agreement(out, f'CHS-T{T}', f'REL-T{T}')
        if arms['PAIR']['status'] == 'completed':
            row['agreement_PAIR_vs_REL'] = agreement(out, f'PAIR-T{T}', f'REL-T{T}')
        report['per_T'][f'T{T}'] = row
    readings = {f'T{T}': report['per_T'][f'T{T}']['paired']['reading'] for T in TS}
    any_fail, passed_T = 'fail' in readings.values(), [T for T in TS if readings[f'T{T}'] == 'pass']
    report.update(valid=bool(valid), T_readings=readings, any_T_fail=any_fail, pass_T=passed_T,
                  feasible_CHS={f'T{T}': report['per_T'][f'T{T}']['feasible_CHS'] for T in TS}, frozen_files=len(frozen),
                  cuda_initialized=torch.cuda.is_initialized(), scope=f'{phase} only')
    if main:
        report['reading'] = model_reading(bool(valid), readings)
    else:
        report['gate'] = bool(valid and passed_T and not any_fail)
    (E / f'{model}-{phase}-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ['model', 'phase', 'valid', 'gate', 'reading', 'T_readings', 'feasible_CHS']}))
    assert not torch.cuda.is_initialized()
    if not valid or (not main and not report['gate']):
        raise SystemExit(f'E29 {model} {phase}: ' + ('the run is invalid' if not valid else 'the sanity gate failed'))


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] in ('r34', 'vgg') and sys.argv[2] in ('sanity', 'main'), sys.argv
    check(sys.argv[1], sys.argv[2])
