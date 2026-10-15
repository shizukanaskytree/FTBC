"""E29 CPU check (EXPERIMENT_PLAN.md section 6 item 7): never initializes CUDA. Writes cache/cpu-check.json.
(1) unit checks: the chunked and paired steps against the released step on synthetic networks, the fingerprint, the restores
    (tools/unit_checks.py);
(2) the replay lines of release_chunked.py are the released function's lines, in order, and its capture lines are the released ones up to
    the chunk and the call-time lookup of the hook class;
(3) the out-of-memory signatures of the acceptance equal the E23 runner's;
(4) configs agree with the protocol (sizes, chunks, seed, alpha, outputs, freeze manifest, the E23 inputs, the E23 release-batch hash);
(5) a dry freeze collects every entry and every file exists;
(6) the five CPU dry runs (cpu-dryrun/<tag>, with the out-of-memory and error injections) passed on the current code;
(7) the job driver runs e29_runner and e29_accept with the sanity gate; the acceptance tolerances are the protocol's;
(8) the per-T reading of section 4.2 (follow-up 1 R1: fingerprints as the premise of P1, the per-T trajectory reading) and the model
    reading on synthetic paired logs; the cross-model reading, the chain completeness (N1) and the cell mapping of section 4.3 on synthetic
    acceptances. The same readings on tampered copies of a real run are checked by the r34 dry run (report key r1_tampered)."""
import ast, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
E = Path(__file__).resolve().parents[1]; C1 = E.parent
PY = '/home/wxf/miniconda3/envs/snn/bin/python'
sys.path.insert(0, str(E / 'source')); sys.path.insert(0, str(E))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report, checked = {'checks': {}}, []
def ok(name, cond, detail=None):
    report['checks'][name] = {'passed': bool(cond), **({'detail': detail} if detail is not None else {})}
    assert cond, (name, detail)


# (1)
export = C1 / 'e1/calibration-export'; qcfs = Path('/data3/snn_prjs/ftbc_neural_networks/code/ftbc-eccv/QCFS/ANN_SNN_QCFS')
r = subprocess.run([PY, '-I', '-B', str(E / 'tools/unit_checks.py'), str(export), str(qcfs)], capture_output=True, text=True,
                   env={'CUDA_VISIBLE_DEVICES': '', 'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '16'})
unit = json.loads(r.stdout) if r.returncode == 0 else {'passed': False, 'stderr': r.stderr[-2000:]}
cases = unit.get('cases', []); e2e = [c for c in cases if c.get('check') == 'chunked vs released']; pair = [c for c in cases if c.get('check') == 'paired per call']
ok('unit_checks', r.returncode == 0 and unit['passed'] and len(e2e) == 24 and len(pair) == 8 and all(c['passed'] for c in cases),
   {'cases': len(cases), 'max_float64_diff': max((c['max_abs_diff'] for c in e2e if c['dtype'] == 'torch.float64'), default=None),
    'max_float32_diff': max((c['max_abs_diff'] for c in e2e if c['dtype'] == 'torch.float32'), default=None),
    'paired_max_CHB_vs_REL': {d: max((c['max_CHB_vs_REL_over_theta'] for c in pair if c['dtype'] == d), default=None) for d in ['torch.float64', 'torch.float32']},
    'other_checks': [c['check'] for c in cases if c.get('check') not in ('chunked vs released', 'paired per call')]})
# (2)
def body(src, start, stop):
    lines = [l.strip() for l in src.splitlines()]; i = lines.index(start); j = lines.index(stop, i)
    return [l for l in lines[i:j] if l and not l.startswith('#')]
rel = (export / 'serial.py').read_text(); chk = (E / 'source/release_chunked.py').read_text()
want = body(rel, 'for t in range(module_snn.T):', 'if len(bias.shape) == 4:')
got = body(chk, 'for t in range(module_snn.T):', 'if len(bias.shape) == 4:')
ok('replay_lines_identical', want == got, {'released': want, 'chunked': got})
cap_rel = body(rel, 'ann_get_out = GetLayerInputOutput(ann, module_ann)', 'thre = module_snn.thresh.data')
cap_chk = [l.replace('inherited.GetLayerInputOutput(', 'GetLayerInputOutput(').replace('(part_data)', '(train_data)')
           for l in body(chk, 'ann_get_out = inherited.GetLayerInputOutput(ann, module_ann)', 'v_t_1 = 0.5 * thre')]
ok('capture_lines_identical_up_to_the_chunk', cap_rel == cap_chk, {'released': cap_rel, 'chunked': cap_chk})
# (3)
def assigned(path, name):
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
ok('lib_oom_equal', assigned(E / 'source/e29_accept.py', 'LIB_OOM') == assigned(C1 / 'e23/source/e23_r34_runner.py', 'LIB_OOM'))
# (4)
outs = set()
for model in ['r34', 'vgg']:
    for phase in ['sanity', 'main']:
        c = json.loads((E / f'config/{model}-{phase}.json').read_text()); e23 = json.loads((C1 / f'e23/config/{model}-{phase}.json').read_text())
        cond = c['model'] == model and c['ftbc_alpha'] == 0.4 and c['release_n'] == 3 and c['release_seed'] == 2026101031
        cond &= (c['release_batch'], c['chunk_small']) == ((200, 50) if phase == 'main' else (4, 1))
        cond &= c['output'] == f'/data7/projects/ftbc/runs/c1pre-e29-20261011/{model}-{phase}-v1' and c['freeze_manifest'] == str(E / 'protocol-freeze.json')
        cond &= c['calibration_manifest'] == e23['calibration_manifest'] and c['calibration_manifest_sha256'] == e23['calibration_manifest_sha256'] == sha(c['calibration_manifest'])
        cond &= c['fresh_manifest'] == e23['fresh_manifest'] and c['e23_r34_runner'] == str(C1 / 'e23/source/e23_r34_runner.py')
        cond &= all(c[k] == e23[k] for k in e23 if k not in ('output', 'freeze_manifest', 'release_n', 'release_batch', 'release_seed'))   # every other E23 key unchanged (VGG had no release keys)
        rec23 = json.loads(Path(c['e23_release_record']).read_text()); r34cfg = json.loads((C1 / f'e23/config/r34-{phase}.json').read_text())
        hashes = {v['inputs_sha256'] for k, v in rec23.items() if k.startswith('F0rel')}
        cond &= c['e23_release_record'] == f'/data7/projects/ftbc/runs/c1pre-e23-20261010/r34-{phase}-v1/ftbc.json' and hashes == {c['e23_release_inputs_sha256']}
        cond &= (r34cfg['release_n'], r34cfg['release_batch'], r34cfg['release_seed'], r34cfg['calibration_manifest']) == \
            (c['release_n'], c['release_batch'], c['release_seed'], c['calibration_manifest'])                     # E23 built those batches from the same rows and seed
        cond &= not Path(c['output']).exists(); ok(f'config_{model}_{phase}', cond, {'e23_release_inputs_sha256': c['e23_release_inputs_sha256']}); outs.add(c['output'])
ok('outputs_distinct', len(outs) == 4)
# (5)
import e29_freeze_submit as FZ
with tempfile.TemporaryDirectory() as d:
    files, staged = FZ.freeze(dry_dir=d)
ok('dry_freeze', len(files) > 590 and all(Path(p).exists() for p in files), {'entries': len(files), 'staged': len(staged)})
# (6)
for t in FZ.DRYRUNS:
    p = E / f'cpu-dryrun/{t}/dryrun-report.json'; rep = json.loads(p.read_text()) if p.exists() else {}
    fresh = rep.get('passed') and not rep.get('cuda_initialized') and rep.get('tag') == t and all(sha(q) == h for q, h in rep.get('code_sha256', {}).items())
    fresh &= set(rep.get('code_sha256', {})) >= {str(E / 'source' / f) for f in ['e29_runner.py', 'e29_accept.py', 'release_chunked.py']} | {str(E / 'tools/cpu_dryrun.py')}
    if t == 'r34':                                                    # follow-up 1 R1 on tampered copies of the run
        r1 = rep.get('r1_tampered', {}); nd = {'T2': 'pass', 'T4': 'not decided', 'T8': 'pass'}
        fresh &= sorted(r1) == ['fingerprints-evidence', 'fingerprints-gross', 'trajectory-evidence', 'trajectory-no-evidence']
        fresh &= r1['trajectory-no-evidence']['valid'] is False and all(r1[k]['T_readings'] == nd and r1[k]['gate'] for k in ['trajectory-evidence', 'fingerprints-evidence'])
        fresh &= r1['fingerprints-gross']['T_readings']['T4'] == 'fail' and r1['fingerprints-gross']['gate'] is False
    ok(f'dryrun_{t}', fresh, {'arms': rep.get('arms') or rep.get('recorded_arms'), 'expected_failure': rep.get('expected_failure'),
                              'r1_tampered': rep.get('r1_tampered')})
# (7)
job = (E / 'source/e29_job.py').read_text()
ok('job_driver', "source/e29_runner.py" in job and "source/e29_accept.py'), model, 'sanity'" in job and "a['gate'] and a['valid']" in job
   and "a['scope'] == 'sanity only'" in job and 'e23' not in job.split('"""')[2])
import e29_accept as A
ok('tolerances', A.TOL == {'paired_max_over_theta': 1e-4, 'gross_max_over_theta': 0.1})
# (8)
def synthetic_log(L=3, nb=2, differ=(), evidence=(), chb=1e-8, chs_first=1e-8, chs=1e-3):
    """A PAIR log of L interfaces x nb batches (interface 'i0' first); calls listed in differ get a changed CHB capture, those in evidence a
    num_ooms increase in CHB; chb, chs_first and chs set the per-call maxima (a dict gives per-call values)."""
    log = []
    for b in range(nb):
        for i in range(L):
            k = len(log); fp = [[[[4, 3], 'torch.float32', 1, 2], [[4, 3], 'torch.float32', 3, 4]] for _ in range(2)]
            fpc = json.loads(json.dumps(fp))
            if k in differ:
                fpc[1][0][2] += 1
            val = lambda v: v.get(k, 1e-8) if isinstance(v, dict) else v
            log.append({'interface': f'i{i}', 'batch': b, 'theta': 1.0, 'fingerprints': {'REL': fp, 'CHB': fpc},
                        'num_ooms': {'REL': [0, 0], 'CHB': [0, 1] if k in evidence else [0, 0], 'CHS': [0, 0]},
                        'compare': {'CHB_vs_REL': {'max': val(chb), 'mean': 1e-9}, 'CHS_vs_CHB': {'max': val(chs_first) if i == 0 else 1e-3, 'mean': 1e-9},
                                    'CHS_vs_REL': {'max': val(chs), 'mean': 1e-9}}})
    return log
IFC = {f'i{i}': {'position': i, 'rank': 4 if i < 2 else 2, 'spatial': 49 if i < 2 else 1} for i in range(3)}
cases_t = [('all equal, small differences', dict(), True, 'equal', 'pass'),
           ('fingerprints differ with evidence, CHB 0.01', dict(differ=(1, 4), evidence=(1, 4), chb={1: 0.01, 4: 0.01}), True, 'equal', 'not decided'),
           ('fingerprints differ without evidence', dict(differ=(2,)), True, 'equal', 'not decided'),
           ('fingerprints equal, CHB 2e-4', dict(chb={3: 2e-4}), True, 'equal', 'fail'),
           ('fingerprints differ, CHB 0.2 (gross cap)', dict(differ=(1,), evidence=(1,), chb={1: 0.2}), True, 'equal', 'fail'),
           ('P2: CHS-CHB 2e-4 at the first interface', dict(chs_first={3: 2e-4}), True, 'equal', 'fail'),
           ('P3: CHS-REL 0.2 in main', dict(chs={5: 0.2}), True, 'equal', 'fail'),
           ('P3: CHS-REL 0.2 in sanity (not decided there)', dict(chs={5: 0.2}), False, 'equal', 'pass'),
           ('trajectory differs with evidence', dict(), True, 'not decided', 'not decided')]
for name, kw, main_phase, traj, want in cases_t:
    got = A.paired(synthetic_log(**kw), IFC, 0.4, 200, main_phase, traj)
    ok(f'paired_reading: {name}', got['reading'] == want, {'reading': got['reading'], 'reason': got['reason']})
ok('trajectory_reading', A.trajectory_reading(True, 0, 0) == 'equal' and A.trajectory_reading(False, 1, 0) == 'not decided'
   and A.trajectory_reading(False, 0, 2) == 'not decided' and A.trajectory_reading(False, 0, 0) == 'invalid')
ok('model_reading', A.model_reading(True, {'T2': 'pass', 'T4': 'not decided', 'T8': 'not decided'}) == 'pass'
   and A.model_reading(True, {'T2': 'pass', 'T4': 'fail', 'T8': 'pass'}) == 'fail' and A.model_reading(False, {'T2': 'pass', 'T4': 'pass', 'T8': 'pass'}) == 'invalid'
   and A.model_reading(True, {'T2': 'not decided', 'T4': 'not decided', 'T8': 'not decided'}) == 'not decided')
import verify_preservation as V
def fake(model_reading, rel, chs, valid=True, gate=True, any_fail=False):
    """A model's acceptances; as in the job, main is absent when the sanity gate did not hold."""
    per_T = {T: {'arms': {'REL': {'status': 'completed' if T in rel else 'out_of_memory'}}} for T in ['T2', 'T4', 'T8']}
    main = {'valid': True, 'reading': model_reading, 'per_T': per_T, 'feasible_CHS': {T: T in chs for T in ['T2', 'T4', 'T8']}}
    return {'sanity': {'valid': valid, 'gate': gate and valid, 'any_T_fail': any_fail}, 'main': main if (gate and valid) else None}
def acc(r34, vgg):
    return {f'{m}-{p}': x[p] for m, x in [('r34', r34), ('vgg', vgg)] for p in ['sanity', 'main'] if x[p] is not None}
ALL = ['T2', 'T4', 'T8']
cases8 = [(acc(fake('pass', ['T2', 'T4'], ALL), fake('pass', ALL, ALL)), {'r34': True, 'vgg': True}, {'r34-T8': 'chunked', 'vgg-T2': 'chunked'}),
          (acc(fake('pass', ['T2', 'T4'], ALL), fake('fail', ALL, ALL)), {'r34': False, 'vgg': False}, {'r34-T2': 'original', 'r34-T8': 'infeasible (显存不可行，未测)'}),
          (acc(fake('pass', ['T2', 'T4'], ALL), fake('not decided', [], ALL)), {'r34': True, 'vgg': False}, {'r34-T8': 'chunked', 'vgg-T2': 'infeasible (显存不可行，未测)'}),
          (acc(fake('pass', ['T2', 'T4'], ['T2', 'T4']), fake('pass', ALL, ALL)), {'r34': True, 'vgg': True}, {'r34-T8': 'infeasible (显存不可行，未测)'}),
          (acc(fake('pass', ['T2', 'T4'], ALL), fake('pass', ALL, ALL, gate=False, any_fail=True)), {'r34': False, 'vgg': False}, {'vgg-T2': 'infeasible (显存不可行，未测)', 'r34-T4': 'original'}),
          (acc(fake('pass', ['T2', 'T4'], ALL), fake('pass', ALL, ALL, gate=False, any_fail=False)), {'r34': True, 'vgg': False}, {'vgg-T2': 'infeasible (显存不可行，未测)', 'r34-T4': 'chunked'}),
          (acc(fake('pass', ['T2', 'T4'], ALL), fake('pass', ALL, ALL, valid=False)), {'r34': True, 'vgg': False}, {'vgg-T2': 'infeasible (显存不可行，未测)', 'r34-T2': 'chunked'})]
results8 = []
for a, want_adopt, want_cells in cases8:
    readings, adopted, cells = V.combined_reading(a); results8.append({'readings': readings, 'adopted': adopted})
    ok(f'combined_reading_case_{len(results8)}', adopted == want_adopt and all(cells[k] == v for k, v in want_cells.items()), {'readings': readings, 'cells': cells})
gate_stop = acc(fake('pass', ['T2', 'T4'], ALL), fake('pass', ALL, ALL, gate=False, any_fail=False))      # follow-up 1 N1
ok('chain_complete', V.complete(gate_stop, 'r34') and V.complete(gate_stop, 'vgg') and not V.complete({}, 'r34')
   and not V.complete({'r34-sanity': {'valid': True, 'gate': True, 'any_T_fail': False}}, 'r34'))
import torch
checked = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json')) + [E / 'tools/unit_checks.py', E / 'tools/cpu_dryrun.py', E / 'EXPERIMENT_PLAN.md',
                                                                                   E / 'record_costs.py', E / 'verify_preservation.py', E / 'preservation-before.json']
report.update(passed=all(v['passed'] for v in report['checks'].values()), cuda_initialized=torch.cuda.is_initialized(),
              files_sha256={str(p): sha(p) for p in checked})
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': {k: v['passed'] for k, v in report['checks'].items()}}))
