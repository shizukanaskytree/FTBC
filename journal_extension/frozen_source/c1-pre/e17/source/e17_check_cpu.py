"""E17 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e17_runner, e17_accept, e17_freeze_submit


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'e14_runner', 'e15_runner', 'e16_runner', 'ce_solver', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_artifacts',
          'e5_roles', 'e0_predictions', 'calibration_export', 'calibration_export_manifest', 'freeze_manifest', 'fresh_manifest', 'fresh_manifest_sha256', 'calibration_manifest',
          'calibration_manifest_sha256', 'previous_fresh_manifests', 'previous_calibration_manifests', 'allowed_uuids', 'seed', 'solver_batch', 'bisection_steps', 'bound_ratio',
          'ftbc_alpha', 'ftbc_arms', 'ftbc_ns', 'score_roles', 'fit_count']
prep = json.loads((E / 'input/preparation.json').read_text())
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert set(c['allowed_uuids']) == {'GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894'}
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32 and c['ftbc_arms'] == [['F0-n3', 3, 0.4]]
    assert c['small_n'] == c['fit_small'] == 3 * c['ftbc_batch'] < c['draw_n'] and 'stages' not in c
assert (cfg['main']['draw_n'], cfg['main']['small_n'], cfg['main']['ftbc_batch'], cfg['main']['fresh_count'], cfg['main']['dev_count']) == (256, 96, 32, 2048, 512)
assert cfg['sanity']['sanity'] and not cfg['main']['sanity'] and cfg['sanity']['output'] != cfg['main']['output']
report['configs_checked'] = True
# 2. Inputs: identity, provenance, disjointness from every earlier role, split and calibration set.
c = cfg['main']
assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] == prep['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256'] == prep['calibration_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e16prep = json.loads((P / 'e16/input/preparation.json').read_text())
assert prep['excluded_ids'] == e16prep['excluded_ids'] + 2048 + 3 * 256 + 3 * 600 and prep['excluded_content_hashes'] == e16prep['excluded_content_hashes'] + 2048 + 3 * 256 + 3 * 600
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']; cal = json.loads(Path(c['calibration_manifest']).read_text())
assert len(fresh) == 2048 and sorted(cal) == ['C1', 'C2', 'C3'] and all(len(v) == 256 for v in cal.values())
e0 = json.loads((P / 'input-roles.json').read_text()); e5 = json.loads(Path(c['e5_roles']).read_text())
old = [r for roles in [e0, e5] for v in roles.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
new = fresh + [r for v in cal.values() for r in v]
assert len({int(r['index']) for r in new}) == len(new) and len({r['sha256'] for r in new}) == len(new)
assert not {int(r['index']) for r in new} & {int(r['index']) for r in old} and not {r['sha256'] for r in new} & {r['sha256'] for r in old}
report['inputs'] = {'fresh': len(fresh), 'calibration_sets': {k: len(v) for k, v in cal.items()}, 'disjoint_from_all_earlier': True, 'fresh_distinct_labels': len({r['label'] for r in fresh})}
# 3. Deployment state and teacher identity.
art = json.loads(Path(c['e0_artifacts']).read_text())
for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt')]:
    assert Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
spec = importlib.util.spec_from_file_location('e17_check_e0', c['e0_runner']); e0m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0m)
teacher, classes = e0m.source_model(json.loads(Path(c['parent_config']).read_text()))
pp = torch.load(c['plain_proxy'], map_location='cpu', weights_only=True)
fd = e17_accept.state_dict_digest(pp['state_dict']); assert e0m.state_digest(teacher) == fd
report['states'] = {'e0_artifact_hashes_match': True, 'teacher_equals_plain_proxy_parameters': True, 'plain_proxy_digest': fd}
del teacher
# 4. Reused frozen code and inheritance.
e16_freeze = json.loads((P / 'e16/protocol-freeze.json').read_text())['files_sha256']
for f in ['e16/source/e16_runner.py', 'e16/source/e16_accept.py', 'e15/source/e15_runner.py', 'e14/source/e14_runner.py', 'e14/source/step_solver.py', 'e14/source/e14_accept.py',
          'e13/source/e13_runner.py', 'e13/source/serial_solver.py']:
    p = str(P / f); assert e16_freeze[p] == sha(p), 'frozen code changed: ' + p
E13, E14, E15, E16, ss, ps, ce, Run = e17_runner.build(c)
own = {'__init__', 'reference_output', 'step_calibrate_ce'}; assert own == set(Run.__dict__) & (own | {'net', 'use_rows_target', 'ftbc_batches_ann'}), set(Run.__dict__)
Run16 = Run.__mro__[1]; Run15 = Run.__mro__[2]; Run14 = Run.__mro__[3]
for m in ['net', 'use_rows_target', 'proxy_means', 'levels', 'ftbc_batches_ann', 'clear_release_hooks']: assert getattr(Run, m) is Run16.__dict__[m], m
for m in ['registered_batches']: assert getattr(Run, m) is Run15.__dict__[m], m
for m in ['capture', 'replay_check', 'step_calibrate', 'count_calibrate_fit', 'score', 'score_ann']: assert getattr(Run, m) is Run14.__dict__[m], m
for m in ['count_calibrate', 'teacher_means']: assert getattr(Run, m) is getattr(E13.Run, m), m
assert ce.__file__ == c['ce_solver']
report['reuse'] = {'own': sorted(own), 'from_E16': 6, 'from_E15': 1, 'from_E14': 6, 'from_E13': 2}
# 5. CE solver CPU checks are current.
chk = json.loads((E / 'cache/check-ce-solver.json').read_text()); assert chk['passed'] and chk['tie_heavy_near_ties'] == 0 and len(chk['cases']) == 12
assert chk['ce_solver_sha256'] == sha(E / 'source/ce_solver.py') and chk['int64_path_equals_float64_reference'], 'CE solver checks are stale or failed'
report['ce_solver_checks'] = {'cases': len(chk['cases']), 'tie_heavy_cases': sum(c2['tie_heavy'] for c2 in chk['cases']), 'passed': True, 'ce_solver_sha256': chk['ce_solver_sha256'],
                              'int64_equals_float64_comparisons': chk['cut_comparisons']}
# 6. Decision logic.
B, W_, N_ = 'candidate better', 'candidate worse', 'no difference detected'
def mk(q1, q2, q3=N_, q4=N_, q5a=N_, q5b=N_, net2=(0, 0, 0)):
    st = lambda rs: e17_accept.status(rs)
    dec = lambda rs, nets=(0, 0, 0): {'per_draw': {d: {'reading': r, 'net': n} for d, r, n in zip(['C1', 'C2', 'C3'], rs, nets)}, 'status': st(rs)}
    return e17_accept.adoption({'decisions': {'Q1': dec(q1), 'Q2': dec(q2, net2), 'Q3': dec([q3] * 3), 'Q4': dec([q4] * 3), 'Q5-T2': dec([q5a] * 3), 'Q5-T4': dec([q5b] * 3)}})
a = mk([B, B, B], [N_, N_, N_], net2=(5, -3, 0)); assert a['adopt_CE_A_as_serial_solver'] and a['claim_CE_A_not_worse_than_P0_at_T4'] and not a['claim_CE_A_better_than_P0_at_T4_replicated']
a = mk([B, B, B], [N_, N_, N_], net2=(-25, -20, -30)); assert not a['adopt_CE_A_as_serial_solver'] and not a['claim_CE_A_not_worse_than_P0_at_T4'] and a['Q2_net_sum'] == -75   # B2
a = mk([B, B, B], [N_, N_, N_], net2=(-10, 4, 6)); assert a['adopt_CE_A_as_serial_solver'] and a['Q2_net_sum'] == 0
a = mk([B, B, B], [N_, W_, B]); assert not a['adopt_CE_A_as_serial_solver'] and a['claim_CE_A_better_than_M0_at_T2_replicated'] and not a['claim_CE_A_not_worse_than_P0_at_T4']
a = mk([B, N_, B], [B, B, B]); assert not a['adopt_CE_A_as_serial_solver'] and a['claim_CE_A_better_than_P0_at_T4_replicated']
a = mk([B, B, B], [B, B, B], B, B, B, B); assert a['adopt_CE_A_as_serial_solver'] and a['claim_CE_better_than_M0_at_T2_replicated'] and a['claim_CE_A_better_than_registered_FTBC_same_images_T4_replicated']
assert e17_accept.decide is e17_accept.A14.decide and e17_accept.status is e17_accept.A16.status
report['decision_logic_checked'] = True
# 7. Dry freeze.
with tempfile.TemporaryDirectory() as tmp:
    frozen, staged = e17_freeze_submit.freeze(dry_dir=tmp)
    w = Path(c['workspace'])
    assert all((w / k).resolve().is_relative_to(w) for k in staged) and str(P / 'e16/protocol-freeze.json') in frozen and c['ce_solver'] in frozen
    assert all(p in frozen for p in c['previous_fresh_manifests'] + c['previous_calibration_manifests'] + [c['e0_artifacts'], c['e5_roles']])
report['dry_freeze'] = {'entries_without_review_files': len(frozen), 'staged': len(staged)}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e17_runner.py', E / 'source/e17_accept.py', E / 'source/ce_solver.py', E / 'source/check_ce_solver.py', E / 'source/prepare_inputs.py',
         E / 'source/e17_freeze_submit.py', E / 'config/sanity.json', E / 'config/main.json', E / 'EXPERIMENT_PLAN.md', E / 'input/preparation.json', E / 'record_costs.py',
         E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=1, default=str))
