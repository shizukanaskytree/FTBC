"""E16 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e16_runner, e16_accept, e16_freeze_submit


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main', 'release']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'e14_runner', 'e15_runner', 'parent_config', 'plain_deployment', 'plain_proxy', 'spatial_deployment', 'spatial_proxy',
          'e0_artifacts', 'e5_roles',
          'e0_predictions', 'calibration_export', 'calibration_export_manifest', 'freeze_manifest', 'fresh_manifest', 'fresh_manifest_sha256', 'calibration_manifest',
          'calibration_manifest_sha256', 'previous_fresh_manifests', 'allowed_uuids', 'seed', 'solver_batch', 'bisection_steps', 'bound_ratio', 'ftbc_alpha', 'ftbc_arms',
          'ftbc_ns', 'score_roles', 'release_n', 'release_seeds', 'expandable_segments', 'fit_count']
prep = json.loads((E / 'input/preparation.json').read_text())
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert set(c['allowed_uuids']) == {'GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894'}
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32 and c['release_n'] == 3 and c['ftbc_arms'] == [['F0-n3', 3, 0.4]]
    assert c['small_n'] == c['fit_small'] == 3 * c['ftbc_batch'] < c['draw_n']
    assert sorted(c['release_seeds']) == ['Q1', 'Q2', 'Q3'] and len(set(c['release_seeds'].values())) == 3 and prep['seed'] not in c['release_seeds'].values()
assert cfg['main']['stages'] == ['main'] and cfg['release']['stages'] == ['release'] and cfg['sanity']['stages'] == ['main', 'release']
assert all((cfg[k]['draw_n'], cfg[k]['small_n'], cfg[k]['release_batch'], cfg[k]['fresh_count'], cfg[k]['dev_count']) == (256, 96, 200, 2048, 512) for k in ['main', 'release'])
assert len({c['output'] for c in cfg.values()}) == 3
report['configs_checked'] = True
# 2. Inputs: identity, provenance, disjointness, outside the E0 fit pool.
c = cfg['main']
assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] == prep['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256'] == prep['calibration_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e15prep = json.loads((P / 'e15/input/preparation.json').read_text())
assert prep['excluded_ids'] == e15prep['excluded_ids'] + 2048 and prep['excluded_content_hashes'] == e15prep['excluded_content_hashes'] + 2048
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']; cal = json.loads(Path(c['calibration_manifest']).read_text())
assert len(fresh) == 2048 and [len(cal[k]) for k in ['C1', 'C2', 'C3', 'Q1', 'Q2', 'Q3']] == [256, 256, 256, 600, 600, 600]
e0 = json.loads((P / 'input-roles.json').read_text()); e5 = json.loads(Path(c['e5_roles']).read_text()); assert c['e5_roles'] in prep['exclusion_sources']
old = [r for roles in [e0, e5] for v in roles.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
new = fresh + [r for v in cal.values() for r in v]
assert len({int(r['index']) for r in new}) == len(new) and len({r['sha256'] for r in new}) == len(new)
assert not {int(r['index']) for r in new} & {int(r['index']) for r in old} and not {r['sha256'] for r in new} & {r['sha256'] for r in old}
assert not {int(r['index']) for r in new} & {int(r['index']) for r in e0['fit']}     # the 4096 images that trained the E0 spatial BN state
report['inputs'] = {'fresh': len(fresh), 'calibration_sets': {k: len(v) for k, v in cal.items()}, 'disjoint_from_E0_E5_E14_E15': True, 'outside_E0_fit_pool': True}
# 3. Deployment states.
plain = torch.load(c['plain_deployment'], map_location='cpu', weights_only=False); spatial = torch.load(c['spatial_deployment'], map_location='cpu', weights_only=False)
pp = torch.load(c['plain_proxy'], map_location='cpu', weights_only=True); sp = torch.load(c['spatial_proxy'], map_location='cpu', weights_only=True)
assert all(not torch.count_nonzero(x) for v in spatial['runtime_biases'].values() for x in v) and sp['proxy_quantization_level'] == 4
diff = [k for k in plain['state_dict'] if not torch.equal(plain['state_dict'][k], spatial['state_dict'][k])]
assert len(diff) == 180 and all(any(s in k for s in ['.weight', '.bias', 'running_mean', 'running_var', 'num_batches_tracked']) for k in diff)
shared_keys = set(spatial['state_dict']) & set(sp['state_dict'])
assert all(torch.equal(spatial['state_dict'][k], sp['state_dict'][k]) for k in shared_keys), 'spatial deployment and proxy disagree'
art = json.loads(Path(c['e0_artifacts']).read_text())
for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt'), ('spatial_deployment', 'deployment-spatial.pt'), ('spatial_proxy', 'proxy-spatial.pt')]:
    assert Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
# The Q8 teacher has exactly the plain proxy parameters (so the teacher target is A8(phi0)); digests by the run_probe.state_digest rule.
spec = importlib.util.spec_from_file_location('e16_check_e0', c['e0_runner']); e0m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0m)
teacher, classes = e0m.source_model(json.loads(Path(c['parent_config']).read_text()))
fd = {'plain': e16_accept.state_dict_digest(pp['state_dict']), 'spatial': e16_accept.state_dict_digest(sp['state_dict'])}
assert e0m.state_digest(teacher) == fd['plain'] != fd['spatial'] and sorted({m.L for m in teacher.modules() if isinstance(m, classes['activation'])}) == [8]
report['states'] = {'bn_keys_differing_plain_vs_spatial': len(diff), 'spatial_deployment_equals_proxy_on_shared_keys': len(shared_keys),
                    'e0_artifact_hashes_match': True, 'teacher_equals_plain_proxy_parameters': True, 'proxy_file_digests': fd}
del teacher
# 4. Reused frozen code and inheritance.
e15_freeze = json.loads((P / 'e15/protocol-freeze.json').read_text())['files_sha256']
for f in ['e15/source/e15_runner.py', 'e15/source/e15_accept.py', 'e14/source/e14_runner.py', 'e14/source/step_solver.py', 'e14/source/e14_accept.py',
          'e13/source/e13_runner.py', 'e13/source/serial_solver.py']:
    p = str(P / f); assert e15_freeze[p] == sha(p), 'frozen code changed: ' + p
E13, E14, E15, ss, ps, Run = e16_runner.build(c)
own = {'__init__', 'net', 'grid_proxy', 'levels', 'proxy_means', 'use_rows_target', 'ftbc_batches_ann', 'clear_release_hooks'}; assert own == set(Run.__dict__) & (own | {'twin_means'}), set(Run.__dict__)
Run15 = Run.__mro__[1]; Run14 = Run15.__mro__[1]
for m in ['ftbc_batches', 'registered_batches', 'release_batches']: assert getattr(Run, m) is Run15.__dict__[m], m
for m in ['capture', 'replay_check', 'step_calibrate', 'count_calibrate_fit', 'score', 'score_ann']: assert getattr(Run, m) is Run14.__dict__[m], m
for m in ['count_calibrate', 'ftbc_calibrate', 'teacher_means']: assert getattr(Run, m) is getattr(E13.Run, m), m
report['reuse'] = {'own': sorted(own), 'from_E15': 3, 'from_E14': 6, 'from_E13': 3}
# 5. Decision logic.
B, W_, N_ = 'candidate better', 'candidate worse', 'no difference detected'
assert e16_accept.status([B, B, B]) == 'replicated' and e16_accept.status([B, N_, B]) == 'partially replicated' and e16_accept.status([W_, B, B]) == 'not replicated'
def mk(q1, q2, q2b, q4, q3, q3b, lf=0):
    return e16_accept.combine({'decisions': {'Q1': {'status': q1}, 'Q2': {'status': q2}, 'Q2b': {'status': q2b}, 'Q4': {'status': q4}}, 'label_free_twin_target_better_draws': lf},
                              {'decisions': {'Q3': {'pairs_pass': q3}, 'Q3b': {'pairs_pass': q3b}}})
R, PR, NR = 'replicated', 'partially replicated', 'not replicated'
k = lambda x: f'claim_two_stage_better_than_BN_then_{x}'
assert mk(R, R, R, R, True, True)[k('release_default_FTBC_both_references_allowed')] and mk(R, R, R, R, True, True)[k('registered_FTBC_both_references_replicated')]
assert mk(R, R, PR, R, True, True)[k('release_default_FTBC_same_target_allowed')] and not mk(R, R, PR, R, True, True)[k('release_default_FTBC_both_references_allowed')]
assert not mk(R, R, R, R, True, False)[k('release_default_FTBC_both_references_allowed')] and mk(R, R, R, R, True, False)[k('release_default_FTBC_same_target_allowed')]
assert not mk(R, PR, R, R, True, True)[k('registered_FTBC_same_target_replicated')] and not mk(R, PR, R, R, True, True)[k('release_default_FTBC_same_target_allowed')]
assert not mk(NR, R, R, R, True, True)[k('registered_FTBC_same_target_replicated')] and mk(R, NR, NR, R, False, False)['claim_two_stage_better_than_P0_alone_replicated']
assert mk(R, R, R, PR, True, True)['claim_twin_target_better_than_teacher_target_on_BN_state_replicated'] is False and e16_accept.combine(None, None)['Q1'] == 'absent'
assert mk(R, R, R, R, True, True, lf=1)['Q1_wording_must_disclose_label_free_target_gain'] and not mk(R, R, R, R, True, True)['Q1_wording_must_disclose_label_free_target_gain']
t = lambda x: f'claim_pipeline_better_than_BN_then_{x}_teacher_reference'
assert mk(R, NR, R, R, False, True)[t('registered_FTBC') + '_replicated'] and mk(R, NR, R, R, False, True)[t('release_default_FTBC') + '_allowed']
assert not mk(R, NR, R, R, False, True)[k('registered_FTBC_same_target_replicated')] and not mk(R, R, PR, R, True, True)[t('registered_FTBC') + '_replicated']
assert not mk(PR, R, R, R, True, True)[t('registered_FTBC') + '_replicated'] and not mk(R, R, R, R, True, False)[t('release_default_FTBC') + '_allowed']
assert e16_accept.decide is e16_accept.A14.decide
report['decision_logic_checked'] = True
# 6. Dry freeze.
with tempfile.TemporaryDirectory() as tmp:
    frozen, staged = e16_freeze_submit.freeze(dry_dir=tmp)
    w = Path(c['workspace'])
    assert all((w / k).resolve().is_relative_to(w) for k in staged) and not any('ssd-runs' in k for k in staged) and any('ssd-runs' in p for p in frozen)
    assert str(P / 'e15/protocol-freeze.json') in frozen and all(c[x] in frozen for x in ['spatial_deployment', 'spatial_proxy', 'e0_artifacts', 'e5_roles'])
report['dry_freeze'] = {'entries_without_review_files': len(frozen), 'staged': len(staged)}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e16_runner.py', E / 'source/e16_accept.py', E / 'source/prepare_inputs.py', E / 'source/e16_freeze_submit.py',
         E / 'config/sanity.json', E / 'config/main.json', E / 'config/release.json', E / 'EXPERIMENT_PLAN.md', E / 'input/preparation.json',
         E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps(report, indent=1, default=str))
