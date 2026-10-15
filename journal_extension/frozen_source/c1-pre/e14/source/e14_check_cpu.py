"""E14 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_runner, e14_accept


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. The per-step solver check artifact is current (bitwise IF agreement, residuals, monotonicity).
step = json.loads((E / 'cache/step-solver-check.json').read_text())
assert step['passed'] and not step['cuda_initialized']
for p, h in step['files_sha256'].items():
    assert sha(p) == h, 'step-solver check is stale for ' + p
report['step_solver_check'] = {k: step[k] for k in ['cases', 'per_step_residual_within_quantum_or_flagged', 'per_step_monotonicity_checks',
                                                    'later_single_step_increase_events_when_raising_b_t', 'cumulative_count_increase_events_when_raising_b_t']}
# 2. Pre-registered statistics on hand-checkable cases.
assert e14_accept.upper_tail(0, 0) == 1.0 and e14_accept.upper_tail(3, 3) == 1 / 8 and abs(e14_accept.upper_tail(2, 3) - 0.5) < 1e-15
assert abs(e14_accept.upper_tail(60, 100) - 0.028443966820490392) < 1e-12           # P(Bin(100,1/2) >= 60)
lab = torch.tensor([0, 1, 2, 3, 4, 5]); x = torch.tensor([0, 1, 2, 9, 9, 5]); y = torch.tensor([0, 9, 9, 3, 9, 9])
pr = e14_accept.paired(lab, x, y)
assert (pr['x_only_correct'], pr['y_only_correct'], pr['net'], pr['prediction_disagreements']) == (3, 1, 2, 4)   # index 4: both predict 9
assert abs(pr['one_sided_p_x_better'] - 5 / 16) < 1e-15 and abs(pr['one_sided_p_y_better'] - 15 / 16) < 1e-15
strong = {'net': 30, 'one_sided_p_x_better': 0.01, 'one_sided_p_y_better': 0.99}
assert e14_accept.decide(strong, 1.0, 1.1, 5)['reading'] == 'candidate better'
assert e14_accept.decide(strong, 1.2, 1.1, 5)['reading'] == 'no difference detected'       # CE disagrees
assert e14_accept.decide(strong, 1.0, 1.1, -2)['reading'] == 'no difference detected'      # batch-8 sign disagrees
assert e14_accept.decide({'net': -30, 'one_sided_p_x_better': 0.99, 'one_sided_p_y_better': 0.01}, 1.0, 1.1)['reading'] == 'candidate worse'
report['statistics_checked'] = True
# 3. Bias hash identical in runner and acceptance; float32 bound comparison.
g = torch.Generator().manual_seed(14); names = ['a', 'b']; biases = {n: [torch.randn(5, generator=g) for _ in range(4)] for n in names}
assert e14_runner.bias_hash(biases, names, 4) == e14_accept.bias_hash(biases, names, 4)
th = 1.2345678; edge = torch.full((3,), -0.5 * th, dtype=torch.float32)
assert e14_accept.within(edge, 0.5 * th) and not e14_accept.within(edge * 1.001, 0.5 * th)
report['bias_hash_and_bound_checked'] = True
# 4. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main', 'extra']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'e13_output', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_predictions', 'calibration_export',
          'calibration_export_manifest', 'freeze_manifest', 'fresh_manifest', 'fresh_manifest_sha256', 'allowed_uuids', 'seed', 'solver_batch', 'bisection_steps',
          'bound_ratio', 'ftbc_alpha', 'ftbc_arms', 'ftbc_ns', 'f1_n', 'perturb_batch', 'score_roles']
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert set(c['allowed_uuids']) == {'GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894'}
    assert c['fit_small'] == 3 * c['ftbc_batch'] <= c['fit_count'] and c['ftbc_ns'] == [3, 6] and c['f1_n'] == 3
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32
    arms = {a: (n, al) for a, n, al in c['ftbc_arms']}; assert c['ftbc_arms'][0] == ['F0-n3', 3, 0.4] and len(arms) == 4
    assert arms['F0-g1'][0] == 3 and abs(arms['F0-g1'][1] * 3 - 1) < 1e-15 and arms['F0-n6-g1.2'] == (6, 0.2) and arms['F0-n3-g2.4'] == (3, 0.8)
    assert max(n for n, _ in arms.values()) == max(c['ftbc_ns'])      # E13 sizes rows['ftbc'] from max(ftbc_ns)
assert cfg['main']['stages'] == ['main'] and cfg['extra']['stages'] == ['extra'] and cfg['sanity']['stages'] == ['main', 'extra']
assert (cfg['main']['fit_count'], cfg['main']['fit_small'], cfg['main']['fresh_count'], cfg['main']['dev_count'], cfg['main']['diag_count']) == (256, 96, 2048, 512, 64)
assert cfg['main']['e13_anchor'] and not cfg['extra']['e13_anchor'] and not cfg['sanity']['e13_anchor'] and cfg['sanity']['sanity']
assert len({c['output'] for c in cfg.values()}) == 3
report['configs_checked'] = True
# 5. Fresh development: identity, disjointness from E0 and E5 roles, provenance.
c = cfg['main']; assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']
prep = json.loads((E / 'input/preparation.json').read_text()); assert prep['passed'] and prep['new_manifest_sha256'] == c['fresh_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e0 = json.loads((P / 'input-roles.json').read_text()); e5 = json.loads((P / 'e5/input-roles.json').read_text())
old_ids = {int(r['index']) for roles in [e0, e5] for v in roles.values() for r in v}; old_sha = {r['sha256'] for roles in [e0, e5] for v in roles.values() for r in v}
ids = [int(r['index']) for r in fresh]
assert len(fresh) == 2048 and len(set(ids)) == 2048 and not set(ids) & old_ids and not {r['sha256'] for r in fresh} & old_sha
assert all(Path(r['path']).is_absolute() and Path(r['path']).is_relative_to(c['workspace']) and 0 <= int(r['label']) < 1000 for r in fresh)
audit = json.loads((E / 'input/exclusion-audit.json').read_text()); assert audit['passed'] and audit['script_sha256'] == sha(E / 'source/audit_exclusions.py')
assert audit['union_excluded_IDs'] == prep['excluded_ids'] and audit['union_excluded_hashes'] == prep['excluded_content_hashes']
assert all(v['new_development_ID_overlap'] == 0 and v['new_development_content_overlap'] == 0 for v in audit['sources'].values())
report['fresh_development'] = {'count': len(fresh), 'distinct_labels': prep['distinct_labels'], 'disjoint_from_E0_E5_ids_and_hashes': True, 'images_read': False,
                               'exclusion_audit_sources': len(audit['sources'])}
# 6. Reused frozen E13 code and the E13 artifacts the anchors read.
e13_freeze = json.loads((P / 'e13/protocol-freeze.json').read_text())['files_sha256']
for f in ['source/e13_runner.py', 'source/serial_solver.py']:
    p = str(P / 'e13' / f); assert e13_freeze[p] == sha(p), 'E13 frozen code changed: ' + p
assert c['e13_runner'] == str(P / 'e13/source/e13_runner.py')
d = Path(c['e13_output']); assert json.loads((d / 'result.json').read_text())['status'] == 'completed'
for f in ['predictions.pt', 'bias-M0.pt', 'bias-F0-n3.pt', 'teacher-targets.pt', 'hardware.json']:
    assert (d / f).is_file(), f
E13, ss, Run = e14_runner.build(c)
own = {'capture', 'replay_check', 'step_calibrate', 'count_calibrate_fit', 'ftbc_with_alpha', 'score', 'score_perturbed', 'score_ann', 'teacher_means_on',
       'step_objective', 'e13_anchors', 'use_fit', 'free', '__init__'}
assert own <= set(Run.__dict__) and issubclass(Run, E13.Run)
for m in ['count_calibrate', 'ftbc_calibrate', 'diagnostics', 'teacher_means', 'anchors', 'set_biases', 'zero_biases', 'net', 'layer_channels']:
    assert m not in Run.__dict__ and getattr(Run, m) is getattr(E13.Run, m), m
report['e13_reuse'] = {'frozen_runner_and_solver_unchanged': True, 'inherited_unchanged': ['count_calibrate', 'ftbc_calibrate', 'diagnostics', 'teacher_means', 'anchors'],
                       'overridden_or_new': sorted(own)}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/step_solver.py', E / 'source/check_step_solver.py', E / 'source/e14_runner.py', E / 'source/e14_accept.py',
         E / 'source/prepare_development.py', E / 'source/audit_exclusions.py', E / 'source/e14_freeze_submit.py', E / 'input/exclusion-audit.json',
         E / 'config/sanity.json', E / 'config/main.json', E / 'config/extra.json', E / 'EXPERIMENT_PLAN.md']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps(report, indent=1, default=str))
