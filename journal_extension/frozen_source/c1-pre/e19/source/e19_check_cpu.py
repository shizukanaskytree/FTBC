"""E19 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e19_runner, e19_accept, e19_freeze_submit, e19_job


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'e14_runner', 'e15_runner', 'e16_runner', 'e17_runner', 'ce_solver', 'ftbc_variants', 'parent_config', 'plain_deployment',
          'plain_proxy', 'e0_artifacts', 'e5_roles', 'e0_predictions', 'calibration_export', 'calibration_export_manifest', 'freeze_manifest', 'fresh_manifest',
          'fresh_manifest_sha256', 'calibration_manifest', 'calibration_manifest_sha256', 'previous_fresh_manifests', 'previous_calibration_manifests', 'allowed_uuids', 'seed',
          'solver_batch', 'bisection_steps', 'bound_ratio', 'ftbc_alpha', 'ftbc_arms', 'ftbc_ns', 'score_roles', 'fit_count', 'eccv_alpha', 'eccv_batch']
prep = json.loads((E / 'input/preparation.json').read_text())
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert c['allowed_uuids'] == ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'] and c['output'].endswith('-v2')   # attempt v2 on the H100 (GPU-0)
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32 and c['ftbc_arms'] == [['F0-n3', 3, 0.4]]
    assert c['small_n'] == c['fit_small'] == 3 * c['ftbc_batch'] < c['draw_n'] and c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2
    assert c['eccv_n'] == c['eccv_batch'] * c['eccv_batches'] < c['draw_n'] and c['diag_n'] <= c['fresh_count']
assert (cfg['main']['draw_n'], cfg['main']['small_n'], cfg['main']['ftbc_batch'], cfg['main']['fresh_count'], cfg['main']['dev_count']) == (256, 96, 32, 2048, 512)
assert (cfg['main']['eccv_batches'], cfg['main']['eccv_n'], cfg['main']['diag_n']) == (50, 100, 256)
assert cfg['sanity']['sanity'] and not cfg['main']['sanity'] and cfg['sanity']['output'] != cfg['main']['output']
assert e19_runner.REGISTERED == [('F0', 'released', 1), ('FA', 'FA', 1), ('FD', 'FD', 1), ('FD10', 'FD', 10)]
rc = (E / 'record_costs.py').read_text(); assert "['run_id']" in rc and 'c1pre-e19-job-v' not in rc, 'record_costs must take the run ID from the receipt'
report['configs_checked'] = True
# 2. Inputs: identity, provenance, disjointness from every earlier role, split and calibration set.
c = cfg['main']
assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] == prep['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256'] == prep['calibration_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e18prep = json.loads((P / 'e18/input/preparation.json').read_text())
assert prep['excluded_ids'] == e18prep['excluded_ids'] + 2048 + 3 * 256 and prep['excluded_content_hashes'] == e18prep['excluded_content_hashes'] + 2048 + 3 * 256
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']; cal = json.loads(Path(c['calibration_manifest']).read_text())
assert len(fresh) == 2048 and sorted(cal) == ['C1', 'C2', 'C3'] and all(len(v) == 256 for v in cal.values())
e0 = json.loads((P / 'input-roles.json').read_text()); e5 = json.loads(Path(c['e5_roles']).read_text())
old = [r for roles in [e0, e5] for v in roles.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
new = fresh + [r for v in cal.values() for r in v]
assert len({int(r['index']) for r in new}) == len(new) and len({r['sha256'] for r in new}) == len(new)
assert not {int(r['index']) for r in new} & {int(r['index']) for r in old} and not {r['sha256'] for r in new} & {r['sha256'] for r in old}
assert [Path(p).parent.parent.name for p in c['previous_fresh_manifests']] == ['e14', 'e15', 'e16', 'e17', 'e18']
assert [Path(p).parent.parent.name for p in c['previous_calibration_manifests']] == ['e16', 'e17', 'e18']
report['inputs'] = {'fresh': len(fresh), 'calibration_sets': {k: len(v) for k, v in cal.items()}, 'disjoint_from_all_earlier': True, 'fresh_distinct_labels': len({r['label'] for r in fresh})}
# 3. Deployment state and teacher identity.
art = json.loads(Path(c['e0_artifacts']).read_text())
for key, name in [('plain_deployment', 'deployment-plain.pt'), ('plain_proxy', 'proxy-plain.pt')]:
    assert Path(c[key]).parent == Path(c['e0_artifacts']).parent and sha(c[key]) == art[name]['sha256'], key
spec = importlib.util.spec_from_file_location('e19_check_e0', c['e0_runner']); e0m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0m)
teacher, classes = e0m.source_model(json.loads(Path(c['parent_config']).read_text()))
pp = torch.load(c['plain_proxy'], map_location='cpu', weights_only=True)
fd = e19_accept.state_dict_digest(pp['state_dict']); assert e0m.state_digest(teacher) == fd
report['states'] = {'e0_artifact_hashes_match': True, 'teacher_equals_plain_proxy_parameters': True, 'plain_proxy_digest': fd}
del teacher
# 4. Reused frozen code and inheritance.
e17_freeze = json.loads((P / 'e17/protocol-freeze.json').read_text())['files_sha256']
for f in ['e17/source/e17_runner.py', 'e17/source/ce_solver.py', 'e16/source/e16_runner.py', 'e16/source/e16_accept.py', 'e15/source/e15_runner.py', 'e14/source/e14_runner.py',
          'e14/source/step_solver.py', 'e14/source/e14_accept.py', 'e13/source/e13_runner.py', 'e13/source/serial_solver.py', 'e1/calibration-export/serial.py']:
    p = str(P / f); assert e17_freeze[p] == sha(p), 'frozen code changed: ' + p
E13, E14, E15, E16, ss, ps, ce, V, Run = e19_runner.build(c)
own = {'__init__', 'ftbc_batches', 'ftbc_run', 'deployed_residuals'}; assert set(Run.__dict__) & (own | {'net', 'use_rows_target', 'ftbc_batches_ann', 'step_calibrate', 'score'}) == own
Run17, Run16, Run15, Run14 = Run.__mro__[1:5]
for m in ['net', 'use_rows_target', 'ftbc_batches_ann', 'clear_release_hooks']: assert getattr(Run, m) is Run16.__dict__[m], m
for m in ['registered_batches']: assert getattr(Run, m) is Run15.__dict__[m], m
for m in ['capture', 'replay_check', 'step_calibrate', 'score', 'score_ann']: assert getattr(Run, m) is Run14.__dict__[m], m
for m in ['teacher_means']: assert getattr(Run, m) is getattr(E13.Run, m), m
assert V.__file__ == c['ftbc_variants']
report['reuse'] = {'own': sorted(own), 'from_E16': 4, 'from_E15': 1, 'from_E14': 5, 'from_E13': 1}
# 5. Variant CPU checks are current.
chk = json.loads((E / 'cache/check-variants.json').read_text()); assert chk['passed'] and not chk['cuda_initialized']
for p, h in chk['files_sha256'].items():
    assert sha(p) == h, 'variant checks are stale for ' + p
cases = [v for k, v in chk.items() if isinstance(v, dict) and 'fa_first_call_equals_released' in v]
keys = ['fa_first_call_equals_released', 'fa_nonzero_step0_update_from_deployed_spikes', 'fd_trajectory_equals_deployed_IF',
        'fd_second_call_updates_from_mixed_deployed_spikes', 'fd_second_call_trajectory_equals_deployed_IF']
assert len(cases) == 8 and all(all(v[k] for k in keys) for v in cases)
report['variant_checks'] = {'cases': len(cases), 'fd_repeated_calls': chk['fd_repeated_calls_mean_abs_step_residual']}
# 6. Decision logic.
B, W_, N_ = 'candidate better', 'candidate worse', 'no difference detected'
Qs = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']; assert list(e19_accept.QUESTIONS) == Qs and e19_accept.PRIMARY == Qs[:3]
def mk(readings, fe_below=(False, False, False)):
    dec = {k: {'per_draw': {d: {'reading': r} for d, r in zip(['C1', 'C2', 'C3'], rs)}, 'status': e19_accept.status(rs)} for k, rs in zip(Qs, readings)}
    return e19_accept.claims({'decisions': dec, 'fe_below_s0_by_draw': dict(zip(['C1', 'C2', 'C3'], fe_below))})
a = mk([[B] * 3] * 6); assert a['claim_three_part_decomposition_all_replicated'] and a['claim_exact_solve_helps_over_ten_alpha_passes_replicated'] and not a['reading_D4_all_draws_no_difference']
a = mk([[B] * 3, [N_] * 3, [B] * 3, [N_] * 3, [B] * 3, [B] * 3], (False, True, False))      # one draw below S0 is enough (review B1(d))
assert a['claim_including_own_correction_in_replay_helps_replicated'] and not a['claim_deployed_advancement_helps_over_residual_injection_replicated']
assert not a['claim_three_part_decomposition_all_replicated'] and a['reading_D4_all_draws_no_difference'] and a['FE_below_S0_any_draw']
assert a['reading_D6_ECCV_schedule_overcorrects'] and not a['claim_P0_better_than_ECCV_schedule_FTBC_R34_T4_replicated'] and a['D6'] == 'replicated'
assert mk([[B] * 3] * 6)['claim_P0_better_than_ECCV_schedule_FTBC_R34_T4_replicated'] and not mk([[B] * 3] * 6)['reading_D6_ECCV_schedule_overcorrects']
a = mk([[B, B, W_], [B] * 3, [B] * 3, [B, N_, B], [B] * 3, [N_] * 3]); assert a['D1'] == 'not replicated' and a['D4'] == 'partially replicated' and not a['reading_D4_all_draws_no_difference']
assert e19_accept.decide is e19_accept.A14.decide and e19_accept.status is e19_accept.A16.status
arms = e19_accept.expected_arms(); assert len(arms) == len(set(arms)) == 5 + 3 * 9
assert sorted(e19_accept.FTBC_ARMS) == ['F0-T4', 'F0-T8', 'FA-T4', 'FD-T4', 'FD10-T4', 'FE-T4']
report['decision_logic_checked'] = True
# 7. Dry freeze.
with tempfile.TemporaryDirectory() as tmp:
    frozen, staged = e19_freeze_submit.freeze(dry_dir=tmp)
    w = Path(c['workspace'])
    assert all((w / k).resolve().is_relative_to(w) for k in staged) and str(P / 'e17/protocol-freeze.json') in frozen and c['ftbc_variants'] in frozen
    assert all(p in frozen for p in c['previous_fresh_manifests'] + c['previous_calibration_manifests'] + [c['e0_artifacts'], c['e5_roles'], c['e17_runner']])
report['dry_freeze'] = {'entries_without_review_files': len(frozen), 'staged': len(staged)}
# 8. Job payload and submission command (static; nothing is submitted), then the in-allocation gate on stand-in stages.
st = e19_job.stages()
assert [n for n, _ in st] == ['sanity', 'sanity-acceptance', 'main'] and all(cmd[:2] == ['/home/wxf/miniconda3/envs/snn/bin/python', '-B'] for _, cmd in st)
assert st[0][1][2:] == [str(E / 'source/e19_runner.py'), '--config', str(E / 'config/sanity.json')] and st[1][1][2:] == [str(E / 'source/e19_accept.py'), 'sanity']
assert st[2][1][2:] == [str(E / 'source/e19_runner.py'), '--config', str(E / 'config/main.json')]
cmd = e19_freeze_submit.command()
assert cmd[:5] == ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', 'c1pre-e19-job-v2'] and cmd[cmd.index('--time') + 1] == '07:00:00'
assert cmd[cmd.index('--gpu') + 1] == 'h100' and cmd[cmd.index('--gpus') + 1] == '1' and [cmd[i + 1] for i, x in enumerate(cmd) if x == '--allowed-uuid'] == c['allowed_uuids']
assert cmd[cmd.index('--source-manifest') + 1] == str(E / 'slurm-source-manifest.json') and cmd[-3:] == ['/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e19_job.py')]
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp); log = t / 'log'; saved = (e19_job.E, e19_job.stages)
    tag = lambda name: [sys.executable, '-c', f"open({str(log)!r}, 'a').write({name!r} + chr(10))"]
    def acc(passed):
        return [sys.executable, '-c', f"import json; open({str(t / 'sanity-acceptance.json')!r}, 'w').write(json.dumps(dict(passed={passed}, cuda_initialized=False, scope='sanity only')))"]
    def trial(first, accept):
        for f in [log, t / 'sanity-acceptance.json']:
            f.unlink(missing_ok=True)
        e19_job.stages = lambda: [('sanity', first), ('sanity-acceptance', accept), ('main', tag('main'))]
        try:
            e19_job.main(); outcome = 'completed'
        except (AssertionError, SystemExit) as err:
            outcome = type(err).__name__
        return outcome, log.read_text().split() if log.exists() else []
    try:
        e19_job.E = t
        assert trial(tag('sanity'), acc(True)) == ('completed', ['sanity', 'main'])
        assert trial(tag('sanity'), acc(False)) == ('AssertionError', ['sanity'])
        assert trial([sys.executable, '-c', 'raise SystemExit(3)'], acc(True)) == ('SystemExit', [])
        (t / 'sanity-acceptance.json').write_text('{}')
        try: e19_job.main()
        except AssertionError: pass
        else: raise AssertionError('a stale sanity acceptance was not refused')
    finally:
        e19_job.E, e19_job.stages = saved
report['job'] = {'stages': [n for n, _ in st], 'gpurun_command': cmd, 'gate_trials': ['pass -> main', 'failed acceptance -> no main', 'failed sanity -> stop', 'stale acceptance refused']}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e19_runner.py', E / 'source/e19_accept.py', E / 'source/ftbc_variants.py', E / 'source/check_variants.py', E / 'source/prepare_inputs.py',
         E / 'source/e19_freeze_submit.py', E / 'source/e19_job.py', E / 'config/sanity.json', E / 'config/main.json', E / 'EXPERIMENT_PLAN.md', E / 'input/preparation.json',
         E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=1, default=str))
