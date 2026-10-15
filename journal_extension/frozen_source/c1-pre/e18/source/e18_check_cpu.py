"""E18 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import copy, hashlib, importlib, json, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(8)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e18_runner, e18_accept, e18_freeze_submit, e18_job


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'step_solver', 'serial_solver', 'calibration_export', 'calibration_export_manifest', 'source_snapshot', 'source_checkpoint',
          'source_checkpoint_sha256', 'source_level', 'interfaces', 'dataset_manifest', 'dataset_manifest_sha256', 'e5_roles', 'fresh_manifest', 'fresh_manifest_sha256',
          'calibration_manifest', 'calibration_manifest_sha256', 'previous_fresh_manifests', 'previous_calibration_manifests', 'allowed_uuids', 'seed', 'num_workers',
          'eval_batch_size', 'solver_batch', 'bisection_steps', 'bound_ratio', 'ftbc_alpha', 'eccv_alpha', 'eccv_batch', 'score_roles', 'freeze_manifest']
prep = json.loads((E / 'input/preparation.json').read_text())
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert c['allowed_uuids'] == ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211']          # H100 (GPU-0), allowed by the author on 2026-10-10
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32 and c['source_level'] == 16 and c['interfaces'] == 15
    assert c['small_n'] == 3 * c['ftbc_batch'] < c['draw_n']
    assert c['eccv_alpha'] == 0.5 and c['eccv_batch'] == 2 and c['eccv_n'] == c['eccv_batch'] * c['eccv_batches'] < c['draw_n']   # ECCV supplement schedule
assert (cfg['main']['draw_n'], cfg['main']['small_n'], cfg['main']['ftbc_batch'], cfg['main']['fresh_count'], cfg['main']['dev_count']) == (256, 96, 32, 2048, 512)
assert (cfg['main']['eccv_batches'], cfg['main']['eccv_n'], cfg['sanity']['eccv_batches'], cfg['sanity']['eccv_n']) == (50, 100, 4, 8)
assert cfg['sanity']['sanity'] and not cfg['main']['sanity'] and cfg['sanity']['output'] != cfg['main']['output']
report['configs_checked'] = True
# 2. Inputs.
c = cfg['main']
assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256'] == prep['fresh_manifest_sha256'] and sha(c['calibration_manifest']) == c['calibration_manifest_sha256'] == prep['calibration_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e17prep = json.loads((P / 'e17/input/preparation.json').read_text())
assert prep['excluded_ids'] == e17prep['excluded_ids'] + 2048 + 3 * 256 and prep['excluded_content_hashes'] == e17prep['excluded_content_hashes'] + 2048 + 3 * 256
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']; cal = json.loads(Path(c['calibration_manifest']).read_text())
assert len(fresh) == 2048 and sorted(cal) == ['C1', 'C2', 'C3'] and all(len(v) == 256 for v in cal.values())
roles = json.loads(Path(c['dataset_manifest']).read_text()); e5 = json.loads(Path(c['e5_roles']).read_text())
old = [r for rr in [roles, e5] for v in rr.values() for r in v] + [r for p in c['previous_fresh_manifests'] for r in json.loads(Path(p).read_text())['development']]
old += [r for p in c['previous_calibration_manifests'] for v in json.loads(Path(p).read_text()).values() for r in v]
new = fresh + [r for v in cal.values() for r in v]
assert len({int(r['index']) for r in new}) == len(new) and len({r['sha256'] for r in new}) == len(new)
assert not {int(r['index']) for r in new} & {int(r['index']) for r in old} and not {r['sha256'] for r in new} & {r['sha256'] for r in old}
assert sha(c['dataset_manifest']) == c['dataset_manifest_sha256']
report['inputs'] = {'fresh': len(fresh), 'calibration_sets': {k: len(v) for k, v in cal.items()}, 'disjoint_from_all_earlier': True, 'fresh_distinct_labels': len({r['label'] for r in fresh})}
# 3. Source model identity and construction (CPU).
assert sha(c['source_checkpoint']) == c['source_checkpoint_sha256']
sys.path.insert(0, c['source_snapshot']); layer = importlib.import_module('Models.layer'); VGG = importlib.import_module('Models.VGG')
renamed = e18_runner.renamed_source_state(c['source_checkpoint'])
src = VGG.vgg16(1000, 0.0); src.set_L(16); src.load_state_dict(renamed, strict=True); src.eval()
ifs = [(n, m) for n, m in src.named_modules() if isinstance(m, layer.IF)]
assert len(ifs) == 15 and [n for n, _ in ifs] == [k[:-len('.thresh')] for k in renamed if k.endswith('.thresh')] and all(m.L == 16 for _, m in ifs)
sdig = e18_accept.state_dict_digest(renamed)
spec = importlib.util.spec_from_file_location('e18_check_e0', c['e0_runner']); e0m = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0m)
assert e0m.state_digest(src) == sdig
for T in (4, 8):                                       # twin and SNN keep the source parameters; L and T are not in the state_dict
    proxy = copy.deepcopy(src); proxy.set_L(T); snn = copy.deepcopy(proxy); snn.set_T(T)
    assert e0m.state_digest(proxy) == sdig == e0m.state_digest(snn)
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        assert tuple(snn(x).shape) == (T, 1, 1000) and tuple(proxy(x).shape) == (1, 1000)
exp = json.loads(Path(c['calibration_export_manifest']).read_text())['families']['serial']
assert sha(c['calibration_export']) == exp['export_sha256'] and exp['definition_ast_identity'] and not exp['protected_entrypoint_exported']
report['source'] = {'checkpoint_sha256': c['source_checkpoint_sha256'], 'state_digest': sdig, 'interfaces': [n for n, _ in ifs], 'thresholds': [round(float(m.thresh), 4) for _, m in ifs]}
del src, proxy, snn
# 4. Frozen reused code.
e17_freeze = json.loads((P / 'e17/protocol-freeze.json').read_text())['files_sha256']
for k in ['e0_runner', 'e13_runner', 'step_solver', 'serial_solver', 'calibration_export']:
    assert e17_freeze[c[k]] == sha(c[k]), 'frozen code changed: ' + c[k]
# 5. Decision logic.
B, W_, N_ = 'candidate better', 'candidate worse', 'no difference detected'
Qs = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']; assert list(e18_accept.QUESTIONS) == Qs and e18_accept.SECOND_MODEL == Qs[:4]
def mk(st, below4=(False, False, False), below8=(False, False, False), worse4=(False, False, False), worse8=(False, False, False)):
    z = lambda a, b: {'T4': dict(zip(['C1', 'C2', 'C3'], a)), 'T8': dict(zip(['C1', 'C2', 'C3'], b))}
    return e18_accept.claims({'decisions': {k: {'status': s} for k, s in zip(Qs, st)}, 'fe_below_s0_by_draw': z(below4, below8), 'fe_below_f0_by_draw': z(worse4, worse8)})
R, PR, NR = 'replicated', 'partially replicated', 'not replicated'
comb = 'claim_P0_advantage_carries_to_second_model_at_T4_and_T8'
assert mk([R] * 6)[comb] and not mk([R, R, R, PR, R, R])[comb] and mk([R, R, R, R, NR, NR])[comb]     # Q5/Q6 stay outside the combined statement
assert mk([R, NR, PR, R, R, R])['claim_P0_better_than_M0_VGG16_T4_replicated'] and not mk([R, NR, PR, R, R, R])['claim_P0_better_than_M0_VGG16_T8_replicated']
assert mk([PR, PR, PR, R, PR, PR])['claim_P0_better_than_registered_FTBC_VGG16_T8_replicated'] and e18_accept.decide is e18_accept.A14.decide and e18_accept.status is e18_accept.A16.status
assert mk([NR] * 4 + [R, PR])['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T4_replicated'] and not mk([NR] * 4 + [R, PR])['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T8_replicated']
a = mk([R] * 6, below4=(False, True, False))                   # one draw below S0 at T=4 is enough for the overcorrection reading
assert a['reading_Q5_ECCV_schedule_overcorrects'] and not a['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T4_replicated'] and a['Q5'] == R
assert a['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T8_replicated'] and not a['reading_Q6_ECCV_schedule_overcorrects'] and a[comb]
a = mk([R] * 6, worse8=(True, False, False))                    # FE above S0 but below the same-draw F0 (review B2): no Q6 method statement
assert a['reading_Q6_ECCV_schedule_worse_than_registered'] and not a['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T8_replicated'] and not a['reading_Q6_ECCV_schedule_overcorrects']
assert a['claim_P0_better_than_ECCV_schedule_FTBC_VGG16_T4_replicated'] and a['claim_P0_better_than_registered_FTBC_VGG16_T8_replicated']
arms = e18_accept.expected_arms(); assert len(arms) == len(set(arms)) == 41 and sum(a.startswith(('FE', 'F0')) for a in arms) == 12
assert e18_accept.status([B, B, B]) == R and e18_accept.status([B, N_, B]) == PR and e18_accept.status([W_, B, B]) == NR
report['decision_logic_checked'] = True
# 6. Dry freeze.
with tempfile.TemporaryDirectory() as tmp:
    frozen, staged = e18_freeze_submit.freeze(dry_dir=tmp)
    w = Path(c['workspace'])
    assert all((w / k).resolve().is_relative_to(w) for k in staged) and str(P / 'e17/protocol-freeze.json') in frozen and c['source_checkpoint'] in frozen
    assert all(p in frozen for p in c['previous_fresh_manifests'] + c['previous_calibration_manifests'] + [c['e5_roles'], c['dataset_manifest']])
    assert all(str(Path(c['source_snapshot']) / 'Models' / f) in frozen for f in ['layer.py', 'VGG.py', '__init__.py'])
report['dry_freeze'] = {'entries_without_review_files': len(frozen), 'staged': len(staged)}
# 7. Job payload and submission command (static; nothing is submitted), then the in-allocation gate on stand-in stages.
st = e18_job.stages()
assert [n for n, _ in st] == ['sanity', 'sanity-acceptance', 'main'] and all(cmd[:2] == ['/home/wxf/miniconda3/envs/snn/bin/python', '-B'] for _, cmd in st)
assert st[0][1][2:] == [str(E / 'source/e18_runner.py'), '--config', str(E / 'config/sanity.json')] and st[1][1][2:] == [str(E / 'source/e18_accept.py'), 'sanity']
assert st[2][1][2:] == [str(E / 'source/e18_runner.py'), '--config', str(E / 'config/main.json')]
cmd = e18_freeze_submit.command()
assert cmd[:5] == ['gpurun', '--project', 'ftbc-neural-networks', '--run-id', 'c1pre-e18-job-v1'] and cmd[cmd.index('--time') + 1] == '04:00:00'
assert cmd[cmd.index('--gpu') + 1] == 'h100' and cmd[cmd.index('--gpus') + 1] == '1' and [cmd[i + 1] for i, x in enumerate(cmd) if x == '--allowed-uuid'] == c['allowed_uuids']
assert cmd[cmd.index('--source-manifest') + 1] == str(E / 'slurm-source-manifest.json') and cmd[-3:] == ['/home/wxf/miniconda3/envs/snn/bin/python', '-B', str(E / 'source/e18_job.py')]
with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp); log = t / 'log'; saved = (e18_job.E, e18_job.stages)
    tag = lambda name: [sys.executable, '-c', f"open({str(log)!r}, 'a').write({name!r} + chr(10))"]
    def acc(passed):
        return [sys.executable, '-c', f"import json; open({str(t / 'sanity-acceptance.json')!r}, 'w').write(json.dumps(dict(passed={passed}, cuda_initialized=False, scope='sanity only')))"]
    def trial(first, accept):
        for f in [log, t / 'sanity-acceptance.json']:
            f.unlink(missing_ok=True)
        e18_job.stages = lambda: [('sanity', first), ('sanity-acceptance', accept), ('main', tag('main'))]
        try:
            e18_job.main(); outcome = 'completed'
        except (AssertionError, SystemExit) as err:
            outcome = type(err).__name__
        return outcome, log.read_text().split() if log.exists() else []
    try:
        e18_job.E = t
        assert trial(tag('sanity'), acc(True)) == ('completed', ['sanity', 'main'])
        assert trial(tag('sanity'), acc(False)) == ('AssertionError', ['sanity'])                       # failed acceptance: main not started
        assert trial([sys.executable, '-c', 'raise SystemExit(3)'], acc(True)) == ('SystemExit', [])    # failed sanity: nothing later runs
        assert not (t / 'sanity-acceptance.json').exists()
        (t / 'sanity-acceptance.json').write_text('{}')
        try: e18_job.main()
        except AssertionError: pass
        else: raise AssertionError('a stale sanity acceptance was not refused')
    finally:
        e18_job.E, e18_job.stages = saved
report['job'] = {'stages': [n for n, _ in st], 'gpurun_command': cmd, 'gate_trials': ['pass -> main', 'failed acceptance -> no main', 'failed sanity -> stop', 'stale acceptance refused']}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e18_runner.py', E / 'source/e18_accept.py', E / 'source/prepare_inputs.py', E / 'source/e18_freeze_submit.py', E / 'source/e18_job.py',
         E / 'config/sanity.json', E / 'config/main.json', E / 'EXPERIMENT_PLAN.md', E / 'input/preparation.json', E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=1, default=str))
