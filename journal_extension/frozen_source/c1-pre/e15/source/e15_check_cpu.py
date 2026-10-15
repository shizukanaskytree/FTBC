"""E15 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e15_runner, e15_accept


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
# 1. Configs.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main', 'release', 'extra']}
shared = ['workspace', 'e0_runner', 'e13_runner', 'e14_runner', 'e13_output', 'e14_fresh_manifest', 'parent_config', 'plain_deployment', 'plain_proxy', 'e0_predictions',
          'calibration_export', 'calibration_export_manifest', 'freeze_manifest', 'fresh_manifest', 'fresh_manifest_sha256', 'allowed_uuids', 'seed', 'solver_batch',
          'bisection_steps', 'bound_ratio', 'ftbc_alpha', 'ftbc_arms', 'ftbc_ns', 'f1_n', 'score_roles', 'release_n', 'exclude_fit_prefix', 'fit_count', 'expandable_segments']
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert all(c[k] == cfg['main'][k] for k in shared), w
    assert set(c['allowed_uuids']) == {'GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894'}
    assert c['ftbc_alpha'] == 0.4 and c['bound_ratio'] == 0.5 and c['bisection_steps'] == 32 and c['ftbc_arms'] == [['F0-n3', 3, 0.4]] and c['release_n'] == 3
    assert c['fit_small'] == 3 * c['ftbc_batch'] == c['small_n'] and all(hi - lo > c['small_n'] for lo, hi in c['fit_draws'].values())
    assert all(v[1] - v[0] == c['release_n'] * c['release_batch'] for v in c['release_sets'].values())
    seeds = [v[2] for v in c['release_sets'].values()]; assert len(set(seeds)) == 3 and json.loads((E / 'input/preparation.json').read_text())['seed'] not in seeds
    spans = sorted([tuple(v) for v in c['fit_draws'].values()] + [tuple(v[:2]) for v in c['release_sets'].values()])
    assert spans[0][0] >= c['exclude_fit_prefix'] == 256 and all(a[1] <= b[0] for a, b in zip(spans, spans[1:])) and spans[-1][1] <= 4096
assert cfg['main']['stages'] == ['main'] and cfg['release']['stages'] == ['release'] and cfg['extra']['stages'] == ['extra'] and cfg['sanity']['stages'] == ['main', 'release', 'extra']
assert all(cfg[k]['fit_draws'] == {'D1': [256, 512], 'D2': [512, 768], 'D3': [768, 1024]} for k in ['main', 'release', 'extra'])
assert all(cfg[k]['release_sets'] == {'R1': [1024, 1624, 2026100931], 'R2': [1624, 2224, 2026100932], 'R3': [2224, 2824, 2026100933]} and cfg[k]['release_batch'] == 200 for k in ['main', 'release', 'extra'])
assert all((cfg[k]['fresh_count'], cfg[k]['dev_count'], cfg[k]['small_n']) == (2048, 512, 96) and cfg[k]['extra_T'] == [2] for k in ['main', 'release', 'extra'])
assert len({c['output'] for c in cfg.values()}) == 4
report['configs_checked'] = True
# 2. fresh-B identity, disjointness and provenance.
c = cfg['main']; assert sha(c['fresh_manifest']) == c['fresh_manifest_sha256']
fresh = json.loads(Path(c['fresh_manifest']).read_text())['development']
prep = json.loads((E / 'input/preparation.json').read_text()); assert prep['passed'] and prep['new_manifest_sha256'] == c['fresh_manifest_sha256']
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, 'exclusion source changed: ' + p
e14prep = json.loads((P / 'e14/input/preparation.json').read_text())
assert prep['excluded_ids'] == e14prep['excluded_ids'] + 2048 and prep['excluded_content_hashes'] == e14prep['excluded_content_hashes'] + 2048
old = [r for name in ['input-roles.json', 'e5/input-roles.json'] for v in json.loads((P / name).read_text()).values() for r in v] + json.loads(Path(c['e14_fresh_manifest']).read_text())['development']
ids = [int(r['index']) for r in fresh]
assert len(fresh) == 2048 and len(set(ids)) == 2048 and not set(ids) & {int(r['index']) for r in old} and not {r['sha256'] for r in fresh} & {r['sha256'] for r in old}
assert all(Path(r['path']).is_absolute() and Path(r['path']).resolve().is_relative_to(c['workspace']) and 0 <= int(r['label']) < 1000 for r in fresh)
report['fresh_B'] = {'count': 2048, 'distinct_labels': prep['distinct_labels'], 'excluded_ids': prep['excluded_ids'], 'excluded_hashes': prep['excluded_content_hashes'],
                     'disjoint_from_E0_E5_E14': True, 'images_read': False}
# 3. Reused frozen code and the E14 v2 freeze.
e14_freeze = json.loads((P / 'e14/protocol-freeze.json').read_text())['files_sha256']
for f in ['e14/source/e14_runner.py', 'e14/source/step_solver.py', 'e14/source/e14_accept.py', 'e13/source/e13_runner.py', 'e13/source/serial_solver.py']:
    p = str(P / f); assert e14_freeze[p] == sha(p), 'frozen code changed: ' + p
E13, E14, ss, ps, Run = e15_runner.build(c)
own = {'__init__', 'use_rows', 'ftbc_batches', 'registered_batches', 'release_batches', 'clear_release_hooks'}
assert own <= set(Run.__dict__)
for m in ['capture', 'replay_check', 'step_calibrate', 'count_calibrate_fit', 'score', 'score_ann']:
    assert m not in Run.__dict__ and getattr(Run, m) is getattr(Run.__mro__[1], m), m
for m in ['count_calibrate', 'ftbc_calibrate', 'teacher_means']:
    assert getattr(Run, m) is getattr(E13.Run, m), m
# 4. Draw rows sit in the E0 fit pool, disjoint, outside fit[:256].
parent = json.loads(Path(c['parent_config']).read_text()); roles = json.loads(Path(parent['dataset_manifest']).read_text())
draws, release = e15_runner.draw_rows(roles, c)
sets = [{int(r['index']) for r in v} for v in list(draws.values()) + list(release.values())] + [{int(r['index']) for r in roles['fit'][:256]}]
assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:]) and [len(s) for s in sets[:6]] == [256, 256, 256, 600, 600, 600]
report['draws'] = {k: len(v) for k, v in draws.items()} | {k: len(v) for k, v in release.items()}
# 5. Statistics helpers are the frozen E14 ones.
assert e15_accept.A14.upper_tail(3, 3) == 1 / 8 and e15_accept.decide is e15_accept.A14.decide
B, W_, N_ = 'candidate better', 'candidate worse', 'no difference detected'
assert e15_accept.status([B, B, B]) == 'replicated' and e15_accept.status([B, B, N_]) == 'partially replicated' and e15_accept.status([B, B, W_]) == 'not replicated'
mk = lambda q1, q2, q3: e15_accept.combine({'decisions': {'Q1': {'status': q1}, 'Q2': {'status': q2}}}, {'decisions': {'Q3': {'pairs_pass': q3}}})
assert mk('replicated', 'replicated', True)['claim_better_than_release_default_FTBC_allowed'] and not mk('replicated', 'partially replicated', True)['claim_better_than_release_default_FTBC_allowed']
assert mk('replicated', 'not replicated', True)['claim_P0_better_than_M0_replicated'] and not mk('replicated', 'not replicated', True)['claim_P0_better_than_registered_FTBC_replicated']
assert not mk('partially replicated', 'replicated', True)['claim_better_than_release_default_FTBC_allowed'] and not e15_accept.combine(None, None)['claim_P0_better_than_M0_replicated']
report['statistics_from_frozen_e14'] = True; report['decision_logic_checked'] = True
# 6. Dry freeze: every path the real freeze would hash exists and hashes; the Slurm source manifest excludes symlinked entries.
import tempfile, e15_freeze_submit
with tempfile.TemporaryDirectory() as tmp:
    frozen, staged = e15_freeze_submit.freeze(dry_dir=tmp)
    w = Path(c['workspace'])
    assert all(Path(p).resolve().is_relative_to(w) for p in (w / k for k in staged)) and not any('ssd-runs' in k for k in staged)
    assert any('ssd-runs' in p for p in frozen) and str(P / 'e14/protocol-freeze.json') in frozen
    rel = w / 'code/ftbc-eccv/QCFS/ANN_SNN_QCFS'
    assert all(str(rel / f) in frozen for f in ['main_cali.py', 'Preprocess/getdataloader.py', 'Preprocess/__init__.py'])
report['dry_freeze'] = {'entries_without_review_files': len(frozen), 'staged': len(staged), 'symlinked_entries_excluded_from_staging': True}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e15_runner.py', E / 'source/e15_accept.py', E / 'source/prepare_development.py', E / 'source/e15_freeze_submit.py',
         E / 'config/sanity.json', E / 'config/main.json', E / 'config/release.json', E / 'config/extra.json', E / 'EXPERIMENT_PLAN.md', E / 'input/preparation.json',
         E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps(report, indent=1, default=str))
