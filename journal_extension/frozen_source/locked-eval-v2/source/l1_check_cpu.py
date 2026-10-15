"""Locked evaluation v2 CPU check (plan sections 3-5): never initializes CUDA. Writes cache/cpu-check.json.
(1) inputs: the calibration manifests and the validation manifest match input/preparation.json; pools are disjoint from every exclusion source
    and from the validation contents (ImageNet); the validation rows equal the 2026-10-02 inventory; the CIFAR test files are identified by the
    inventory hashes (not read);
(2) configs agree with the plan: arms, G* (E24/E25 selections), T, sizes, release mode, evaluation role, outputs;
(3) the statistics: the bootstrap's multinomial form matches explicit image resampling on synthetic data; the four-way reading;
(4) a dry freeze collects every entry and every file exists; the test pickles are not among them;
(5) the five CPU dry runs passed on the current code (r34-t8, vgg-t8, c10-vgg16, the CIFAR test path c10-vgg16-test, c100-resnet20);
(6) the out-of-memory signatures of the acceptance equal the E23 runner's."""
import hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]; C1 = E.parent / 'c1-pre'
sys.path.insert(0, str(E / 'source'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {'checks': {}}
def ok(name, cond, detail=None):
    report['checks'][name] = {'passed': bool(cond), **({'detail': detail} if detail is not None else {})}
    assert cond, (name, detail)


prep = json.loads((E / 'input/preparation.json').read_text())
# (1)
ok('imagenet_manifest', sha(prep['imagenet']['calibration_manifest']) == prep['imagenet']['calibration_manifest_sha256'])
ok('val_manifest', sha(prep['imagenet_val']['manifest']) == prep['imagenet_val']['manifest_sha256'] and prep['imagenet_val']['rows'] == 50000 and not prep['imagenet_val']['images_opened'])
for ds in ['cifar10', 'cifar100']:
    ok(f'{ds}_manifest', sha(prep[ds]['calibration_manifest']) == prep[ds]['calibration_manifest_sha256'] and not prep[ds]['test_file_opened'])
cal = json.loads((E / 'input/imagenet-calibration.json').read_text()); val = json.loads((E / 'input/imagenet-val.json').read_text())['evaluation']
inv = [json.loads(l) for l in Path(prep['imagenet_val']['source']).open()]
ok('val_rows_equal_inventory', [(r['index'], r['label'], r['sha256']) for r in val] == [(r['index'], r['label'], r['sha256']) for r in inv] and sha(prep['imagenet_val']['source']) == prep['imagenet_val']['source_sha256'])
inventory = json.loads(Path(prep['imagenet_val']['inventory']).read_text())['datasets']['ImageNet']            # initial review N3
c2i = inventory['class_to_idx']; vsplit = inventory['splits']['val']
ok('val_records_hash_equals_inventory', prep['imagenet_val']['source_sha256'] == vsplit['records_sha256'] and vsplit['count'] == 50000)
ok('val_labels_follow_class_to_idx', all(r['label'] == c2i[Path(r['path']).parent.name] for r in val) and len(c2i) == 1000)
vsha = {r['sha256'] for r in val}; csha = [r['sha256'] for rows in cal.values() for r in rows]
ok('imagenet_pools_disjoint_from_validation_contents', not set(csha) & vsha and len(set(csha)) == len(csha) == 1800)
old = set()
for p in prep['imagenet']['exclusion_sources']:
    if p.endswith(('fresh-development.json', 'calibration-rows.json')):
        old |= {r['index'] for rows in json.loads(Path(p).read_text()).values() for r in rows}
ok('imagenet_pools_disjoint_from_e_series', not old & {r['index'] for rows in cal.values() for r in rows}, {'previous_rows': len(old)})
for ds in ['cifar10', 'cifar100']:
    pools = json.loads((E / f'input/{ds}-calibration.json').read_text()); e25 = json.loads((C1 / f'e25/input/{ds}-rows.json').read_text())
    prev = {r['index'] for rows in e25.values() for r in rows}; mine = [r['index'] for rows in pools.values() for r in rows]
    ok(f'{ds}_pools_disjoint_from_e25', not prev & set(mine) and len(set(mine)) == 1800)
# (2)
import ast
def assigned(path, name):
    """The literal value assigned to a module-level name, read with ast (make_configs.py is not imported: it writes the configs)."""
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
ok('release_mode_is_skip', assigned(E / 'source/make_configs.py', 'RELEASE_MODE') == 'skip')
G_IN = {'r34': {2: {'alpha': 0.3, 'passes': 1}, 4: None, 8: None}, 'vgg': {2: {'alpha': 0.6, 'passes': 1}, 4: {'alpha': 0.6, 'passes': 1}, 8: {'alpha': 0.6, 'passes': 1}}}
for model in ['r34', 'vgg']:
    for T in (2, 4, 8):
        for phase in ['sanity', 'main']:
            c = json.loads((E / f'config/{model}-t{T}-{phase}.json').read_text())
            cond = c['model'] == model and c['T'] == T and c['tuned'] == G_IN[model][T] and c['ftbc_alpha'] == 0.4
            cond &= c['release_mode'] == 'skip'                            # plan candidate 4: the release-default arm is not run
            h100, a100 = ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'], ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']
            cond &= sorted(c['allowed_uuids']) == sorted(h100 if (model, T) in (('r34', 4), ('r34', 8)) else h100 + a100)   # plan section 5
            cond &= c['output'] == f'/data7/projects/ftbc/runs/locked-eval-v2-20261011/{model}-t{T}-{phase}-v1' and c['freeze_manifest'] == str(E / 'protocol-freeze.json')
            if phase == 'main':
                cond &= c['evaluation_manifest'] == str(E / 'input/imagenet-val.json') and c['evaluation_n'] == 50000 and c['release_batch'] == 200 and c['release_chunk'] == 50
                cond &= (c['small_n'], c['tiny_n'], c['ftbc_batch']) == (96, 32, 32)
            ok(f'config_{model}_t{T}_{phase}', cond)
sel = {}
for t in ['c10-vgg16', 'c10-resnet20', 'c100-vgg16', 'c100-resnet20']:
    a = json.loads((C1 / f'e25/{t}-main-acceptance.json').read_text())
    for phase in ['sanity', 'main']:
        c = json.loads((E / f'config/{t}-{phase}.json').read_text()); cond = True
        for T in (2, 4, 8):
            s = a['selection'][f'T{T}']['selected']; want = None if (s['alpha'], s['passes']) == (0.4, 1) else {'alpha': s['alpha'], 'passes': s['passes']}
            cond &= c['tuned'][f'T{T}'] == want
        cond &= c['release_mode'] == 'skip'                                # plan candidate 4
        ds = 'cifar100' if t.startswith('c100') else 'cifar10'           # every dataset-specific key belongs to the job's own dataset
        rows = json.loads(Path(c['rows_manifest']).read_text()); pools = json.loads((E / f'input/{ds}-calibration.json').read_text())
        e25dev = json.loads((C1 / f'e25/input/{ds}-rows.json').read_text())['development'][:16]
        cond &= c['dataset'] == ds and c['rows_manifest'] == str(E / f'input/{ds}-rows-for-e25-constructor.json') and sha(c['rows_manifest']) == c['rows_manifest_sha256']
        cond &= all(rows[d] == pools[d] for d in ['C1', 'C2', 'C3']) and rows['development'] == e25dev
        cond &= c['release_seed'] == {'cifar10': 2026101141, 'cifar100': 2026101142}[ds]
        if phase == 'sanity':
            cond &= c['standin_manifest'] == str(C1 / f'e25/input/{ds}-rows.json')
        if phase == 'main':
            cond &= c['evaluation'] == 'test' and c['evaluation_n'] == 10000 and c['test_file'] == prep[ds]['test_file']
            cond &= c['test_file_sha256'] == prep[ds]['test_file_inventory_sha256']
            cond &= (c['small_n'], c['tiny_n'], c['release_n'], c['release_batch']) == (96, 32, 600, 200)
        ok(f'config_{t}_{phase}', cond, {'dataset': c['dataset'], 'rows_manifest': c['rows_manifest'], 'test_file': c.get('test_file')})
for model in ['r34', 'vgg']:                                          # initial review N2: every T, from the E24 acceptance of that T
    for T in (2, 4, 8):
        a = json.loads((C1 / f'e24/{"vgg" if model == "vgg" else f"r34-t{T}"}-main-acceptance.json').read_text())
        s = a['selection'][f'T{T}']['selected']; want = None if (s['alpha'], s['passes']) == (0.4, 1) else {'alpha': s['alpha'], 'passes': s['passes']}
        ok(f'imagenet_gstar_{model}_T{T}', G_IN[model][T] == want, {'selected': s})
# (3)
import l1_accept as A
rng = np.random.default_rng(0); xc = rng.random(4000) < 0.7; yc = rng.random(4000) < 0.66
b = A.bootstrap(xc, yc, 5); explicit = []
g = np.random.default_rng(1)
for _ in range(4000):
    i = g.integers(0, 4000, 4000); explicit.append(xc[i].mean() - yc[i].mean())
lo, hi = np.quantile(explicit, [0.025, 0.975])
ok('bootstrap_matches_explicit_resampling', abs(100 * lo - b['ci95_pp'][0]) < 0.3 and abs(100 * hi - b['ci95_pp'][1]) < 0.3, {'multinomial': b['ci95_pp'], 'explicit': [100 * lo, 100 * hi]})
ok('four_way_reading', A.reading(['candidate better'] * 3) == 'replicated' and A.reading(['candidate better', 'candidate worse', 'candidate better']).startswith('failed')
   and A.reading(['candidate better', 'no difference detected', 'no difference detected']).startswith('partially') and A.reading(['no difference detected'] * 3).startswith('not'))
# (4)
import l1_freeze_submit as FZ
with tempfile.TemporaryDirectory() as d:
    files, staged = FZ.freeze(dry_dir=d)
tests = {prep[ds]['test_file'] for ds in ['cifar10', 'cifar100']}
ok('dry_freeze', all(Path(p).exists() for p in files) and not tests & set(files), {'entries': len(files)})
# (5)
for t in ['r34-t8', 'vgg-t8', 'c10-vgg16', 'c10-vgg16-test', 'c100-resnet20']:
    p = E / f'cpu-dryrun/{t}/dryrun-report.json'; rep = json.loads(p.read_text()) if p.exists() else {}
    fresh = rep.get('passed') and not rep.get('cuda_initialized') and rep.get('tag') == t and all(sha(q) == h for q, h in rep.get('code_sha256', {}).items())
    fresh &= rep.get('injected', False) == t.endswith('-oom') and rep.get('variant') == next((v for v in ('oom', 'original', 'skip', 'test') if t.endswith('-' + v)), None)
    ok(f'dryrun_{t}', fresh, {'per_T': rep.get('per_T')})
ok('lib_oom_equal', assigned(E / 'source/l1_accept.py', 'LIB_OOM') == assigned(C1 / 'e23/source/e23_r34_runner.py', 'LIB_OOM'))
import torch
checked = list((E / 'source').glob('*.py')) + list((E / 'config').glob('*.json')) + list((E / 'input').glob('*.json')) + [E / 'tools/cpu_dryrun.py', E / 'README.md',
                                                                                                                       E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=all(v['passed'] for v in report['checks'].values()), cuda_initialized=torch.cuda.is_initialized(), files_sha256={str(p): sha(p) for p in checked})
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': {k: v['passed'] for k, v in report['checks'].items()}}))
