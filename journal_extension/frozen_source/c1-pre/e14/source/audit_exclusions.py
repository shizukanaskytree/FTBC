"""Per-source exclusion audit of the E14 fresh development (analogue of E5's exclusion-audit.json). CPU only; identity metadata only."""
import hashlib, json, re
from pathlib import Path
ROOT = Path('/data3/snn_prjs/ftbc_neural_networks'); E = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ccfa-workfiles/experiments/ftbc-neural-networks'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ints(x):
    if isinstance(x, dict): return set().union(*(ints(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        if x and all(isinstance(v, int) for v in x): return set(x)
        return set().union(*(ints(v) for v in x)) if x else set()
    return set()


def hashes(x):
    if isinstance(x, dict): return set().union(*(hashes(v) for v in x.values())) if x else set()
    if isinstance(x, list): return set().union(*(hashes(v) for v in x)) if x else set()
    return {x} if isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) else set()


prep = json.loads((E / 'input/preparation.json').read_text())
for p, h in prep['exclusion_sources'].items():
    assert sha(p) == h, p
paths = list(prep['exclusion_sources'])
main, old, exposure, stability, e0, e5 = [json.loads(Path(p).read_text()) for p in paths]
roles = main['datasets']['ImageNet']['roles']
src_ids = {'main_calibration': ints(roles['calibration']), 'main_development': ints(roles['development']), 'main_confirmation': ints(roles['confirmation']),
           'old_protocol': ints(old['settings']['imagenet_relu_r18_t4']), 'h004': set(exposure['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4']),
           'E0': {r['index'] for v in e0.values() for r in v}, 'E5': {r['index'] for v in e5.values() for r in v}, 'stability': set()}
all_ids = set().union(*src_ids.values())
inventory = ROOT / main['datasets']['ImageNet']['splits']['train']['records_path']
assert sha(inventory) == main['datasets']['ImageNet']['splits']['train']['records_sha256']
id_sha = {}
for line in inventory.open():
    index = int(re.search(r'"index"\s*:\s*(\d+)', line).group(1))
    if index in all_ids:
        id_sha[index] = re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"', line).group(1)
assert set(id_sha) == all_ids
src_sha = {k: {id_sha[i] for i in v} for k, v in src_ids.items()}
src_sha['stability'] = hashes(stability['settings']['imagenet'])
fresh = json.loads((E / 'input/fresh-development.json').read_text())['development']
f_ids = {int(r['index']) for r in fresh}; f_sha = {r['sha256'] for r in fresh}
sources = {k: {'excluded_IDs': len(src_ids[k]), 'excluded_hashes': len(src_sha[k]), 'new_development_ID_overlap': len(src_ids[k] & f_ids),
               'new_development_content_overlap': len(src_sha[k] & f_sha), 'nonempty_source': bool(src_ids[k] or src_sha[k])} for k in src_ids}
assert all(v['new_development_ID_overlap'] == 0 and v['new_development_content_overlap'] == 0 and v['nonempty_source'] for v in sources.values())
union_ids = len(all_ids); union_sha = len(set().union(*src_sha.values()))
assert union_ids == prep['excluded_ids'] and union_sha == prep['excluded_content_hashes'], (union_ids, union_sha)
e5_prep = json.loads((E.parent / 'e5/input-preparation.json').read_text())
out = {'passed': True, 'sources': sources, 'fresh_count': len(fresh), 'union_excluded_IDs': union_ids, 'union_excluded_hashes': union_sha,
       'reconciliation_with_E5': {'E5_excluded_IDs': e5_prep['excluded_ids'], 'E5_excluded_hashes': e5_prep['excluded_content_hashes'],
                                  'added_by_E5_fresh_development_IDs': union_ids - e5_prep['excluded_ids'],
                                  'added_by_E5_fresh_development_hashes': union_sha - e5_prep['excluded_content_hashes']},
       'protected_scope': 'ID/hash metadata only for exclusions; no protected image files or prediction payload', 'script_sha256': sha(__file__)}
(E / 'input/exclusion-audit.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
