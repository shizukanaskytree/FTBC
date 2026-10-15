"""E22 prospective train-derived inputs from one seeded permutation: fresh-J (2048, evaluation) and C1-C3 (3x256 calibration; all 256 for P0, the
first 96 for the registered FTBC recipe). Excludes every ID and content hash of the E21 exclusion sources plus E21 fresh-H and the E21 calibration
rows, so also the E0 roles, E5, E14-E20 splits and calibration rows. Adapted from E21 prepare_inputs.py. CPU only; identity metadata only."""
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path('/data3/snn_prjs/ftbc_neural_networks'); E = Path(__file__).resolve().parents[1]; SEED = 2026101022
SIZES = [('fresh', 2048)] + [(f'C{k}', 256) for k in (1, 2, 3)]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''): h.update(b)
    return h.hexdigest()


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


prev = json.loads((E.parent / 'e21/input/preparation.json').read_text())
paths = list(prev['exclusion_sources']) + [str(E.parent / 'e21/input/fresh-development.json'), str(E.parent / 'e21/input/calibration-rows.json')]
for p in paths[:-2]: assert sha(p) == prev['exclusion_sources'][p], p
assert sha(paths[-2]) == prev['fresh_manifest_sha256'] and sha(paths[-1]) == prev['calibration_manifest_sha256']
docs = [json.loads(Path(p).read_text()) for p in paths]
main, old, exposure, stability = docs[:4]
roles = main['datasets']['ImageNet']['roles']
excluded = ints(roles['calibration']) | ints(roles['development']) | ints(roles['confirmation'])
excluded |= ints(old['settings']['imagenet_relu_r18_t4']) | set(exposure['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4'])
for d in docs[4:]:
    excluded |= {r['index'] for rows in d.values() for r in rows}
excluded_sha = hashes(stability['settings']['imagenet']) | {r['sha256'] for d in docs[4:] for rows in d.values() for r in rows}
inventory = ROOT / main['datasets']['ImageNet']['splits']['train']['records_path']
assert sha(inventory) == main['datasets']['ImageNet']['splits']['train']['records_sha256']
need = sum(n for _, n in SIZES)
order = np.random.Generator(np.random.PCG64(SEED)).permutation(main['datasets']['ImageNet']['splits']['train']['count'])
candidates = [int(i) for i in order if int(i) not in excluded][:2 * need]; want = set(candidates); rows = {}
for line in inventory.open():
    index = int(re.search(r'"index"\s*:\s*(\d+)', line).group(1))
    if index in excluded:
        excluded_sha.add(re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"', line).group(1))
    elif index in want:
        rows[index] = json.loads(line)
chosen = []; seen = set(); skipped = 0
for i in candidates:
    row = rows[i]
    if row['sha256'] in excluded_sha or row['sha256'] in seen:
        skipped += 1; continue
    chosen.append({'index': int(row['index']), 'label': int(row['label']), 'path': str((ROOT / 'datasets/imagenet/train' / row['path']).resolve()), 'sha256': row['sha256']})
    seen.add(row['sha256'])
    if len(chosen) == need: break
assert len(chosen) == need and all(Path(r['path']).is_relative_to(ROOT) for r in chosen)
def check(r): assert sha(r['path']) == r['sha256'], r['index']
with ThreadPoolExecutor(8) as pool: list(pool.map(check, chosen))
parts, at = {}, 0
for name, n in SIZES:
    parts[name] = chosen[at:at + n]; at += n
for name, fname in [('fresh', 'fresh-development.json')]:
    out = E / f'input/{fname}'; assert not out.exists(); out.write_text(json.dumps({'development': parts['fresh']}, indent=2) + '\n')
cal = E / 'input/calibration-rows.json'; assert not cal.exists()
cal.write_text(json.dumps({k: v for k, v in parts.items() if k != 'fresh'}, indent=2) + '\n')
labels = [r['label'] for r in parts['fresh']]
prep = {'passed': True, 'role': 'E22 prospective research-internal inputs (evaluation fresh-J, calibration C1-C3), not confirmation',
        'seed': SEED, 'sizes': dict(SIZES), 'excluded_ids': len(excluded), 'excluded_content_hashes': len(excluded_sha), 'skipped_duplicate_or_excluded_content': skipped,
        'fresh_distinct_labels': len(set(labels)), 'exclusion_sources': {p: sha(p) for p in paths},
        'fresh_manifest_sha256': sha(E / 'input/fresh-development.json'), 'calibration_manifest_sha256': sha(cal), 'image_hashes_verified': True,
        'protected_image_label_prediction_payload_opened': False, 'protected_ID_hash_metadata_used_for_exclusion': True,
        'note': 'all rows are new: outside every earlier role, split and calibration set (E0, E5, E14-E21)'}
(E / 'input/preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
print(json.dumps({k: prep[k] for k in ['passed', 'excluded_ids', 'excluded_content_hashes', 'fresh_distinct_labels', 'sizes']}))
