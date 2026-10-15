"""Create the E15 prospective train-derived development split fresh-B (2048 images), adapted from E14's prepare_development.py.
Excludes every ID and content hash of the E14 exclusion sources plus the E14 fresh2048 split. CPU only.
Protected roles contribute identity metadata (index, content hash) only; no protected image, label or prediction payload is read."""
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path('/data3/snn_prjs/ftbc_neural_networks'); E = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ccfa-workfiles/experiments/ftbc-neural-networks'; SEED = 2026100915; COUNT = 2048


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''): h.update(b)
    return h.hexdigest()


def save(p, x):
    assert not p.exists(), p
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2) + '\n')


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


paths = [BASE / 'parallel-main-inputs.json', BASE / 'protocol-inputs.json', BASE / 'h004-development-exposure-01.json',
         BASE / 'ssd-runs/FTBC-R054-CALIBRATION-SAMPLE-PREPARE-01/preparation.json', E.parent / 'input-roles.json', E.parent / 'e5/input-roles.json',
         E.parent / 'e14/input/fresh-development.json']
main, old, exposure, stability, e0, e5, e14 = [json.loads(p.read_text()) for p in paths]
roles = main['datasets']['ImageNet']['roles']
excluded = ints(roles['calibration']) | ints(roles['development']) | ints(roles['confirmation'])
excluded |= ints(old['settings']['imagenet_relu_r18_t4']) | set(exposure['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4'])
excluded |= {r['index'] for rows in e0.values() for r in rows} | {r['index'] for rows in e5.values() for r in rows} | {r['index'] for rows in e14.values() for r in rows}
excluded_sha = hashes(stability['settings']['imagenet']) | {r['sha256'] for rows in list(e0.values()) + list(e5.values()) + list(e14.values()) for r in rows}
inventory = ROOT / main['datasets']['ImageNet']['splits']['train']['records_path']
assert sha(inventory) == main['datasets']['ImageNet']['splits']['train']['records_sha256']
order = np.random.Generator(np.random.PCG64(SEED)).permutation(main['datasets']['ImageNet']['splits']['train']['count'])
candidates = [int(i) for i in order if int(i) not in excluded][:2 * COUNT]; want = set(candidates); rows = {}
for line in inventory.open():
    index = int(re.search(r'"index"\s*:\s*(\d+)', line).group(1))
    if index in excluded:
        excluded_sha.add(re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"', line).group(1))
    elif index in want:
        rows[index] = json.loads(line)
chosen = []; seen = set(); skipped_duplicate_content = 0
for i in candidates:
    row = rows[i]
    if row['sha256'] in excluded_sha or row['sha256'] in seen:
        skipped_duplicate_content += 1; continue
    chosen.append({'index': int(row['index']), 'label': int(row['label']), 'path': str((ROOT / 'datasets/imagenet/train' / row['path']).resolve()),
                   'sha256': row['sha256']})
    seen.add(row['sha256'])
    if len(chosen) == COUNT: break
assert len(chosen) == COUNT and all(Path(r['path']).is_relative_to(ROOT) for r in chosen)
def check(r): assert sha(r['path']) == r['sha256'], r['index']
with ThreadPoolExecutor(8) as pool: list(pool.map(check, chosen))
out = E / 'input/fresh-development.json'; save(out, {'development': chosen})
labels = [r['label'] for r in chosen]
save(E / 'input/preparation.json', {
    'passed': True, 'role': 'prospective research-internal development fresh-B for E15, not confirmation', 'seed': SEED, 'fresh_development_count': COUNT,
    'excluded_ids': len(excluded), 'excluded_content_hashes': len(excluded_sha), 'skipped_duplicate_or_excluded_content': skipped_duplicate_content,
    'distinct_labels': len(set(labels)), 'max_images_per_label': max(labels.count(l) for l in set(labels)),
    'exclusion_sources': {str(p): sha(p) for p in paths}, 'new_manifest_sha256': sha(out), 'image_hashes_verified': True,
    'protected_image_label_prediction_payload_opened': False, 'protected_ID_hash_metadata_used_for_exclusion': True,
    'historical_exposure': 'Source ANN and historical full training may have seen these training images; no historically pristine claim. '
                           'Excluded the known calibration/development/confirmation roles, prior recorded pilot content, the E0 roles, the E5 fresh development and the E14 fresh2048.'})
print(json.dumps({'passed': True, 'fresh_development': COUNT, 'excluded_ids': len(excluded), 'excluded_hashes': len(excluded_sha), 'distinct_labels': len(set(labels))}))
