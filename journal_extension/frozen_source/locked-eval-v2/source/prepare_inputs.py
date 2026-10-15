"""Locked evaluation v2 inputs (../innovation/official-evaluation-plan-v2.md section 3). CPU only; identity metadata only (no model, no prediction).

Calibration pools: three new 600-image draws C1-C3 per dataset from the training split, one seeded permutation per dataset; both models of a
dataset share them.
  ImageNet: excludes every ID and content hash of the E28 exclusion sources and E28's own fresh and calibration rows (the E28 chain covers
            E0, E5, E14-E24 and E27; E29 reuses the E23 rows, which the chain covers), as the E22-E28 preparations did.
  CIFAR-10/100: excludes every ID and content hash of the E25 exclusion sources and E25's dev-L and C1-C3 (E26 used the same rows); earlier
            calibration-only use of training rows is disclosed, not excluded (it covers most of the split), as in E25.
Evaluation identities (no image or pickle of an evaluation set is read here):
  ImageNet: the 50,000 rows of the 2026-10-02 raw-data inventory imagenet-val.jsonl (index, train class_to_idx label, path, SHA-256), with
            absolute paths under datasets/imagenet/val; the runner checks every image hash when it loads it.
  CIFAR: the whole-file SHA-256 of each official test pickle as recorded in the same inventory; the runner checks it before reading.
Writes input/{imagenet,cifar10,cifar100}-calibration.json, input/imagenet-val.json and input/preparation.json."""
import hashlib, json, pickle, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path('/data3/snn_prjs/ftbc_neural_networks'); E = Path(__file__).resolve().parents[1]; C1 = E.parent / 'c1-pre'
INV = ROOT / 'ccfa-workfiles/experiments/ftbc-neural-networks/raw-data-inventory'
DATA = ROOT / 'code/parallel-conversion/downloaded_dataset'               # read-only bind mounts of /data4/datasets
SEEDS = {'imagenet': 2026101130, 'cifar10': 2026101131, 'cifar100': 2026101132}
POOLS = [('C1', 600), ('C2', 600), ('C3', 600)]
NAMES = {'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100'}
TEST_FILE = {'cifar10': 'cifar-10-batches-py/test_batch', 'cifar100': 'cifar-100-python/test'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''): h.update(b)
    return h.hexdigest()


def md5(p):
    h = hashlib.md5()
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


def imagenet(prep, val_sha):
    """The E28 preparation chain (adapted from E23/E28 prepare_inputs.py) plus E28's own rows, and every training row whose bytes equal a
    validation image (797 train-val groups in the 2026-10-02 duplicate audit): excluded by content, from the inventory hashes."""
    prev = json.loads((C1 / 'e28/input/preparation.json').read_text())
    paths = list(prev['exclusion_sources']) + [str(C1 / 'e28/input/fresh-development.json'), str(C1 / 'e28/input/calibration-rows.json')]
    for p in paths[:-2]: assert sha(p) == prev['exclusion_sources'][p], p
    assert sha(paths[-2]) == prev['fresh_manifest_sha256'] and sha(paths[-1]) == prev['calibration_manifest_sha256']
    docs = [json.loads(Path(p).read_text()) for p in paths]
    main, old, exposure, stability = docs[:4]
    roles = main['datasets']['ImageNet']['roles']
    excluded = ints(roles['calibration']) | ints(roles['development']) | ints(roles['confirmation'])
    excluded |= ints(old['settings']['imagenet_relu_r18_t4']) | set(exposure['exclude_from_future_study_confirmation']['imagenet_relu_r18_t4'])
    for d in docs[4:]:
        excluded |= {r['index'] for rows in d.values() for r in rows}
    excluded_sha = hashes(stability['settings']['imagenet']) | {r['sha256'] for d in docs[4:] for rows in d.values() for r in rows} | set(val_sha)
    inventory = ROOT / main['datasets']['ImageNet']['splits']['train']['records_path']
    assert sha(inventory) == main['datasets']['ImageNet']['splits']['train']['records_sha256']
    need = sum(n for _, n in POOLS)
    order = np.random.Generator(np.random.PCG64(SEEDS['imagenet'])).permutation(main['datasets']['ImageNet']['splits']['train']['count'])
    candidates = [int(i) for i in order if int(i) not in excluded][:2 * need]; want = set(candidates); rows = {}
    for line in inventory.open():
        index = int(re.search(r'"index"\s*:\s*(\d+)', line).group(1))
        if index in excluded:
            excluded_sha.add(re.search(r'"sha256"\s*:\s*"([0-9a-f]{64})"', line).group(1))
        elif index in want:
            rows[index] = json.loads(line)
    chosen, seen, skipped = [], set(), 0
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
    for name, n in POOLS:
        parts[name] = chosen[at:at + n]; at += n
    out = E / 'input/imagenet-calibration.json'; assert not out.exists(); out.write_text(json.dumps(parts, indent=2) + '\n')
    prep['imagenet'] = {'seed': SEEDS['imagenet'], 'excluded_ids': len(excluded), 'excluded_content_hashes': len(excluded_sha), 'skipped_duplicate_or_excluded_content': skipped,
                        'exclusion_sources': {p: sha(p) for p in paths}, 'calibration_manifest': str(out), 'calibration_manifest_sha256': sha(out),
                        'image_hashes_verified': True, 'validation_contents_excluded': len(val_sha),
                        'note': 'new rows: outside every earlier role, split and calibration set (E0, E5, E14-E28), and no row has the bytes of a validation image'}
    return excluded, excluded_sha


def imagenet_val(prep):
    """Evaluation identities from the raw-data inventory: no evaluation image is opened."""
    inv = json.loads((INV / 'inventory.json').read_text()); val = INV / 'imagenet-val.jsonl'
    assert inv['scope'].startswith('raw byte identity only') and inv['expected_counts']['ImageNet']['val'] == 50000
    rows = [json.loads(l) for l in val.open()]
    assert len(rows) == 50000 and [r['index'] for r in rows] == list(range(50000))
    distinct = len({r['sha256'] for r in rows})                       # the validation set has 29 byte-identical groups; all 50,000 rows are scored
    root = ROOT / 'datasets/imagenet/val'
    out_rows = [{'index': r['index'], 'label': int(r['label']), 'path': str(root / r['path']), 'sha256': r['sha256']} for r in rows]
    classes = sorted({r['path'].split('/')[0] for r in rows}); assert len(classes) == 1000
    out = E / 'input/imagenet-val.json'; assert not out.exists(); out.write_text(json.dumps({'evaluation': out_rows}) + '\n')
    prep['imagenet_val'] = {'source': str(val), 'source_sha256': sha(val), 'inventory': str(INV / 'inventory.json'), 'inventory_sha256': sha(INV / 'inventory.json'),
                            'rows': len(out_rows), 'distinct_contents': distinct, 'manifest': str(out), 'manifest_sha256': sha(out), 'images_opened': False,
                            'labels_per_class_min_max': [int(np.bincount([r['label'] for r in out_rows]).min()), int(np.bincount([r['label'] for r in out_rows]).max())]}
    return {r['sha256'] for r in out_rows}


def cifar(prep, ds, val_sha):
    """The E25 preparation (adapted) plus E25's own rows; the test pickle is identified by its inventory hash and not opened."""
    prev = json.loads((C1 / 'e25/input/preparation.json').read_text()); sources = list(prev['exclusion_sources'])
    for p in sources: assert sha(p) == prev['exclusion_sources'][p], p
    e25 = C1 / f'e25/input/{ds}-rows.json'; assert sha(e25) == prev['datasets'][ds]['manifest_sha256']
    main, old, exposure = [json.loads(Path(p).read_text()) for p in sources]
    roles = main['datasets'][NAMES[ds]]['roles']
    excluded = set(roles['development']['ids']) | set(roles['confirmation']['ids'])
    if ds == 'cifar100':
        s = old['settings']['cifar100_r20_t2']; excluded |= set(s['tuning']) | set(s['confirmation'])
        excluded |= set(exposure['exclude_from_future_study_confirmation']['cifar100_r20_t2'])
    previous = json.loads(e25.read_text()); excluded |= {r['index'] for rows in previous.values() for r in rows}
    split = main['datasets'][NAMES[ds]]['splits']['train']; data, labels, files = [], [], {}
    for rel in split['constructor_file_order']:
        p = DATA / rel; assert md5(p) == split['official_md5'][rel], rel; files[rel] = {'md5': split['official_md5'][rel], 'sha256': sha(p)}
        with p.open('rb') as f:
            d = pickle.load(f, encoding='bytes')
        data.append(np.asarray(d[b'data'], dtype=np.uint8)); labels += list(d[b'labels' if ds == 'cifar10' else b'fine_labels'])
    data = np.concatenate(data); n = len(labels); assert data.shape == (split['count'], 3072) and n == split['count']
    row_sha = [hashlib.sha256(data[i].tobytes()).hexdigest() for i in range(n)]
    excluded_sha = {row_sha[i] for i in excluded} | {r['sha256'] for rows in previous.values() for r in rows}
    order = np.random.Generator(np.random.PCG64(SEEDS[ds])).permutation(n)
    need = sum(k for _, k in POOLS); chosen, seen, skipped = [], set(), 0
    for i in map(int, order):
        if i in excluded: continue
        if row_sha[i] in excluded_sha or row_sha[i] in seen:
            skipped += 1; continue
        chosen.append({'index': i, 'label': int(labels[i]), 'sha256': row_sha[i]}); seen.add(row_sha[i])
        if len(chosen) == need: break
    assert len(chosen) == need
    parts, at = {}, 0
    for name, k in POOLS:
        parts[name] = chosen[at:at + k]; at += k
    out = E / f'input/{ds}-calibration.json'; assert not out.exists(); out.write_text(json.dumps(parts, indent=2) + '\n')
    inv = json.loads((INV / 'inventory.json').read_text())['datasets'][NAMES[ds]]['raw_files']
    test = DATA / TEST_FILE[ds]; rec = inv[TEST_FILE[ds]]
    prep[ds] = {'seed': SEEDS[ds], 'train_files': files, 'excluded_ids': len(excluded), 'excluded_content_hashes': len(excluded_sha), 'skipped_duplicate_or_excluded_content': skipped,
                'exclusion_sources': {**{p: sha(p) for p in sources}, str(e25): sha(e25)}, 'calibration_manifest': str(out), 'calibration_manifest_sha256': sha(out),
                'test_file': str(test), 'test_file_inventory_sha256': rec['sha256'], 'test_file_inventory_md5': rec['md5'], 'test_file_bytes': rec['bytes'], 'test_file_opened': False,
                'row_identity': 'sha256 of the 3072 raw bytes of the row as stored in the official pickle (channel-major, uint8)',
                'note': ('earlier calibration-only use of training rows is disclosed, not excluded (as E25); the test pickle is not opened here, so a '
                         'byte-identical calibration/test pair cannot be excluded in advance: the evaluation runner checks and reports it')}


if __name__ == '__main__':
    assert sys.argv[1:] == [], sys.argv
    prep = {'passed': True, 'role': 'locked evaluation v2 inputs: three 600-image calibration pools per dataset and the evaluation identities',
            'pools': dict(POOLS), 'protected_evaluation_payload_opened': False}
    val_sha = imagenet_val(prep); imagenet(prep, val_sha)
    cal = json.loads((E / 'input/imagenet-calibration.json').read_text())
    assert not {r['sha256'] for rows in cal.values() for r in rows} & val_sha, 'a calibration image has the content of a validation image'
    for ds in ['cifar10', 'cifar100']:
        cifar(prep, ds, val_sha)
    (E / 'input/preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('train_files', 'exclusion_sources')} if isinstance(v, dict) else v for k, v in prep.items()}, indent=1))
