"""E25 prospective train-derived CIFAR inputs (EXPERIMENT_PLAN.md section 2), one seeded permutation per dataset: dev-L (10000; the first 5000
are the selection half L-sel, the last 5000 the test half L-test) and C1-C3 (three 600-image calibration pools, the locked-evaluation design).
Both models of a dataset use the same rows. Excludes every ID of the earlier development, tuning and confirmation roles and the H004 exclusion
list (parallel-main-inputs.json, protocol-inputs.json, h004-development-exposure-01.json) and every row whose content equals an excluded row;
earlier calibration-only use is disclosed, not excluded (it covers most of the training split). The official pickles are opened only after
their MD5 matches the official value recorded in parallel-main-inputs.json. CPU only; identity metadata only (no model, no prediction)."""
import hashlib, json, pickle
from pathlib import Path
import numpy as np
ROOT = Path('/data3/snn_prjs/ftbc_neural_networks'); E = Path(__file__).resolve().parents[1]
DATA = ROOT / 'code/parallel-conversion/downloaded_dataset'          # read-only bind mounts of /data4/datasets
X = ROOT / 'ccfa-workfiles/experiments/ftbc-neural-networks'
SOURCES = [X / 'parallel-main-inputs.json', X / 'protocol-inputs.json', X / 'h004-development-exposure-01.json']
SEEDS = {'cifar10': 2026101025, 'cifar100': 2026101125}
SIZES = [('development', 10000)] + [(f'C{k}', 600) for k in (1, 2, 3)]
NAMES = {'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100'}


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


def train_rows(ds, main):
    """(raw CHW uint8 rows (50000, 3072), labels) of the official training split in torchvision constructor order."""
    split = main['datasets'][NAMES[ds]]['splits']['train']; data, labels, files = [], [], {}
    for rel in split['constructor_file_order']:
        p = DATA / rel; assert md5(p) == split['official_md5'][rel], rel; files[rel] = {'md5': split['official_md5'][rel], 'sha256': sha(p)}
        with p.open('rb') as f:
            d = pickle.load(f, encoding='bytes')
        data.append(np.asarray(d[b'data'], dtype=np.uint8)); labels += list(d[b'labels' if ds == 'cifar10' else b'fine_labels'])
    data = np.concatenate(data); assert data.shape == (split['count'], 3072) and len(labels) == split['count']
    return data, labels, files


main, old, exposure = [json.loads(p.read_text()) for p in SOURCES]
prep = {'passed': True, 'role': 'E25 prospective research-internal CIFAR inputs (dev-L in two halves, calibration pools C1-C3), not confirmation',
        'sizes': dict(SIZES), 'exclusion_sources': {str(p): sha(p) for p in SOURCES}, 'datasets': {},
        'row_identity': 'sha256 of the 3072 raw bytes of the row as stored in the official pickle (channel-major, uint8)',
        'protected_test_split_opened': False, 'note': ('train-derived: the source ANNs were trained on the training split; earlier '
        'calibration-only use of training images (most of the split) is disclosed, not excluded')}
for ds in ['cifar10', 'cifar100']:
    roles = main['datasets'][NAMES[ds]]['roles']
    excluded = set(roles['development']['ids']) | set(roles['confirmation']['ids'])
    if ds == 'cifar100':
        s = old['settings']['cifar100_r20_t2']; excluded |= set(s['tuning']) | set(s['confirmation'])
        excluded |= set(exposure['exclude_from_future_study_confirmation']['cifar100_r20_t2'])
    data, labels, files = train_rows(ds, main); n = len(labels)
    row_sha = [hashlib.sha256(data[i].tobytes()).hexdigest() for i in range(n)]
    excluded_sha = {row_sha[i] for i in excluded}
    order = np.random.Generator(np.random.PCG64(SEEDS[ds])).permutation(n)
    need = sum(k for _, k in SIZES); chosen, seen, skipped = [], set(), 0
    for i in map(int, order):
        if i in excluded: continue
        if row_sha[i] in excluded_sha or row_sha[i] in seen:
            skipped += 1; continue
        chosen.append({'index': i, 'label': int(labels[i]), 'sha256': row_sha[i]}); seen.add(row_sha[i])
        if len(chosen) == need: break
    assert len(chosen) == need
    parts, at = {}, 0
    for name, k in SIZES:
        parts[name] = chosen[at:at + k]; at += k
    out = E / f'input/{ds}-rows.json'; assert not out.exists(); out.write_text(json.dumps(parts, indent=2) + '\n')
    dev = parts['development']; half = len(dev) // 2
    prep['datasets'][ds] = {'seed': SEEDS[ds], 'train_files': files, 'excluded_ids': len(excluded), 'excluded_content_hashes': len(excluded_sha),
                            'skipped_duplicate_or_excluded_content': skipped, 'manifest': str(out), 'manifest_sha256': sha(out),
                            'labels_per_class_dev_min_max': [int(np.bincount([r['label'] for r in dev]).min()), int(np.bincount([r['label'] for r in dev]).max())],
                            'distinct_labels_selection_half': len({r['label'] for r in dev[:half]}), 'distinct_labels_test_half': len({r['label'] for r in dev[half:]})}
(E / 'input/preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
print(json.dumps({ds: {k: v for k, v in d.items() if k != 'train_files'} for ds, d in prep['datasets'].items()}, indent=1))
