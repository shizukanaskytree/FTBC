"""Locked evaluation v2: the official CIFAR test split for the evaluation role (plan section 3). The test pickle is read only after its whole-file
SHA-256 equals the 2026-10-02 raw-data inventory record; all 10,000 rows are used in file order with the released evaluation transform of the
E25 CIFAR helpers (cifar_e0.py, frozen). Also the check of section 3 that no calibration row has the bytes of a test row (reported, not used to
change the draw)."""
import hashlib, pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def init_test(CE, dataset, test_file, expected_sha256):
    assert CE.sha256(Path(test_file)) == expected_sha256, 'the CIFAR test pickle differs from the inventory record'
    with Path(test_file).open('rb') as f:
        d = pickle.load(f, encoding='bytes')
    data = np.asarray(d[b'data'], dtype=np.uint8); labels = [int(v) for v in d[b'labels' if dataset == 'cifar10' else b'fine_labels']]
    assert data.shape == (10000, 3072) and len(labels) == 10000
    rows = [{'index': i, 'label': labels[i], 'sha256': hashlib.sha256(data[i].tobytes()).hexdigest()} for i in range(10000)]
    return {'data': data, 'labels': labels, 'rows': rows, 'file_sha256': expected_sha256}


class TestRows(Dataset):
    def __init__(self, store, transform):
        self.store, self.transform = store, transform

    def __len__(self):
        return len(self.store['rows'])

    def __getitem__(self, i):
        r = self.store['rows'][i]; raw = self.store['data'][i]
        if hashlib.sha256(raw.tobytes()).hexdigest() != r['sha256'] or self.store['labels'][i] != r['label']:
            raise ValueError(f'CIFAR test row identity mismatch at {i}')
        image = Image.fromarray(raw.reshape(3, 32, 32).transpose(1, 2, 0))
        return self.transform(image), r['label'], i


@torch.no_grad()
def evaluate(CE, model, store, config, spiking):
    """cifar_e0.evaluate with the test rows; returns the same dictionary."""
    model.eval(); outputs, targets, ids = [], [], []
    dl = DataLoader(TestRows(store, CE.eval_transform(config['dataset'])), batch_size=config.get('eval_batch_size', 200), shuffle=False,
                    num_workers=config.get('num_workers', 0), drop_last=False)
    for images, labels, idx in dl:
        outputs.append(CE.logits(model, images.cuda(non_blocking=True), config, spiking).cpu()); targets.append(labels); ids.append(idx)
    outputs, targets, ids = torch.cat(outputs), torch.cat(targets), torch.cat(ids)
    predictions = outputs.argmax(1)
    return {'ids': ids, 'labels': targets, 'logits': outputs, 'predictions': predictions,
            'metrics': {'n': len(store['rows']), 'correct': int((predictions == targets).sum()),
                        'accuracy_percent': float((predictions == targets).double().mean() * 100),
                        'cross_entropy': float(nn.functional.cross_entropy(outputs.double(), targets))}}


def calibration_overlap(store, calibration):
    """Section 3: calibration rows (training split) whose bytes equal a test row; reported, the draw is not changed."""
    test = {r['sha256']: r['index'] for r in store['rows']}
    return [{'draw': d, 'train_index': int(r['index']), 'test_index': test[r['sha256']]} for d, rows in calibration.items() for r in rows if r['sha256'] in test]
