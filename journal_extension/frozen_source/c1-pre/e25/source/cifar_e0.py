"""E25 CIFAR data helpers with the E0 run_probe interface that the frozen E18 Run methods call through `self.e0`: loader, evaluate and logits
for CIFAR rows (the released evaluation transform), plus E0's atomic writes and state_digest (taken unchanged from run_probe.py). Also the
release-default FTBC batches (the released CIFAR training augmentation, shuffled, seeded). The official training pickles are read once from
the read-only bind mounts of /data4/datasets after their SHA-256 matches preparation.json; every row is checked against its manifest hash."""
import hashlib, importlib.util, json, pickle, random, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

W = Path('/data3/snn_prjs/ftbc_neural_networks')
E0_PATH = W / 'ccfa-workfiles/planning/ftbc-next-stage/route-c/c1-pre/source/run_probe.py'
DATA = W / 'code/parallel-conversion/downloaded_dataset'
AUGMENT = W / 'code/ftbc-eccv/QCFS/ANN_SNN_QCFS/Preprocess/augment.py'        # released CIFAR10Policy and Cutout
AUGMENT_SHA256 = '6b7302c03c377b78364bb319a05332841752684bad1bdb9fbd5b07bcf606dc6b'
NORM = {'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                # released Preprocess/getdataloader.py
        'cifar100': ([n / 255. for n in [129.3, 124.1, 112.4]], [n / 255. for n in [68.2, 65.4, 70.4]])}
CLASSES = {'cifar10': 10, 'cifar100': 100}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


E0 = _load('e25_cifar_e0_run_probe', E0_PATH)
sha256, atomic_json, atomic_torch, state_digest = E0.sha256, E0.atomic_json, E0.atomic_torch, E0.state_digest
STORES = {}


def init(dataset, preparation):
    """Load the official training rows of `dataset` once (file hashes from preparation.json)."""
    if dataset in STORES:
        return STORES[dataset]
    files = json.loads(Path(preparation).read_text())['datasets'][dataset]['train_files']; data, labels = [], []
    for rel, h in files.items():
        p = DATA / rel; assert sha256(p) == h['sha256'], rel
        with p.open('rb') as f:
            d = pickle.load(f, encoding='bytes')
        data.append(np.asarray(d[b'data'], dtype=np.uint8)); labels += list(d[b'labels' if dataset == 'cifar10' else b'fine_labels'])
    data = np.concatenate(data); assert data.shape == (50000, 3072) and len(labels) == 50000
    STORES[dataset] = {'data': data, 'labels': labels}
    return STORES[dataset]


def eval_transform(dataset):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(*NORM[dataset])])


def release_transform(dataset):
    """The released CIFAR training transform (Preprocess/getdataloader.py GetCifar10 / GetCifar100), file hash checked."""
    assert sha256(AUGMENT) == AUGMENT_SHA256
    aug = _load('e25_released_augment', AUGMENT)
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), aug.CIFAR10Policy(), transforms.ToTensor(),
                               transforms.Normalize(*NORM[dataset]), aug.Cutout(n_holes=1, length=16)])


class Rows(Dataset):
    def __init__(self, rows, dataset, transform):
        self.rows, self.store, self.transform = rows, STORES[dataset], transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, position):
        r = self.rows[position]; i = int(r['index']); raw = self.store['data'][i]
        if hashlib.sha256(raw.tobytes()).hexdigest() != r['sha256'] or self.store['labels'][i] != int(r['label']):
            raise ValueError(f'CIFAR row identity mismatch for training index {i}')
        image = Image.fromarray(raw.reshape(3, 32, 32).transpose(1, 2, 0))      # HWC uint8, as torchvision's CIFAR datasets
        return self.transform(image), int(r['label']), i


def loader(rows, config, batch_size, shuffle=False):
    generator = torch.Generator().manual_seed(config['seed'])
    return DataLoader(Rows(rows, config['dataset'], eval_transform(config['dataset'])), batch_size=batch_size, shuffle=shuffle,
                      num_workers=config.get('num_workers', 0), generator=generator, drop_last=False)


def logits(model, images, config, spiking=False):
    out = model(images).mean(0) if spiking else model(images)
    if out.shape != (images.shape[0], CLASSES[config['dataset']]) or not torch.isfinite(out).all():
        raise ValueError('Invalid or nonfinite logits')
    return out


@torch.no_grad()
def evaluate(model, rows, config, spiking=False):
    """E0 run_probe.evaluate with the CIFAR loader and class count."""
    model.eval()
    outputs, targets, identities = [], [], []
    for images, labels, ids in loader(rows, config, config.get('eval_batch_size', 200)):
        outputs.append(logits(model, images.cuda(non_blocking=True), config, spiking).cpu())
        targets.append(labels); identities.append(ids)
    outputs, targets, identities = torch.cat(outputs), torch.cat(targets), torch.cat(identities)
    predictions = outputs.argmax(1)
    return {'ids': identities, 'labels': targets, 'logits': outputs, 'predictions': predictions,
            'metrics': {'n': len(rows), 'correct': int((predictions == targets).sum()),
                        'accuracy_percent': float((predictions == targets).double().mean() * 100),
                        'cross_entropy': float(nn.functional.cross_entropy(outputs.double(), targets))}}


def release_batches(rows, dataset, seed, batch_size, n_batches):
    """Release-default FTBC inputs: `rows` through the released training transform, shuffled, `n_batches` batches of `batch_size` (the
    released main_cali.py consumes the first batches of a shuffled augmented training loader). Python, NumPy and torch generators are seeded
    with `seed` first, so the order and the augmentation are reproducible. Returns [(images, labels -1)] and the IDs per batch."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dl = DataLoader(Rows(rows, dataset, release_transform(dataset)), batch_size=batch_size, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(seed), drop_last=False)
    out, ids = [], []
    for i, (x, y, idx) in enumerate(dl):
        if i >= n_batches:
            break
        out.append((x, torch.full(y.shape, -1, dtype=y.dtype))); ids.append([int(v) for v in idx])
    assert len(out) == n_batches and all(b[0].shape[0] == batch_size for b in out)
    return out, ids
