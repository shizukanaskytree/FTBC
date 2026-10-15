"""Locked evaluation v2 configs (plan sections 3 and 5): ImageNet jobs from the E23 configs (the constructor keys of the frozen chains), CIFAR jobs
from the E25 configs; plus the CIFAR rows manifests that the frozen E25 constructor reads (its 'development' role: the first 16 E25 dev-L rows,
used only for the frozen strict-reload scoring; C1-C3: the new pools). Run once before the freeze; CPU only."""
import hashlib, json
from pathlib import Path
E = Path(__file__).resolve().parents[1]; C1 = E.parent / 'c1-pre'
RUNS = '/data7/projects/ftbc/runs/locked-eval-v2-20261011'
TUNED_IMAGENET = {'r34': {2: {'alpha': 0.3, 'passes': 1}, 4: None, 8: None}, 'vgg': {T: {'alpha': 0.6, 'passes': 1} for T in (2, 4, 8)}}
TUNED_CIFAR = {'c10-vgg16': {2: (0.4, 3), 4: (0.1, 3), 8: (0.4, 3)}, 'c10-resnet20': {2: (0.3, 1), 4: (0.3, 1), 8: None},
               'c100-vgg16': {2: (0.8, 1), 4: (0.3, 3), 8: (0.8, 1)}, 'c100-resnet20': {2: (0.3, 1), 4: (0.2, 1), 8: (0.6, 1)}}
RELEASE_SEED = {'imagenet': 2026101140, 'cifar10': 2026101141, 'cifar100': 2026101142}


CARDS = {'h100': ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211'], 'a100': ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']}
# plan section 5: every ImageNet job may run on the H100; all but R34 T=4 and T=8 may also run on either A100 (those two would risk their time
# limits there, follow-up 1); the job records its card
IMAGENET_CARDS = lambda model, T: CARDS['h100'] if (model, T) in (('r34', 4), ('r34', 8)) else CARDS['a100'] + CARDS['h100']
RELEASE_MODE = 'skip'   # plan section 3 (candidate 4): the release-default arm FTBC-def is not run (author's decision of 10-11 evening); only the
                        # ECCV paper's reported values are cited. Applies to ImageNet and CIFAR.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


prep = json.loads((E / 'input/preparation.json').read_text())
for model in ['r34', 'vgg']:
    for T in (2, 4, 8):
        for phase in ['sanity', 'main']:
            c = json.loads((C1 / f'e23/config/{model}-{phase}.json').read_text()); tag = f'{model}-t{T}'
            c.update(output=f'{RUNS}/{tag}-{phase}-v1', freeze_manifest=str(E / 'protocol-freeze.json'), model=model, T=T,
                     calibration_manifest=str(E / 'input/imagenet-calibration.json'), calibration_manifest_sha256=sha(E / 'input/imagenet-calibration.json'),
                     e23_r34_runner=str(C1 / 'e23/source/e23_r34_runner.py'), e15_runner=str(C1 / 'e15/source/e15_runner.py'),
                     release_chunked=str(C1 / 'e29/source/release_chunked.py'), tuned=TUNED_IMAGENET[model][T], release_mode=RELEASE_MODE,
                     allowed_uuids=IMAGENET_CARDS(model, T),
                     release_seed=RELEASE_SEED['imagenet'], release_n=3, tiny_n=32 if phase == 'main' else 4)
            if phase == 'main':
                c.update(release_batch=200, release_chunk=50, evaluation_manifest=str(E / 'input/imagenet-val.json'), evaluation_key='evaluation',
                         evaluation_n=50000, evaluation_manifest_sha256=sha(E / 'input/imagenet-val.json'), evaluation_batch_size=50, evaluation_workers=12)
            else:                                                  # sanity: 24 release images (3 batches of 8), chunks of 2, the E23 fresh-K stand-in
                c.update(release_batch=8, release_chunk=2, evaluation_manifest=c['fresh_manifest'], evaluation_key='development', evaluation_n=16,
                         evaluation_manifest_sha256=c['fresh_manifest_sha256'], evaluation_batch_size=8, evaluation_workers=4)
            assert c['small_n'] == 3 * c['ftbc_batch'] and c['tiny_n'] < c['small_n'] < c['release_n'] * c['release_batch']
            (E / f'config/{tag}-{phase}.json').write_text(json.dumps(c, indent=1) + '\n')
for tag in TUNED_CIFAR:
    ds = 'cifar100' if tag.startswith('c100') else 'cifar10'          # 'c100-...' also starts with 'c10'
    e25rows = json.loads((C1 / f'e25/input/{ds}-rows.json').read_text()); pools = json.loads((E / f'input/{ds}-calibration.json').read_text())
    rows_path = E / f'input/{ds}-rows-for-e25-constructor.json'
    if not rows_path.exists():
        rows_path.write_text(json.dumps({'development': e25rows['development'][:16], **pools}, indent=2) + '\n')
    for phase in ['sanity', 'main']:
        c = json.loads((C1 / f'e25/config/{tag}-{phase}.json').read_text())
        c.update(output=f'{RUNS}/{tag}-{phase}-v1', freeze_manifest=str(E / 'protocol-freeze.json'), tag=tag, e25_runner=str(C1 / 'e25/source/e25_runner.py'),
                 cifar_test=str(E / 'source/cifar_test.py'), rows_manifest=str(rows_path), rows_manifest_sha256=sha(rows_path), dev_count=16,
                 tuned={f'T{T}': (None if v is None else {'alpha': v[0], 'passes': v[1]}) for T, v in TUNED_CIFAR[tag].items()},
                 release_seed=RELEASE_SEED[ds], tiny_n=32 if phase == 'main' else 4, release_mode=RELEASE_MODE)
        if phase == 'main':
            c.update(evaluation='test', test_file=prep[ds]['test_file'], test_file_sha256=prep[ds]['test_file_inventory_sha256'], evaluation_n=10000)
        else:
            c.update(evaluation='standin', standin_manifest=str(C1 / f'e25/input/{ds}-rows.json'), standin_n=64, evaluation_n=64,
                     release_batch=8, release_n=24)                # 24 release images (3 batches of 8) so that small_n (12) < release_n
        assert c['small_n'] == 3 * c['ftbc_batch'] and c['tiny_n'] < c['small_n'] < c['release_n'] == 3 * c['release_batch'] <= c['draw_n']
        (E / f'config/{tag}-{phase}.json').write_text(json.dumps(c, indent=1) + '\n')
print('configs:', sorted(p.name for p in (E / 'config').glob('*.json')))
