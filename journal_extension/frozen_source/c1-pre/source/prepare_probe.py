"""CPU-only input whitelist and immutable source export for C1-pre E0."""
import hashlib
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[4]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), p
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def main():
    assert (ROOT / 'env.sh').is_file(), ROOT
    parent = WORK.parent / 'evidence/c1-core-input-roles.json'
    roles = json.loads(parent.read_text())
    chosen = {'fit':roles['fit_ids'][:4096], 'diagnostic':roles['fit_ids'][4096:4160],
              'development':roles['development_ids'][:512]}
    need = set().union(*map(set, chosen.values()))
    assert len(need) == sum(map(len, chosen.values())) == 4672
    inventory = ROOT / 'ccfa-workfiles/experiments/ftbc-neural-networks/raw-data-inventory/imagenet-train.jsonl'
    assert sha(inventory) == 'ee17a28cf07c0f9cc0a2caac801772bfec299b6d07ef1710fbe68154864285e6'
    rows = {}
    with inventory.open() as f:
        for line in f:
            row = json.loads(line)
            if row['index'] in need:
                path = ROOT / 'datasets/imagenet/train' / row['path']
                rows[row['index']] = dict(row, path=str(path.resolve()))
    assert set(rows) == need
    def check(row):
        assert sha(row['path']) == row['sha256'], row['index']
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, rows.values()))
    exported = {role:[rows[i] for i in ids] for role, ids in chosen.items()}
    for a,b in [('fit','diagnostic'),('fit','development'),('diagnostic','development')]:
        assert not ({r['sha256'] for r in exported[a]} & {r['sha256'] for r in exported[b]})
    write(WORK/'input-roles.json', exported)
    sources = {}
    for family, src, subfolder in [
        ('parallel', ROOT/'code/parallel-conversion', 'models'),
        ('serial', ROOT/'code/ftbc-eccv/QCFS/ANN_SNN_QCFS', 'Models')]:
        paths = list((src/subfolder).rglob('*.py'))
        if family == 'parallel':
            paths.append(src/'modules.py')
        for p in paths:
            target = WORK/'source-snapshot'/family/p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists()
            shutil.copyfile(p, target)
            assert sha(p) == sha(target)
            sources[str(target.relative_to(ROOT))] = {'sha256':sha(target), 'origin':str(p.relative_to(ROOT))}
    checkpoint = ROOT/'code/parallel-conversion/checkpoints/imagenet-resnet34/source_ann/ImageNet-ResNet34-t8.pth'
    expected = '8f98b197a943aee0a1cb8971a04a7e1d1fed0cb80f5d32a0dd89c9bd6ece6bb2'
    assert sha(checkpoint) == expected
    outputs = Path('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008')
    for family, T, uuid in [
        ('parallel',2,'GPU-08d042a1-efba-a7d5-657e-8543e26f3154'),
        ('serial',4,'GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211')]:
        out = outputs/family
        out.mkdir(parents=True, exist_ok=False)
        config = {'workspace':str(ROOT),'family':family,'timesteps':T,
          'source':{'checkpoint':str(checkpoint),'checkpoint_sha256':expected,'quantization_level':8,'num_classes':1000},
          'source_snapshot':{f:str(WORK/'source-snapshot'/f) for f in ['parallel','serial']},
          'dataset_manifest':str(WORK/'input-roles.json'),'output':str(out),'allowed_uuid':uuid,'seed':20261008,
          'fit':{'batch_size':16,'learning_rate':.0002,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,
                 'gradient_clip':1,'warmup_steps':20,'warmup_start_factor':.1,'final_factor':.01},
          'diagnostics':{'channels':16,'positions':4},'num_workers':4,'eval_batch_size':4}
        write(WORK/'config'/f'{family}.json', config)
    write(WORK/'input-preparation.json', {'status':'passed','parent_roles_sha256':sha(parent),
        'inventory_sha256':sha(inventory),'whitelist_sha256':sha(WORK/'input-roles.json'),
        'counts':{k:len(v) for k,v in exported.items()},'all_image_hashes_checked':True,
        'source_snapshots':sources,'checkpoint_sha256':expected,'protected_images_opened':False})
    print(json.dumps({'status':'passed','counts':{k:len(v) for k,v in exported.items()},'source_files':len(sources)}))


if __name__ == '__main__':
    main()
