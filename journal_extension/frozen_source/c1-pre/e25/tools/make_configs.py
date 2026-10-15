"""Write the eight E25 configs (four CIFAR source models x sanity/main; EXPERIMENT_PLAN.md sections 2-3 and 8). Refuses to overwrite; regenerated once
after the initial review, which added gpu_mem_cap_gib (N-e: the process's caching-allocator cap, below the 16 GiB shard) and the H100 to the
allowed UUIDs (either card type, chosen at submission)."""
import hashlib, json
from pathlib import Path
W = Path('/data3/snn_prjs/ftbc_neural_networks'); C1 = W / 'ccfa-workfiles/planning/ftbc-next-stage/route-c/c1-pre'; E = C1 / 'e25'
CK = W / 'code/parallel-conversion/checkpoints'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
A100 = ['GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894']
H100 = ['GPU-4bfc38ec-7b5c-032d-993e-16750f4fe211']                         # 2026-10-10 22:50: either card type, chosen at submission
MODELS = {'c10-vgg16': ('cifar10', 'vgg16', 10, 4, 15, 'cifar10-vgg16/source_ann/cifar10-vgg16-example.pth', None, '093383192641788a7f847d5bc28671b0d0fb1e7a8f0ccc486e79f13ed6b5da84'),
          'c10-resnet20': ('cifar10', 'resnet20', 10, 4, 19, 'cifar10-resnet20/source_ann/cifar10-resnet20-qcfs-l4.pth', None, 'be300d953f507c48227212a44a2714d7739d19fbad68c612b9bd918c06edf81f'),
          'c100-vgg16': ('cifar100', 'vgg16', 100, 8, 15, 'cifar100-vgg16/source_ann/cifar100-vgg16-l8-example.pth', None, '8da450ef6f867da8b35a092d6de080933d3873ad012978507783b7f8d6ef6339'),
          'c100-resnet20': ('cifar100', 'resnet20', 100, 8, 19, 'cifar100-resnet20/source_ann/cifar100-resnet20-qcfs-l8-best_raw70p22_epoch034.pth', 'model', '609eb2cc5fb85d812599ee2f3427210a78a8f0bf1e0954ad026f7cc6860d2597')}
SIZES = {'main': dict(dev_count=10000, draw_n=600, small_n=96, ftbc_batch=32, p0_sizes=[32, 96, 256], m0_size=96, release_n=600, release_batch=200, eval_batch_size=200),
         'sanity': dict(dev_count=16, draw_n=24, small_n=12, ftbc_batch=4, p0_sizes=[4, 12, 16], m0_size=12, release_n=12, release_batch=4, eval_batch_size=8)}
prep = E / 'input/preparation.json'
for tag, (ds, arch, ncls, level, n_if, rel, key, want) in MODELS.items():
    assert sha(CK / rel) == want, tag
    for which, sz in SIZES.items():
        c = {'workspace': str(W), 'e13_runner': str(C1 / 'e13/source/e13_runner.py'), 'e18_runner': str(C1 / 'e18/source/e18_runner.py'),
             'cifar_e0': str(E / 'source/cifar_e0.py'), 'step_solver': str(C1 / 'e14/source/step_solver.py'), 'serial_solver': str(C1 / 'e13/source/serial_solver.py'),
             'calibration_export': str(C1 / 'e1/calibration-export/serial.py'), 'calibration_export_manifest': str(C1 / 'e1/calibration-export/manifest.json'),
             'source_snapshot': str(C1 / 'source-snapshot/serial'), 'source_checkpoint': str(CK / rel), 'source_checkpoint_sha256': want, 'checkpoint_key': key,
             'source_level': level, 'interfaces': n_if, 'dataset': ds, 'num_classes': ncls, 'arch': arch,
             'preparation': str(prep), 'preparation_sha256': sha(prep), 'rows_manifest': str(E / f'input/{ds}-rows.json'), 'rows_manifest_sha256': sha(E / f'input/{ds}-rows.json'),
             'output': f'/data7/projects/ftbc/runs/c1pre-e25-20261010/{tag}-{which}-v1', 'freeze_manifest': str(E / 'protocol-freeze.json'),
             'allowed_uuids': A100 + H100, 'seed': 20261025, 'solver_batch': 32, 'bisection_steps': 32, 'bound_ratio': 0.5, 'ftbc_alpha': 0.4,
             'alphas': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2], 'Ts': [2, 4, 8], 'score_roles': ['fresh'], 'release_alpha': 0.4, 'release_seed': 2026101030,
             'eccv_alpha': 0.5, 'eccv_batch': 2, 'eccv_batches': 50, 'num_workers': 0, 'sanity': which == 'sanity', 'gpu_mem_cap_gib': 15, **sz}
        out = E / f'config/{tag}-{which}.json'; assert not out.exists(), out; out.write_text(json.dumps(c, indent=1) + '\n')
print('written', len(MODELS) * len(SIZES))
