"""E15 replication of the E14 per-step solver result (EXPERIMENT_PLAN.md). One process and one Slurm-allocated A100 per job.

Reuses the frozen E14 runner (which reuses the frozen E13 runner): E0 construction, GPU guard, prefix capture, the M0
count-mean solver, the P0 per-step solver with per-layer and full-network replay, strict-reload scoring on the fresh and
development roles. New here: explicit fit-row draws (three disjoint 256-image draws from the E0 fit pool, whose first 96
images carry the data-matched FTBC comparison), three independent realizations of the release-default FTBC recipe
(batch 200, RandomResizedCrop + horizontal flip, shuffled, 3 batches, each on its own disjoint 600 images and seed) with a
same-image P0 arm, and a release-job S0 anchor. The released FTBC update itself is the AST-verified E1 export, unchanged.
"""
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
import torch
import PIL
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
if not __debug__:
    raise SystemExit('E15 acceptance-critical checks use assert; do not run with -O')
STAGES = ('main', 'release', 'extra')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)).hexdigest()


class AugmentedRows(Dataset):
    """Release training transform of code/ftbc-eccv/QCFS/ANN_SNN_QCFS/Preprocess/getdataloader.py:83-88 on whitelisted rows."""
    def __init__(self, rows, sha256):
        self.rows = rows; self.sha256 = sha256
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.sha256(Path(row['path'])) != row['sha256']:
            raise ValueError(f"Image identity mismatch for training ID {row['index']}")
        with Image.open(row['path']) as image:
            return self.transform(image.convert('RGB')), int(row['label']), int(row['index'])


def draw_rows(roles, c):
    """Registered positions in the E0 fit pool: three disjoint draws and three disjoint release sets, all outside fit[:exclude_fit_prefix]."""
    fit = roles['fit']
    draws = {name: fit[lo:hi] for name, (lo, hi) in c['fit_draws'].items()}
    release = {name: fit[lo:hi] for name, (lo, hi, _seed) in c['release_sets'].items()}
    spans = sorted([tuple(v) for v in c['fit_draws'].values()] + [tuple(v[:2]) for v in c['release_sets'].values()])
    assert spans[0][0] >= c['exclude_fit_prefix'] and all(a[1] <= b[0] for a, b in zip(spans, spans[1:])) and spans[-1][1] <= len(fit), 'draws must be disjoint'
    assert all(len(v) >= c['small_n'] for v in draws.values()) and all(len(v) == c['release_n'] * c['release_batch'] for v in release.values())
    return draws, release


def build(c):
    E14 = load('e15_e14', c['e14_runner'])
    E13, ss, Run14 = E14.build(c)
    ps = E14.ps

    class Run(Run14):
        def __init__(self, c):
            super().__init__(c)
            e14_fresh = json.loads(Path(c['e14_fresh_manifest']).read_text())['development']
            assert not {int(r['index']) for r in self.rows['fresh']} & {int(r['index']) for r in e14_fresh}, 'fresh-B overlaps E14 fresh2048 (ID)'
            assert not {r['sha256'] for r in self.rows['fresh']} & {r['sha256'] for r in e14_fresh}, 'fresh-B overlaps E14 fresh2048 (content)'
            self.draws, self.release_sets = draw_rows(self.roles, c)
            self.release_record = {}

        def use_rows(self, key, rows):
            """E14 use_fit for an explicit row list (teacher targets recomputed on exactly these rows, batch solver_batch)."""
            self.fit_n = len(rows)
            self.fit_batches = list(self.e0.loader(rows, self.parent, self.c['solver_batch']))
            assert torch.equal(torch.cat([b[2] for b in self.fit_batches]), torch.tensor([int(v['index']) for v in rows]))
            if key not in self.targets_by_n:
                self.targets_by_n[key] = self.teacher_means()
                self.e0.atomic_torch(self.out / f'teacher-targets-{key}.pt', {'means': {k: v.cpu() for k, v in self.targets_by_n[key].items()},
                                                                              'batch': self.c['solver_batch'], 'ids': torch.tensor([int(v['index']) for v in rows])})
            self.targets = self.targets_by_n[key]

        def ftbc_batches(self, arm, T, batches, alpha, images, recipe, extra=None):
            """E13 ftbc_calibrate with prebuilt batches (same counting wrapper, zero start, digest checks)."""
            N = self.net(T); zero = self.zero_biases(T); self.set_biases(T, zero)
            init_hash = hashlib.sha256(b''.join(zero[n][t].to(torch.float32).contiguous().numpy().tobytes() for n in N['names'] for t in range(T))).hexdigest()
            n_batches = len(batches)
            count = {'teacher': 0, 'snn': 0, 'updates': {}}
            hs = [self.teacher.register_forward_pre_hook(lambda *a: count.__setitem__('teacher', count['teacher'] + 1)),
                  N['snn'].register_forward_pre_hook(lambda *a: count.__setitem__('snn', count['snn'] + 1))]
            ids = {id(N['modules'][n]): n for n in N['names']}; original = self.inherited.bias_corr_step_by_step
            def counted(*args, **kw):
                key = ids[id(args[3])]; count['updates'][key] = count['updates'].get(key, 0) + 1; return original(*args, **kw)
            self.inherited.bias_corr_step_by_step = counted
            teacher_before = self.e0.state_digest(self.teacher)
            try:
                with torch.no_grad():
                    self.inherited.bias_corr_model(self.teacher, N['snn'], T, batches, curr_t_alpha=alpha, num_cali_sample_batches=n_batches)
            finally:
                self.inherited.bias_corr_step_by_step = original
                for h in hs: h.remove()
            assert count['teacher'] == 33 * n_batches and count['snn'] == 33 * n_batches, count
            assert set(count['updates']) == set(N['names']) and all(v == n_batches for v in count['updates'].values())
            assert self.e0.state_digest(self.teacher) == teacher_before == self.teacher_digest
            biases = {n: [b.detach().cpu().clone() for b in N['modules'][n].time_based_bias] for n in N['names']}
            assert all(torch.isfinite(b).all() for v in biases.values() for b in v)
            self.ftbc[arm] = {'batches': n_batches, 'alpha': alpha, 'batch_size': int(batches[0][0].shape[0]), 'calls': count, 'init_bias_sha256': init_hash,
                              'images': images, 'recipe': recipe, 'inputs_sha256': tensor_hash([b[0] for b in batches]), **(extra or {}),
                              'max_abs_bias_over_theta': max(float(b.abs().max()) / float(N['modules'][n].thresh.data) for n, v in biases.items() for b in v)}
            return biases

        def clear_release_hooks(self):
            """The released GetLayerInputOutput removes its hook outside a finally block (serial.py:36-42); after a failure, drop any
            leftover ActivationSaverHookEachTimeStep hooks (and the clones they hold) from the teacher and every SNN."""
            dropped = 0
            for model in [self.teacher] + [N['snn'] for N in self.nets.values()]:
                for m in model.modules():
                    for key, hook in list(m._forward_hooks.items()):
                        if isinstance(hook, self.inherited.ActivationSaverHookEachTimeStep):
                            del m._forward_hooks[key]; dropped += 1
            return dropped

        def registered_batches(self, rows):
            out = [(images, torch.full(labels.shape, -1, dtype=labels.dtype)) for images, labels, _ in self.e0.loader(rows, self.parent, self.c['ftbc_batch'])]
            assert len(out) == 3 and sum(b[0].shape[0] for b in out) == len(rows)
            return out

        def release_batches(self, rows, seed):
            """Whitelisted fit images, release training transform, shuffled, batch release_batch (main_cali.py:121,183-190).
            Deterministic: num_workers=0, the global RNG (augmentation) is reseeded right before loading, shuffling uses its own generator."""
            torch.manual_seed(seed)
            loader = DataLoader(AugmentedRows(rows, self.e0.sha256), batch_size=self.c['release_batch'], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(seed), drop_last=False)
            batches, order = [], []
            for images, labels, ids in loader:
                batches.append((images, torch.full(labels.shape, -1, dtype=labels.dtype))); order += [int(i) for i in ids]
            assert len(batches) == self.c['release_n'] and sorted(order) == sorted(int(r['index']) for r in rows)
            return batches, order

    return E13, E14, ss, ps, Run


@torch.no_grad()
def run(c):
    stages = c['stages']; assert stages and set(stages) <= set(STAGES)
    if 'release' in stages and c.get('expandable_segments'):
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # before the first CUDA allocation; no numerical effect
    E13, E14, ss, ps, Run = build(c); r = Run(c); draws = r.draws; small = c['small_n']
    if 'main' in stages:
        T = 4
        r.score_ann('teacher', r.teacher, T); r.score_ann('twin-T4', r.net(T)['proxy'], T)
        r.score('S0', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2'})
        for d, rows in draws.items():
            r.use_rows(d, rows)
            r.free(); b = r.count_calibrate_fit(f'M0-{d}', T); r.score(f'M0-{d}', T, b, {'solver': 'M0', 'rows': d})
            r.free(); b = r.step_calibrate(f'P0-{d}', T); r.score(f'P0-{d}', T, b, {'solver': 'P0', 'rows': d})
            r.use_rows(f'{d}s', rows[:small])
            r.free(); b = r.step_calibrate(f'P0-{d}s', T); r.score(f'P0-{d}s', T, b, {'solver': 'P0', 'rows': f'{d}s'})
            r.free(); b = r.ftbc_batches(f'F0-{d}s', T, r.registered_batches(rows[:small]), c['ftbc_alpha'], [int(x['index']) for x in rows[:small]], 'registered')
            r.score(f'F0-{d}s', T, b, {'ftbc': r.ftbc[f'F0-{d}s']})
    if 'release' in stages:
        T = 4
        if 'main' not in stages:
            r.score('S0', T, r.zero_biases(T), {'schedule': 'IF v0=theta/2', 'role': 'cross-job anchor'})
        versions = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'PIL': PIL.__version__}
        for k, (lo, hi, seed) in c['release_sets'].items():
            rows = r.release_sets[k]
            r.use_rows(k, rows)
            r.free(); b = r.step_calibrate(f'P0-{k}', T); r.score(f'P0-{k}', T, b, {'solver': 'P0', 'rows': k})
            try:
                r.free(); batches, order = r.release_batches(rows, seed)
                b = r.ftbc_batches(f'F0-release-{k}', T, batches, c['ftbc_alpha'], order, 'release-default', {'seed': seed, 'versions': versions})
                del batches; r.free(); r.score(f'F0-release-{k}', T, b, {'ftbc': r.ftbc[f'F0-release-{k}']})
                r.release_record[k] = {'status': 'completed', 'seed': seed}
            except Exception as exc:                        # any failure (OOM or otherwise) is recorded; the remaining arms continue
                r.ftbc.pop(f'F0-release-{k}', None)
                r.release_record[k] = {'status': 'out_of_memory' if 'out of memory' in str(exc).lower() else 'failed', 'seed': seed,
                                       'exception_type': type(exc).__name__, 'traceback': traceback.format_exc()}
                del exc
                r.release_record[k]['leftover_hooks_dropped'] = r.clear_release_hooks()
                r.free(); r.progress('release_failed', set=k, status=r.release_record[k]['status'])
            r.e0.atomic_json(r.out / 'release.json', r.release_record)
    if 'extra' in stages:
        for Tx in c['extra_T']:
            r.score_ann(f'twin-T{Tx}', r.net(Tx)['proxy'], Tx)
            r.score(f'S0-T{Tx}', Tx, r.zero_biases(Tx), {'schedule': 'IF v0=theta/2'})
            for d, rows in draws.items():
                r.use_rows(d, rows)
                r.free(); b = r.count_calibrate_fit(f'M0-{d}-T{Tx}', Tx); r.score(f'M0-{d}-T{Tx}', Tx, b, {'solver': 'M0', 'rows': d})
                r.free(); b = r.step_calibrate(f'P0-{d}-T{Tx}', Tx); r.score(f'P0-{d}-T{Tx}', Tx, b, {'solver': 'P0', 'rows': d})
    r.e0.atomic_torch(r.out / 'solver.pt', r.solver); r.e0.atomic_json(r.out / 'ftbc.json', r.ftbc)
    assert r.e0.state_digest(r.teacher) == r.teacher_digest
    for Tx, N in r.nets.items():
        assert r.e0.state_digest(N['snn']) == N['immutable_digest'] and r.e0.state_digest(N['proxy']) == N['proxy_digest']
    torch.cuda.synchronize()
    result = {'status': 'completed', 'stages': stages, 'arms': list(r.records), 'teacher_unchanged': True, 'non_bias_unchanged': True,
              'twin_unchanged': True, 'proxy_digests': {str(k): v['proxy_digest'] for k, v in r.nets.items()}, 'all_strict_reloads_passed': True,
              'full_network_replay_equal': {k: v['full_network_replay_equal'] for k, v in r.solver.items()}, 'bias_sha256': r.bias_hashes,
              'teacher_targets_sha256': {str(n): hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in t.values())).hexdigest() for n, t in r.targets_by_n.items()},
              'draws': {d: [int(x['index']) for x in rows] for d, rows in draws.items()},
              'release_sets': {k: [int(x['index']) for x in rows] for k, rows in r.release_sets.items()}, 'release': r.release_record,
              'alloc_conf': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), 'hardware': r.hardware, 'seconds_total': time.perf_counter() - r.t0,
              'max_memory_allocated': torch.cuda.max_memory_allocated(), 'max_memory_reserved': torch.cuda.max_memory_reserved(),
              'data_status': 'fresh-B development (prospective) and exposed dev512 continuity'}
    r.e0.atomic_json(r.out / 'result.json', result); r.progress('completed')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args(); c = json.loads(Path(a.config).read_text())
    try:
        print(json.dumps(run(c)))
    except Exception:
        Path(c['output']).mkdir(parents=True, exist_ok=True)
        (Path(c['output']) / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc()}))
        raise
