"""E4 frozen-parent diagnostic runtime. CUDA only inside a Slurm allocation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import time
from pathlib import Path
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(config, cpu=False):
    root = Path(config['workspace'])
    parent = json.loads(Path(config['parent_config']).read_text())
    e0 = load('e4_frozen_e0', config['parent_runner'])
    roles = e0.validate_config(parent)
    # Validate the E3 manifest too: source/data/parent identities were accepted there.
    prior = json.loads(Path(config['prior_freeze']).read_text())
    for path, sha in prior['files_sha256'].items():
        assert e0.sha256(path) == sha, path
    manifest_path = Path(config['freeze_manifest'])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for path, sha in manifest['files_sha256'].items():
            assert e0.sha256(path) == sha, path
    else:
        assert cpu, 'GPU execution requires a frozen protocol'
    selected = {'fit': roles['fit'][:config['fit_count']],
                'development': roles['development'][:config['dev_count']],
                'diagnostic': roles['fit'][:config['diagnostic_count']]}
    # Diagnostics deliberately use fitting images. No new holdout is consumed.
    for field in ('index', 'sha256'):
        assert not {r[field] for r in roles['fit']} & {r[field] for r in roles['development']}
    for rows in selected.values():
        for row in rows:
            assert Path(row['path']).is_relative_to(root / 'datasets/imagenet/train')
    source, classes = e0.source_model(parent)
    proxy = e0.make_proxy(source, parent, classes)
    proxy.load_state_dict(torch.load(config['parent_proxy'], map_location='cpu', weights_only=True)['state_dict'], strict=True)
    bn = torch.load(config['parent_deployment'], map_location='cpu', weights_only=False)
    calibrated = torch.load(config['calibrated_deployment'], map_location='cpu', weights_only=False)
    assert set(bn['state_dict']) == set(calibrated['state_dict'])
    keys = [k for k in bn['state_dict'] if not k.endswith('output_bias_correction')]
    assert all(torch.equal(bn['state_dict'][k], calibrated['state_dict'][k]) for k in keys)
    names = list(bn['runtime_biases'])
    assert len(names) == 33 and set(names) == set(calibrated['runtime_biases'])
    assert bn['interface_thresholds'] == calibrated['interface_thresholds']
    for name in names:
        assert not torch.count_nonzero(bn['runtime_biases'][name])
        x = calibrated['runtime_biases'][name]
        assert torch.isfinite(x).all() and torch.equal(x, x[:1].expand_as(x))
        assert torch.equal(x, calibrated['state_dict'][name + '.output_bias_correction'])
    if cpu:
        return {'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
                'non_bias_parents_equal': True, 'interfaces': names,
                'roles': {k: len(v) for k, v in selected.items()},
                'channels': sum(bn['runtime_biases'][n].shape[2] for n in names),
                'nonpositive_effective_thresholds': sum(int((calibrated['runtime_biases'][n][0] >= bn['interface_thresholds'][n]).sum()) for n in names)}
    return Runtime(config, e0, parent, selected, source, proxy, classes, bn, calibrated, names)


class Runtime:
    def __init__(self, c, e0, parent, rows, source, proxy, classes, bn, calibrated, names):
        self.started = time.perf_counter()
        self.c, self.e0, self.parent, self.rows = c, e0, parent, rows
        self.classes, self.proxy, self.bn, self.calibrated, self.names = classes, proxy, bn, calibrated, names
        self.output = Path(c['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        assert not any((self.output / f).exists() for f in ('progress.json', 'result.json', 'failure.json'))
        self.hardware = e0.gpu_guard(c)
        e0.atomic_json(self.output / 'hardware.json', self.hardware)
        e0.atomic_json(self.output / 'config.json', c)
        torch.set_num_threads(8)
        torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.model = e0.make_snn(proxy, parent, classes)
        self.model.load_state_dict(bn['state_dict'], strict=True)
        self.modules = dict(self.model.named_modules())
        for name in names:
            self.modules[name].v_threshold = bn['interface_thresholds'][name]
        self.immutable = [k for k in self.model.state_dict() if not k.endswith('output_bias_correction')]
        self.initial_digest = e0.state_digest(self.model, self.immutable)
        self.teacher = source.cuda().eval()
        self.teacher_digest = e0.state_digest(self.teacher)
        self.predictions, self.records = {}, []
        self.teacher_capture = None
        self.test_images = next(iter(e0.loader(rows['fit'][:4], parent, 4)))[0].cuda()
        self.restored = e0.make_snn(proxy, parent, classes)
        self.progress('initialized')

    def progress(self, stage, **kw):
        record = {'stage': stage, 'seconds': time.perf_counter() - self.started, **kw}
        self.e0.atomic_json(self.output / 'progress.json', record)
        print(json.dumps(record), flush=True)

    @torch.no_grad()
    def assign(self, biases):
        assert set(biases) == set(self.names)
        for name in self.names:
            value = biases[name]
            assert value.shape == self.bn['runtime_biases'][name].shape
            assert torch.isfinite(value).all() and torch.equal(value, value[:1].expand_as(value))
            self.modules[name].output_bias_correction = value.detach().clone().cuda()

    def snapshot(self):
        return self.e0.runtime_biases(self.model, self.parent, self.classes)

    @torch.no_grad()
    def local_metrics(self):
        if self.teacher_capture is None:
            self.teacher_capture = self.e0.capture(self.teacher, self.rows['diagnostic'], self.parent, self.classes)
        capture = self.e0.capture(self.model, self.rows['diagnostic'], self.parent, self.classes, True)
        results = {}
        for name in self.names:
            a, s = self.teacher_capture[name], capture[name]
            residual = (s['output'].double().mean(0) - a['output'][0].double()) / s['theta'].double()
            results[name] = {'normalized_mse': float(residual.square().mean()),
                             'mean_abs_channel_residual': float(residual.mean(0).abs().mean())}
        return results

    @torch.no_grad()
    def score(self, ident, biases, detail=None, diagnostics=True):
        assert ident not in self.predictions
        self.assign(biases)
        torch.cuda.synchronize(); began = time.perf_counter()
        p = {role: self.e0.evaluate(self.model, self.rows[role], self.parent, True)
             for role in ('fit', 'development')}
        torch.cuda.synchronize(); eval_seconds = time.perf_counter() - began
        base = self.predictions.get('bn-only', p)
        paired = {}
        for role in p:
            assert torch.equal(p[role]['ids'], base[role]['ids'])
            assert torch.equal(p[role]['labels'], base[role]['labels'])
            b = base[role]['predictions'] == p[role]['labels']
            a = p[role]['predictions'] == p[role]['labels']
            paired[role] = {'repair': int((a & ~b).sum()), 'damage': int((b & ~a).sum())}
        before = self.e0.logits(self.model, self.test_images, self.parent, True).cpu()
        saved = {'biases': self.snapshot(), 'parent': self.c['parent_deployment'],
                 'parent_sha256': self.e0.sha256(self.c['parent_deployment']),
                 'thresholds': self.bn['interface_thresholds'], 'detail': detail}
        self.e0.atomic_torch(self.output / ('bias-' + ident + '.pt'), saved)
        reloaded = torch.load(self.output / ('bias-' + ident + '.pt'), map_location='cpu', weights_only=False)
        state = dict(self.bn['state_dict'])
        state.update({n + '.output_bias_correction': v for n, v in reloaded['biases'].items()})
        self.restored.load_state_dict(state, strict=True)
        for n in self.names:
            dict(self.restored.named_modules())[n].v_threshold = reloaded['thresholds'][n]
        after = self.e0.logits(self.restored, self.test_images, self.parent, True).cpu()
        assert torch.equal(before, after), 'Actual bias strict reload differs'
        assert self.e0.state_digest(self.model, self.immutable) == self.initial_digest
        rec = {'id': ident, 'metrics': {k: v['metrics'] for k, v in p.items()},
               'paired_vs_bn': paired, 'strict_reload': True, 'non_bias_frozen': True,
               'evaluation_seconds': eval_seconds, 'detail': detail}
        if diagnostics:
            rec['local_fit_diagnostics'] = self.local_metrics()
        self.predictions[ident] = p; self.records.append(rec)
        self.e0.atomic_json(self.output / 'metrics-progress.json', self.records)
        self.e0.atomic_torch(self.output / 'predictions.pt', self.predictions)
        self.progress('scored', candidate=ident, development=p['development']['metrics'])
        return p

    def anchors(self):
        self.score('bn-only', self.bn['runtime_biases'])
        self.score('ftbc-e1', self.calibrated['runtime_biases'], {'fit_images': 4096, 'eta': .01, 'historical_reference_only_for_256fit_comparisons': True})
        for label, path, key in [('bn-only', self.c['parent_predictions'], 'spatial_snn'),
                                  ('ftbc-e1', self.c['calibrated_predictions'], 'spatial')]:
            old = torch.load(path, map_location='cpu', weights_only=True)[key]
            actual = self.predictions[label]['development']; n = len(actual['ids'])
            assert torch.equal(actual['ids'], old['ids'][:n])
            assert torch.equal(actual['labels'], old['labels'][:n])
            delta = (actual['logits'] - old['logits'][:n]).abs().max().item()
            disagreed = int((actual['predictions'] != old['predictions'][:n]).sum())
            identity = {'id': label, 'max_logit_delta': delta, 'top1_prediction_disagreements': disagreed,
                        'cross_hardware': self.c['allowed_uuid'] != self.parent['allowed_uuid']}
            self.e0.atomic_json(self.output / ('anchor-' + label + '.json'), identity)
            if not identity['cross_hardware']:
                assert torch.equal(actual['logits'], old['logits'][:n])
        self.assign(self.bn['runtime_biases'])

    def finish(self):
        assert self.e0.state_digest(self.teacher) == self.teacher_digest
        torch.cuda.synchronize()
        result = {'status': 'completed', 'mode': self.c['mode'], 'records': len(self.records),
                  'seconds_since_runtime_initialization': time.perf_counter() - self.started,
                  'max_memory_allocated': torch.cuda.max_memory_allocated(),
                  'max_memory_reserved': torch.cuda.max_memory_reserved(),
                  'teacher_unchanged': True, 'non_bias_unchanged': True, 'all_strict_reloads_passed': True,
                  'hardware': self.hardware, 'data_status': 'exploratory_reused_development_only'}
        self.e0.atomic_json(self.output / 'result.json', result)
        self.progress('completed', records=len(self.records))
        return result
