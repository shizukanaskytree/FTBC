"""E29 CPU unit checks (EXPERIMENT_PLAN.md section 6 item 7): the chunked and paired steps against the released step on small synthetic
networks built from the released IF neuron, through the released bias_corr_model. Never initializes CUDA. Usage: python3 -I unit_checks.py
<export_dir> <qcfs_dir>; prints a JSON report and exits nonzero on failure.
  (1) chunked against released, end to end: float64 within 1e-10, float32 within 1e-5; update calls and forward counts as forwards_per_call;
  (2) the per-call pairing (initial review B1): on every call the fingerprints of the REL and CHB captures are equal; with chunk = batch the
      per-call update is bitwise equal in float64 and within 1e-6 theta in float32; CHS against CHB at the first interface within 1e-10 theta
      in float64;
      the paired run ends bitwise on the unpaired released run's corrections;
  (3) the fingerprint: equal for equal tensors, different after a one-ulp change, a swap of two elements, or a different shape;
  (4) the export is restored after an error inside release_fit and inside a paired call."""
import contextlib, io, json, sys
from pathlib import Path
EXPORT, QCFS = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(QCFS)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import importlib.util
import torch, torch.nn as nn
torch.cuda.is_available = lambda: False


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


inherited = load('e29_unit_export', EXPORT / 'serial.py')
RC = load('e29_unit_release_chunked', Path(__file__).resolve().parents[1] / 'source/release_chunked.py')
IF = inherited.IF
RELEASED, GET = inherited.bias_corr_step_by_step, inherited.GetLayerInputOutput


class TimeRepeat(nn.Module):
    """Stand-in for the released time expansion: the batch is repeated T times, time outermost, then merged."""
    def __init__(self, T):
        super().__init__(); self.T = T

    def forward(self, x):
        return x.unsqueeze(0).repeat(self.T, *([1] * x.dim())).flatten(0, 1)


class Flat(nn.Module):
    def forward(self, x):
        return x.flatten(1)


def pair(T, L, dtype, seed):
    """ANN (QCFS activations, T=0) and SNN (IF neurons, T steps) with identical names and weights: two conv interfaces and one vector
    interface, as in the released models."""
    g = torch.Generator().manual_seed(seed)
    conv1, conv2, fc = nn.Conv2d(3, 6, 3, padding=1), nn.Conv2d(6, 5, 3, padding=1), nn.Linear(5 * 8 * 8, 7)
    for m in (conv1, conv2, fc):
        for p in m.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.4
    def build(spiking):
        k = T if spiking else 0
        return nn.Sequential(TimeRepeat(T) if spiking else nn.Identity(), conv1, IF(T=k, L=L, thresh=1.5), conv2, IF(T=k, L=L, thresh=1.2),
                             Flat(), fc, IF(T=k, L=L, thresh=2.0)).to(dtype)
    return build(False), build(True)


def init_biases(snn, T, dtype):
    channels = {'2': 6, '4': 5, '7': 7}
    for n, m in snn.named_modules():
        if isinstance(m, IF):
            m.time_based_bias = [torch.zeros(channels[n], dtype=dtype) for _ in range(T)]


def run(step, T, L, dtype, B, n_batches, seed=0, alpha=0.4, net=None):
    ann, snn = net or pair(T, L, dtype, seed); init_biases(snn, T, dtype)
    g = torch.Generator().manual_seed(seed + 1)
    batches = [(torch.randn(B, 3, 8, 8, generator=g).to(dtype) * 1.5, torch.zeros(B, dtype=torch.long)) for _ in range(n_batches)]
    names = [n for n, m in snn.named_modules() if isinstance(m, IF)]
    with contextlib.redirect_stdout(io.StringIO()):       # the released bias_corr_model prints progress lines
        count = RC.release_fit(inherited, ann, snn, T, batches, alpha, step, names)
    return {n: torch.stack([b.detach().clone() for b in m.time_based_bias]) for n, m in snn.named_modules() if isinstance(m, IF)}, count, snn


def max_diff(a, b):
    return max(float((a[n] - b[n]).abs().max()) for n in a)


report, ok = [], True
NB = 3
# (1)
for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 1e-5)]:
    for T in (2, 4, 8):
        for B, chunk in [(12, 5), (12, 1), (12, 12), (10, 3)]:
            ref, c_ref, _ = run(RELEASED, T, 8, dtype, B, NB)
            got, c_got, _ = run(RC.chunked_step(inherited, chunk), T, 8, dtype, B, NB)
            d = max_diff(ref, got); changed = max(float(v.abs().max()) for v in ref.values())
            counts = c_ref['updates'] == c_got['updates'] and c_ref['ann_forwards'] == c_ref['snn_forwards'] == 3 * NB * RC.forwards_per_call('REL', B, chunk) \
                and c_got['ann_forwards'] == c_got['snn_forwards'] == 3 * NB * RC.forwards_per_call('CHS', B, chunk)
            passed = d <= tol and counts and changed > 0
            report.append({'check': 'chunked vs released', 'dtype': str(dtype), 'T': T, 'B': B, 'chunk': chunk, 'max_abs_diff': d, 'tol': tol,
                           'max_abs_update': changed, 'forwards': [c_ref['ann_forwards'], c_got['ann_forwards']], 'passed': passed})
            ok &= passed
# (2)
for dtype, tol_b in [(torch.float64, 0.0), (torch.float32, 1e-6)]:
    for T in (2, 4):
        for B, chunk in [(12, 5), (12, 4)]:
            ref, _, _ = run(RELEASED, T, 8, dtype, B, NB)
            log = []; net = pair(T, 8, dtype, 0)
            ids = {id(m): n for n, m in net[1].named_modules() if isinstance(m, IF)}
            step = RC.paired_step(inherited, [('REL', RELEASED, True), ('CHB', RC.chunked_step(inherited, B), True), ('CHS', RC.chunked_step(inherited, chunk), False)], log, ids)
            got, c_got, _ = run(step, T, 8, dtype, B, NB, net=net)
            fp_equal = all(r['fingerprints']['REL'] == r['fingerprints']['CHB'] for r in log)
            chb = max(r['compare']['CHB_vs_REL']['max'] for r in log)
            first = max(r['compare']['CHS_vs_CHB']['max'] for r in log if r['interface'] == '2')
            trajectory = all(torch.equal(ref[n], got[n]) for n in ref)
            counts = c_got['ann_forwards'] == c_got['snn_forwards'] == 3 * NB * RC.forwards_per_call('PAIR', B, chunk) and len(log) == 3 * NB
            shapes = all(r['snn_input_shape'][0] == T * B for r in log)
            passed = fp_equal and chb <= tol_b and (first <= 1e-10 or dtype == torch.float32) and trajectory and counts and shapes \
                and inherited.GetLayerInputOutput is GET and inherited.bias_corr_step_by_step is RELEASED
            report.append({'check': 'paired per call', 'dtype': str(dtype), 'T': T, 'B': B, 'chunk': chunk, 'calls': len(log), 'fingerprints_equal': fp_equal,
                           'max_CHB_vs_REL_over_theta': chb, 'tol_CHB': tol_b, 'max_CHS_vs_CHB_first_interface': first,
                           'max_CHS_vs_REL_all': max(r['compare']['CHS_vs_REL']['max'] for r in log), 'ends_on_released_trajectory': trajectory,
                           'forwards': c_got['ann_forwards'], 'passed': passed})
            ok &= passed
# (3)
x = torch.randn(4, 3, 5, 5); y = x.clone(); fx = RC.fingerprint(x)
z = x.clone(); z.view(-1)[17] = torch.nextafter(z.view(-1)[17], torch.tensor(float('inf')))
w = x.clone(); w.view(-1)[[3, 40]] = w.view(-1)[[40, 3]]
x64 = torch.randn(3, 7, dtype=torch.float64); z64 = x64.clone(); z64[1, 2] = torch.nextafter(z64[1, 2], torch.tensor(float('inf'), dtype=torch.float64))
passed = fx == RC.fingerprint(y) and fx != RC.fingerprint(z) and fx != RC.fingerprint(w) and fx != RC.fingerprint(x.reshape(4, 75)) \
    and RC.fingerprint(x64) == RC.fingerprint(x64.clone()) and RC.fingerprint(x64) != RC.fingerprint(z64) \
    and RC.fingerprint(x, words=7) == RC.fingerprint(x, words=1 << 22)
report.append({'check': 'fingerprint', 'passed': passed}); ok &= passed
# (4)
ann, snn = pair(2, 8, torch.float32, 3); init_biases(snn, 2, torch.float32)
def boom(*a, **k):
    raise RuntimeError('injected')
restored = False
try:
    with contextlib.redirect_stdout(io.StringIO()):
        RC.release_fit(inherited, ann, snn, 2, [(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))], 0.4, boom, ['2', '4', '7'])
except RuntimeError:
    restored = inherited.bias_corr_step_by_step is RELEASED and not snn._forward_pre_hooks and not ann._forward_pre_hooks
report.append({'check': 'restore after an error in release_fit', 'passed': restored}); ok &= restored
ann, snn = pair(2, 8, torch.float32, 4); init_biases(snn, 2, torch.float32)
ids = {id(m): n for n, m in snn.named_modules() if isinstance(m, IF)}
def failing_chunk(*a, **k):
    assert inherited.GetLayerInputOutput is not GET                 # the fingerprinting class is installed while CHB runs
    raise RuntimeError('injected inside CHB')
step = RC.paired_step(inherited, [('REL', RELEASED, True), ('CHB', failing_chunk, True)], [], ids)
restored = False
try:
    with contextlib.redirect_stdout(io.StringIO()):
        RC.release_fit(inherited, ann, snn, 2, [(torch.randn(4, 3, 8, 8), torch.zeros(4, dtype=torch.long))], 0.4, step, ['2', '4', '7'])
except RuntimeError as exc:
    restored = 'inside CHB' in str(exc) and inherited.GetLayerInputOutput is GET and inherited.bias_corr_step_by_step is RELEASED
report.append({'check': 'restore after an error inside a paired call', 'passed': restored}); ok &= restored
print(json.dumps({'passed': ok, 'cases': report}, indent=1))
sys.exit(0 if ok else 1)
