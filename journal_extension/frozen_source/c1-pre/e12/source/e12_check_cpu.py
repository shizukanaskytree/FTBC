"""E12 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E.parent / 'e7/source'))
import run_diagnostic as e7run
import solver
from common import prepare, load
import torch
import e12_relative as rel
from e12_runner import derived, bn_state_keys
torch.set_num_threads(4)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main']}
for w, c in cfg.items():
    assert not Path(c['output']).exists(), 'output already exists: ' + c['output']
    for branch in ['p1', 'p0']:
        report[f'prepare_{w}_{branch}'] = prepare(derived(c, branch, c['output']), cpu=True)
    assert c['e7_reference']['output'].endswith(f'{w}-v1')
c = cfg['main']

# Frozen E7 code identity: the solver/runner imported here are the files frozen in E7.
e7freeze = json.loads((E.parent / 'e7/protocol-freeze.json').read_text())['files_sha256']
for name in ['solver.py', 'run_diagnostic.py']:
    path = str(E.parent / 'e7/source' / name)
    assert sha(path) == e7freeze[path], name
assert Path(solver.__file__).resolve() == (E.parent / 'e7/source/solver.py').resolve()
assert e7run.solve is solver.solve and e7run.counts is solver.counts
report['e7_solver_and_runner_unchanged'] = True

# P0/P1/E1/source identity on CPU (the GPU runner repeats this on the loaded models).
parent = json.loads(Path(c['parent_config']).read_text())
e0 = load('e12_cpu_e0', c['parent_runner'])
source, classes = e0.source_model(parent)
src = source.state_dict()
dep = {b: torch.load(c[b]['parent_deployment'], map_location='cpu', weights_only=False) for b in ['p0', 'p1']}
cal = {b: torch.load(c[b]['calibrated_deployment'], map_location='cpu', weights_only=False) for b in ['p0', 'p1']}
s0, s1 = dep['p0']['state_dict'], dep['p1']['state_dict']
assert set(s0) == set(s1) and dep['p0']['interface_thresholds'] == dep['p1']['interface_thresholds']
differ = {k for k in s0 if not torch.equal(s0[k], s1[k])}
bn_suffix = ('.weight', '.bias', '.running_mean', '.running_var', '.num_batches_tracked')
bn_modules = {n for n, m in source.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
bn = {k for k in s0 if k.rsplit('.', 1)[0] in bn_modules}
assert bn == bn_state_keys(source) and len(bn) == 180 and differ == bn, sorted(differ ^ bn)[:5]
shared = set(s0) & set(src)
assert bn <= shared and len(shared) == 218 and all(torch.equal(s0[k], src[k]) for k in shared), 'P0 differs from source checkpoint'
for b in ['p0', 'p1']:
    keys = [k for k in s0 if not k.endswith('output_bias_correction')]
    assert all(torch.equal(dep[b]['state_dict'][k], cal[b]['state_dict'][k]) for k in keys)
    assert all(not torch.count_nonzero(v) for v in dep[b]['runtime_biases'].values())
report['states'] = {'state_tensors': len(s0), 'bn_tensors': len(bn), 'differing_tensors': len(differ), 'p0_source_shared_tensors': len(shared),
                    'only_and_all_bn_differ': differ == bn, 'p0_equals_source_on_shared_keys': True, 'thresholds_equal': True,
                    'e1_non_bias_equal_parents': True, 'zero_parent_biases': True}

# Reference artifacts exist and E7/E5 KL bias identity holds.
for w, cc in cfg.items():
    ref = {k: Path(cc['e7_reference']['output']) / v for k, v in cc['e7_reference']['files'].items()}
    kl = torch.load(cc['kl_bias'], map_location='cpu', weights_only=False)
    e7kl = torch.load(ref['bias_kl'], map_location='cpu', weights_only=False)
    assert all(torch.equal(kl['biases'][n], e7kl['biases'][n]) for n in kl['biases'])
    assert kl['parent_sha256'] == sha(cc['p1']['parent_deployment'])
    t = torch.load(ref['targets'], map_location='cpu', weights_only=False)
    assert len(t['ids']) == cc['fit_count'] and all(v.dtype == torch.float64 for v in t['means'].values())
    p = torch.load(ref['predictions'], map_location='cpu', weights_only=False)
    assert {'bn-only', 'kl-e4', 'finite-solve'} <= set(p) and len(p['bn-only']['development']['ids']) == cc['dev_count']
report['references_present'] = True

# Finite examples of the pre-registered functions.
th = {'L1': 1.0, 'L2': 2.0}
a = {'L1': torch.tensor([.5, .4, .3], dtype=torch.float64), 'L2': torch.tensor([1., .2], dtype=torch.float64)}
p0 = {'L1': torch.tensor([.3, .4, .5], dtype=torch.float64), 'L2': torch.tensor([.6, .2], dtype=torch.float64)}
same = {n: a[n].clone() for n in a}                       # Delta == e  -> rho=cos=kappa=1, R1 applies
d, m = rel.descriptors(a, p0, same, th)
assert abs(d['global_theta_normalized']['rho'] - 1) < 1e-15 and abs(d['global_theta_normalized']['cos'] - 1) < 1e-15
assert d['rules']['R1_double_correction_applies'] and not d['rules']['R2_low_information_applies']
assert all(torch.equal(m[n], a[n] + (same[n] - p0[n])) for n in a)
zero = {n: p0[n].clone() for n in a}                       # Delta == 0 -> rho=0, cos undefined, kappa=0, R2 applies
d, _ = rel.descriptors(a, p0, zero, th)
assert d['global_theta_normalized']['rho'] == 0 and d['global_theta_normalized']['cos'] is None and d['rules']['R2_low_information_applies']
exact = {n: a[n].clone() for n in a}                       # e == 0 and Delta == 0 on every layer (p0 = a = p1)
d, _ = rel.descriptors(exact, exact, same, th)
assert d['layers']['L1']['rho'] is None and d['layers']['L1']['kappa'] is None and d['layers']['L1']['cos'] is None
far = {'L1': torch.tensor([1.5, .4, .3], dtype=torch.float64), 'L2': torch.tensor([1., .2], dtype=torch.float64)}
d, m = rel.descriptors(a, p0, far, th)                     # m_L1[0] = .5 + 1.2 > theta
assert d['layers']['L1']['m_outside_0_theta'] == 1 and d['m_outside_0_theta'] == 1
ctrl = {'bn-only': {'correct': 362, 'ce': 1.43}, 'kl-e4': {'correct': 362, 'ce': 1.25}, 'Cabs': {'correct': 357, 'ce': 1.28}, 'Transfer': {'correct': 350, 'ce': 1.30}}
assert rel.decide({**ctrl, 'Crel': {'correct': 365, 'ce': 1.25}})['exploratory_rule_passed']
assert not rel.decide({**ctrl, 'Crel': {'correct': 364, 'ce': 1.0}})['exploratory_rule_passed']
assert not rel.decide({**ctrl, 'Crel': {'correct': 380, 'ce': 1.2500001}})['exploratory_rule_passed']
assert 'passed' not in rel.decide({**ctrl, 'Crel': {'correct': 365, 'ce': 1.25}})
r1 = rel.rule_r1({'rules': {'R1_double_correction_applies': True}}, {'Crel': {'ce': 1.3, 'correct': 350}, 'Cabs': {'ce': 1.28, 'correct': 357}})
assert r1['held'] and not rel.rule_r1({'rules': {'R1_double_correction_applies': False}}, {})['applies']
assert rel.binomial_two_sided(5, 5) == 1.0 and abs(rel.binomial_two_sided(10, 0) - 2 / 1024) < 1e-15 and rel.binomial_two_sided(0, 0) is None
lab = torch.tensor([0, 1, 2, 3]); x = torch.tensor([0, 1, 9, 9]); y = torch.tensor([0, 9, 2, 8])
pr = rel.paired(lab, x, y); assert pr['x_only_correct'] == 1 and pr['y_only_correct'] == 1 and pr['prediction_disagreements'] == 3
mu = {'p0-zero': p0, 'bn-only': same, 'C0': a}
mu['Crel'] = {n: a[n] + (same[n] - p0[n]) for n in a}; mu['Transfer'] = a; mu['Cabs'] = a
ed = rel.edev(mu, th)
assert ed['Crel']['channel_equal'] < 1e-15 and ed['Transfer']['channel_equal'] > 0
b = {'L1': torch.ones(2, 1, 3, 1, 1), 'L2': torch.zeros(2, 1, 2, 1, 1)}
g = rel.geometry(b, b, th)
assert g['layers']['L1']['relative_distance'] == 0 and abs(g['layers']['L1']['cos'] - 1) < 1e-15 and g['layers']['L2']['relative_distance'] is None
assert rel.bias_summary(b, th)['zero_channels'] == 2
counts = torch.randint(0, 1000, (5,), generator=torch.Generator().manual_seed(20261009))
assert torch.equal(rel.means_from_counts({'L': {'count': counts, 'theta': 1.7, 'observations_per_channel': 999}})['L'], counts.double() * 1.7 / 999)

# Solver-execution identity on a finite single-layer example with the original E7 solver:
# (mu_rel - mu_src) - (p1 - p0) == theta * (r_rel - r_src) up to FP64 rounding.
gen = torch.Generator().manual_seed(1009)
base0 = torch.randn(2, 6, 4, 3, 3, generator=gen); base1 = 0.8 * base0 + 0.1
theta = 1.0
mean = lambda base, dl: solver.counts(base, dl, theta).double() * theta / (base.numel() // base.shape[2])
pp0, pp1 = mean(base0, torch.zeros(4)), mean(base1, torch.zeros(4))
aa = torch.tensor([.31, .12, .55, .02], dtype=torch.float64)
d0, i0 = solver.solve(base0, aa, theta); mm = aa + (pp1 - pp0); dr, ir = solver.solve(base1, mm, theta)
lhs = (mean(base1, dr) - mean(base0, d0)) - (pp1 - pp0)
rhs = theta * (ir['normalized_residual'] - i0['normalized_residual'])
assert torch.allclose(lhs, rhs, rtol=0, atol=1e-12)
report['fit_identity_is_solver_execution_check'] = float((lhs - rhs).abs().max())

assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/e12_runner.py', E / 'source/e12_relative.py', E / 'source/e12_accept.py', E / 'source/e12_freeze_submit.py',
         E / 'config/sanity.json', E / 'config/main.json', E / 'EXPERIMENT_PLAN.md', E / 'summarize.py', E / 'record_costs.py', E / 'verify_preservation.py']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache').mkdir(exist_ok=True)
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print(json.dumps({k: v for k, v in report.items() if not k.startswith('prepare_')}, indent=1, default=str))
