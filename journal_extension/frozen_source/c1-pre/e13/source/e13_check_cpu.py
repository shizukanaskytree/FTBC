"""E13 CPU checks before review/freeze. Never initializes CUDA; reads no images."""
import hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]; P = E.parent
import torch
torch.set_num_threads(4)
sys.path.insert(0, str(P / 'source-snapshot/serial')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from Models.layer import IF
import serial_solver as ss


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


report = {}
g = torch.Generator().manual_seed(20261009)
# 1. Replay equals IF.forward bitwise (5-D and 3-D shapes, random biases, chunking).
cases = 0
for T in [2, 4, 8]:
    for trial in range(15):
        C = 6; m = IF(T=T, L=8, thresh=float(torch.rand(1, generator=g)) * 2 + 0.2)
        x = (torch.randn(T * 5, C, 3, 3, generator=g) * 0.8).float(); b = (torch.randn(T, C, generator=g) * 0.4).float()
        m.time_based_bias = [b[t].clone() for t in range(T)]
        y = m(x).view(T, 5, C, 3, 3); n_if = (y != 0).sum(0).to(torch.uint8)
        assert torch.equal(n_if, ss.replay(x.view(T, 5, C, 3, 3), b, m.thresh.data, chunk=2)); cases += 1
        x2 = (torch.randn(T * 7, C, generator=g)).float(); y2 = m(x2).view(T, 7, C)
        assert torch.equal((y2 != 0).sum(0).to(torch.uint8), ss.replay(x2.view(T, 7, C), b, m.thresh.data, chunk=3)); cases += 1
report['replay_equals_IF_forward_cases'] = cases
# 2. Deferral exact for constant input; offsets->biases sum to zero.
for T in [2, 4, 8]:
    z = torch.linspace(-0.5, 1.5, 8001).float().view(1, -1, 1).expand(T, -1, 1).contiguous(); th = torch.tensor([1.0])
    b = ss.biases_from_offsets(ss.deferral(T), T).float().view(T, 1)
    assert abs(float(ss.biases_from_offsets(ss.deferral(T), T).sum())) < 1e-15
    assert torch.equal(ss.replay(z, b, th), ss.local_twin(z, torch.zeros(T, 1), th)), T
report['deferral_constant_input_exact_T'] = [2, 4, 8]
# 3. Channel mean is monotone non-decreasing in the bias current; solver reaches bracketed targets.
T = 4; x = (torch.randn(T, 16, 3, 4, 4, generator=g) * 0.6 + 0.2).float(); th = torch.tensor([1.0]); b0 = torch.zeros(T, 3)
prev = None
for beta in torch.linspace(-0.5, 0.5, 201):
    mval = ss.channel_counts(ss.replay(x, b0 - beta, th)).double()
    assert prev is None or torch.all(mval >= prev); prev = mval
target = torch.tensor([0.2, 0.45, 0.6], dtype=torch.float64)
beta, info = ss.solve_mean(x, b0, th, target)
quantum = 1.0 / (x.numel() // 3)
assert not info['below_range'].any() and not info['above_range'].any()
assert torch.all(info['normalized_residual'].abs() <= quantum + 1e-12), info['normalized_residual']
assert torch.all(info['normalized_residual'].abs() <= (info['mean_before'] - target).abs() + 1e-15)
far = torch.tensor([0.2, 0.45, 0.9], dtype=torch.float64)       # third target above the reachable range
beta2, info2 = ss.solve_mean(x, b0, th, far)
assert bool(info2['above_range'][2]) and float(beta2[2]) == 0.5 and bool(info2['at_bound'][2])
report['solver_bracketed_within_one_quantum'] = True
report['solver_out_of_range_flagged_and_bounded'] = True
# 4. Schedule fit lowers the local count error on an early-heavy synthetic pattern (typical premature firing).
xe = torch.zeros(T, 64, 2, 2, 2)
xe[0] = 0.9; xe[1] = -0.2; xe[2] = -0.2; xe[3] = 0.1          # strong start, then negative: IF fires early, twin says 1 spike
xe = (xe + 0.05 * torch.randn(xe.shape, generator=g)).float()
off, sched = ss.fit_schedule(xe, th, T)
assert sched['history'][-1]['objective'] <= sched['history'][0]['objective'] <= sched['baseline_IF_objective']
report['schedule_fit_synthetic'] = {'IF': sched['baseline_IF_objective'], 'deferral_init': sched['history'][0]['objective'], 'final': sched['history'][-1]['objective'], 'offsets': off.tolist()}
# 5. Configs and identities.
cfg = {w: json.loads((E / f'config/{w}.json').read_text()) for w in ['sanity', 'main']}
for w, c in cfg.items():
    assert not Path(c['output']).exists(), c['output']
    assert set(c['allowed_uuids']) == {'GPU-08d042a1-efba-a7d5-657e-8543e26f3154', 'GPU-007d796e-235b-fa1b-f840-8387fa0d6894'}
    assert c['f1_n'] in c['ftbc_ns'] and 3 in c['ftbc_ns']
manifest = json.loads(Path(cfg['main']['calibration_export_manifest']).read_text())['families']['serial']
assert sha(cfg['main']['calibration_export']) == manifest['export_sha256'] and manifest['definition_ast_identity'] and not manifest['protected_entrypoint_exported']
spec = importlib.util.spec_from_file_location('e13_cpu_e0', cfg['main']['e0_runner']); e0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e0)
parent = json.loads(Path(cfg['main']['parent_config']).read_text()); roles = e0.validate_config(parent)
assert parent['family'] == 'serial' and parent['timesteps'] == 4 and {k: len(v) for k, v in roles.items()} == {'fit': 4096, 'diagnostic': 64, 'development': 512}
source, classes = e0.source_model(parent)
dep = torch.load(cfg['main']['plain_deployment'], map_location='cpu', weights_only=False)
prox = torch.load(cfg['main']['plain_proxy'], map_location='cpu', weights_only=True)['state_dict']
assert all(not torch.count_nonzero(v) for vals in dep['runtime_biases'].values() for v in vals) and len(dep['runtime_biases']) == 33
src = source.state_dict(); assert set(prox) == set(src) and all(torch.equal(prox[k], src[k]) for k in src)
shared = set(dep['state_dict']) & set(src); assert all(torch.equal(dep['state_dict'][k], src[k]) for k in shared)
e0pred = torch.load(cfg['main']['e0_predictions'], map_location='cpu', weights_only=True)
assert e0pred['plain_snn']['metrics']['correct'] == 67 and e0pred['plain_proxy']['metrics']['correct'] == 377
report['identities'] = {'proxy_equals_source_state': True, 'plain_deployment_equals_source_on_shared': len(shared), 'e0_anchor_counts': [67, 377]}
assert not torch.cuda.is_initialized()
files = [Path(__file__), E / 'source/serial_solver.py', E / 'source/e13_runner.py', E / 'source/e13_accept.py', E / 'config/sanity.json', E / 'config/main.json', E / 'EXPERIMENT_PLAN.md']
report.update(passed=True, cuda_initialized=False, images_read=False, files_sha256={str(p): sha(p) for p in files})
(E / 'cache/cpu-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n'); print(json.dumps(report, indent=1, default=str))
