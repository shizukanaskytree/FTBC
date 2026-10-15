"""CPU checks of the E19 ablation variants (EXPERIMENT_PLAN.md section 3) on small conv and linear IF networks built from the frozen serial
source snapshot's IF class, with the released update from the E1 export. No CUDA. Writes cache/check-variants.json.
  1. With all corrections zero, one FA call changes the stored biases exactly as one call of the released update (bitwise).
  2. With random nonzero corrections, FA's step-0 spikes are the deployed IF's step-0 spikes: the stored step-0 bias after the call equals
     alpha * ChMean(deployed step-0 output - ANN output) + b_0 (bitwise). This tests FA's only change from the released update.
  3. The membrane trajectory advanced inside one FD call equals the deployed IF.forward with the corrections after that call (bitwise),
     for 4D (conv) and 2D (linear) interfaces.
  4. A second FD call (nonzero corrections, new batch): each updated b_t equals alpha * ChMean(spike_t - ANN output) + b_before[t], where spike_t
     is the deployed IF's step-t output with the mixed corrections (steps before t after the call, step t before it); the trajectory check of 3
     holds after the second call as well (bitwise).
  5. Behaviour: repeated FD calls on one batch shrink the deployed per-step channel-mean residual (sign of the update)."""
import copy, hashlib, importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch, torch.nn as nn
E = Path(__file__).resolve().parents[1]; P = E.parent
torch.set_num_threads(8); torch.manual_seed(20261010)
cfg = json.loads((E / 'config/main.json').read_text())
sys.path.insert(0, str(P / 'source-snapshot/serial'))
IF = importlib.import_module('Models.layer').IF


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


inherited = load('chk_export', cfg['calibration_export']); assert inherited.IF is IF
V = load('chk_variants', E / 'source/ftbc_variants.py').make(inherited)


class Toy(nn.Module):
    def __init__(self, kind, T, L=4):
        super().__init__()
        self.kind, self.T = kind, T
        if kind == 'conv':
            self.f1 = nn.Conv2d(3, 6, 3, padding=1); self.f2 = nn.Conv2d(6, 5, 3, padding=1)
        else:
            self.f1 = nn.Linear(12, 7); self.f2 = nn.Linear(7, 5)
        self.a1 = IF(T=T, L=L, thresh=1.0); self.a2 = IF(T=T, L=L, thresh=1.2)

    def set_T(self, T):
        self.T = T
        for m in [self.a1, self.a2]:
            m.T = T; m.expand.T = T; m.merge.T = T

    def forward(self, x):
        if self.T > 0:
            x = x.unsqueeze(0).repeat(self.T, *[1] * x.dim()).flatten(0, 1)
        y = self.a2(self.f2(self.a1(self.f1(x))))
        return y.view(self.T, -1, *y.shape[1:]).mean(0) if self.T > 0 else y


def pair(kind, T):
    ann = Toy(kind, 0).eval(); snn = copy.deepcopy(ann); snn.set_T(T)
    for m, f in [(snn.a1, snn.f1), (snn.a2, snn.f2)]:
        C = f.out_channels if kind == 'conv' else f.out_features
        m.time_based_bias = [torch.zeros(C) for _ in range(T)]
    return ann, snn


def data(kind, n=8):
    return 2 * torch.randn(n, 3, 6, 6) if kind == 'conv' else 2 * torch.randn(n, 12)


def captured_input(snn, module, x):
    return inherited.GetLayerInputOutput(snn, module)(x)[0]


def chmean(r):
    return r.mean(dim=[0, 2, 3]) if r.dim() == 4 else r.mean(dim=0)


report = {}
with torch.no_grad():
    for kind in ['conv', 'linear']:
        for T in [2, 4]:
            x = data(kind)
            for layer in ['a1', 'a2']:
                ann, snn = pair(kind, T)
                A, B = copy.deepcopy(snn), copy.deepcopy(snn)
                inherited.bias_corr_step_by_step(ann, getattr(ann, layer), A, getattr(A, layer), T, x, curr_t_alpha=0.4)
                V['FA'](ann, getattr(ann, layer), B, getattr(B, layer), T, x, curr_t_alpha=0.4)
                fa_equal = all(torch.equal(p, q) for p, q in zip(getattr(A, layer).time_based_bias, getattr(B, layer).time_based_bias))
                assert fa_equal and any(torch.count_nonzero(p) for p in getattr(A, layer).time_based_bias), (kind, T, layer)
                Cm = copy.deepcopy(snn); rec = []; mod = getattr(Cm, layer)
                V['FD'](ann, getattr(ann, layer), Cm, mod, T, x, curr_t_alpha=0.4, record=rec)
                inp = captured_input(Cm, mod, x)                     # the same input as inside the call (only upstream layers precede it)
                deployed = mod(inp).view(T, -1, *inp.shape[1:])
                fd_equal = torch.equal(torch.stack(rec), deployed)
                assert fd_equal, (kind, T, layer)
                # 2. FA with random nonzero corrections: step-0 update from the deployed step-0 spikes
                ann_out = inherited.GetLayerInputOutput(ann, getattr(ann, layer))(x)[1]
                Fm = copy.deepcopy(snn); fmod = getattr(Fm, layer); fmod.time_based_bias = [0.3 * torch.randn_like(b) for b in fmod.time_based_bias]
                b_before = [b.clone() for b in fmod.time_based_bias]; finp = captured_input(Fm, fmod, x)
                dep0 = fmod(finp).view(T, -1, *finp.shape[1:])[0]
                V['FA'](ann, getattr(ann, layer), Fm, fmod, T, x, curr_t_alpha=0.4)
                fa_nonzero = torch.equal(fmod.time_based_bias[0], 0.4 * chmean(dep0 - ann_out) + b_before[0])
                assert fa_nonzero and not torch.equal(fmod.time_based_bias[0], b_before[0]), (kind, T, layer)
                # 4. second FD call on a new batch, starting from the corrections of the first call
                x2 = data(kind); ann_out2 = inherited.GetLayerInputOutput(ann, getattr(ann, layer))(x2)[1]
                b_before = [b.clone() for b in mod.time_based_bias]; rec2 = []
                V['FD'](ann, getattr(ann, layer), Cm, mod, T, x2, curr_t_alpha=0.4, record=rec2)
                b_after = [b.clone() for b in mod.time_based_bias]; inp2 = captured_input(Cm, mod, x2)
                fd_second = True
                for t in range(T):
                    mod.time_based_bias = [b.clone() for b in b_after[:t] + [b_before[t]] + b_after[t + 1:]]
                    spk = mod(inp2).view(T, -1, *inp2.shape[1:])[t]
                    fd_second &= torch.equal(b_after[t], 0.4 * chmean(spk - ann_out2) + b_before[t])
                mod.time_based_bias = [b.clone() for b in b_after]
                fd_second_traj = torch.equal(torch.stack(rec2), mod(inp2).view(T, -1, *inp2.shape[1:]))
                assert fd_second and fd_second_traj, (kind, T, layer)
                report[f'{kind}-T{T}-{layer}'] = {'fa_first_call_equals_released': fa_equal, 'fa_nonzero_step0_update_from_deployed_spikes': fa_nonzero,
                                                  'fd_trajectory_equals_deployed_IF': fd_equal, 'fd_second_call_updates_from_mixed_deployed_spikes': fd_second,
                                                  'fd_second_call_trajectory_equals_deployed_IF': fd_second_traj}
    # 3. behaviour of repeated FD calls on one batch (first interface; ANN target from the source network)
    ann, snn = pair('conv', 4); x = data('conv', 16); mod = snn.a1
    target = inherited.GetLayerInputOutput(ann, ann.a1)(x)[1].mean((0, 2, 3))
    def residual():
        inp = captured_input(snn, mod, x); out = mod(inp).view(4, -1, *inp.shape[1:]).mean((1, 3, 4))
        return float((out - target.view(1, -1)).abs().mean())
    hist = [residual()]
    for _ in range(20):
        V['FD'](ann, ann.a1, snn, mod, 4, x, curr_t_alpha=0.4); hist.append(residual())
    assert hist[-1] < hist[0], hist
    report['fd_repeated_calls_mean_abs_step_residual'] = {'before': hist[0], 'after_1': hist[1], 'after_20': hist[-1]}
report['passed'] = True
report['cuda_initialized'] = torch.cuda.is_initialized(); assert not report['cuda_initialized']
report['files_sha256'] = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [E / 'source/ftbc_variants.py', Path(__file__), cfg['calibration_export']]}
(E / 'cache').mkdir(exist_ok=True); (E / 'cache/check-variants.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=1))
