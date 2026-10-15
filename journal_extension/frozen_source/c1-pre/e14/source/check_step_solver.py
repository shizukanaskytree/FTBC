"""CPU check of the E14 per-step solver: bitwise agreement with IF.forward, per-step residuals, and per-step monotonicity.
Asserted: with steps 1..t-1 fixed, the step-t count is non-increasing in b_t (b_t does not touch earlier states; FP32 ops are monotone).
Descriptive only: later single steps are not monotone in b_t (one spike fewer at t leaves the membrane theta higher afterwards), which
is why the solver goes in time order; the cumulative count is monotone in exact arithmetic (v + N*theta equals cumulative input and at
most one spike per step), FP32 violations near threshold are possible and are only counted here."""
import json, sys, hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
H = Path(__file__).resolve().parents[1]; P = H.parent
sys.path.insert(0, str(P / 'source-snapshot/serial')); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(P / 'e13/source'))
from Models.layer import IF
import step_solver as ps, serial_solver as ss
g = torch.Generator().manual_seed(20261009); cases = 0; within = 0; total_ch = 0; mono_checks = 0; nonmono_total_seen = 0; later_step_increase = 0
for T in [2, 4, 8]:
    for trial in range(8):
        C = 5; thre = torch.tensor([float(torch.rand(1, generator=g)) * 1.5 + 0.4]); th = float(thre)
        x = (torch.randn(T, 12, C, 3, 3, generator=g) * 0.6 + 0.15).float(); target = (torch.rand(C, generator=g) * 0.4 * th).double()
        b, info, total = ps.solve_per_step(x, thre, target, chunk=5)
        m = IF(T=T, L=8, thresh=th); m.time_based_bias = [b[t].clone() for t in range(T)]
        y = m(x.reshape(T * 12, C, 3, 3)).view(T, 12, C, 3, 3)
        assert torch.equal((y != 0).sum(0).to(torch.uint8), total) and torch.equal(ss.replay(x, b, thre), total)
        for t in range(T):
            assert torch.equal((y[t] != 0).double().sum((0, 2, 3)) * th / (12 * 9), info['steps'][t]['mean_after'].double())
            r = info['steps'][t]['normalized_residual'].abs()
            within += int(((r <= 1 / 108 + 1e-12) | info['steps'][t]['above_range'] | info['steps'][t]['below_range']).sum()); total_ch += C
            # per-step monotonicity in b_t with the solved earlier steps fixed
            mem = (0.5 * thre).expand(x.shape[1:]).clone()
            for tau in range(t):
                mem, _ = ps._advance(mem, x[tau], b[tau], thre, 5)
            prev = None
            for v in torch.linspace(-0.6 * th, 0.6 * th, 241):
                cnt = ps._step_counts(mem, x[t], torch.full((C,), float(v)), thre, 5)
                assert prev is None or torch.all(cnt <= prev); prev = cnt; mono_checks += 1
        # descriptive: raising b_t can add spikes at later single steps; cumulative count increases are FP-only exceptions
        for t in range(T - 1):
            prev_total = prev_later = None
            for v in torch.linspace(-0.6 * th, 0.6 * th, 61):
                bb = b.clone(); bb[t] = float(v); mem = (0.5 * thre).expand(x.shape[1:]).clone(); later = []
                for tau in range(T):
                    mem, s = ps._advance(mem, x[tau], bb[tau], thre, 5)
                    if tau > t: later.append(s.to(torch.int64).sum((0, 2, 3)))
                later = torch.stack(later); total = ss.channel_counts(ss.replay(x, bb, thre))
                if prev_total is not None:
                    nonmono_total_seen += int(torch.any(total > prev_total)); later_step_increase += int(torch.any(later > prev_later))
                prev_total, prev_later = total, later
        cases += 1
assert not torch.cuda.is_initialized()
out = {'passed': True, 'cases': cases, 'per_step_residual_within_quantum_or_flagged': f'{within}/{total_ch}', 'per_step_monotonicity_checks': mono_checks,
       'later_single_step_increase_events_when_raising_b_t': later_step_increase, 'cumulative_count_increase_events_when_raising_b_t': nonmono_total_seen,
       'scope': 'asserted: step-t count non-increasing in b_t with earlier steps fixed; descriptive: later single steps and cumulative count', 'cuda_initialized': False,
       'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(ps.__file__)]}}
(H / 'cache/step-solver-check.json').write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=1))
