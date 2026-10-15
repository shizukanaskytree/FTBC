"""E6 experimental hard parallel neuron, with smooth calibration-tangent backward.

No claim that the backward is the derivative of the discontinuous hard solver.
The smooth mode exists solely to check the interior-root mathematical identity.
"""
import torch

REDUCE = (0, 1, 3, 4)


def base_potential(x, T, theta):
    z = x.reshape(T, -1, *x.shape[1:]).mean(0)
    rank = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
    factor = (T / rank).reshape(T, 1, 1, 1, 1)
    initial = (0.5 * theta / rank).reshape(T, 1, 1, 1, 1)
    return factor * z + initial


def response(base, delta, theta, tau, soft=False):
    p = (base + delta.reshape(1, 1, -1, 1, 1)) - theta
    return theta * (torch.sigmoid(p / (tau * theta)) if soft else (p >= 0).to(p.dtype))


@torch.no_grad()
def solve(base, target, theta, tau=.1, steps=32, soft=False, bound=.5):
    """Channel bisection; choose an endpoint or zero, no exact hard-root promise."""
    lo = torch.full_like(target, -bound * theta)
    hi = torch.full_like(target, bound * theta)
    mean = lambda d: response(base, d, theta, tau, soft).mean(REDUCE)
    lower, upper = mean(lo), mean(hi)
    bracket = (lower < target) & (target < upper)
    for _ in range(steps):
        mid = (lo + hi) / 2
        less = mean(mid) < target
        lo = torch.where(less, mid, lo)
        hi = torch.where(less, hi, mid)
    # Complete deterministic tie rule: residual, |delta|, signed delta.
    candidates = [lo, hi, torch.zeros_like(lo),
                  torch.full_like(lo, -bound * theta), torch.full_like(lo, bound * theta)]
    best = candidates[0]; err = (mean(best) - target).abs()
    for cand in candidates[1:]:
        e = (mean(cand) - target).abs()
        take = (e < err) | ((e == err) & ((cand.abs() < best.abs()) |
                  ((cand.abs() == best.abs()) & (cand < best))))
        best, err = torch.where(take, cand, best), torch.where(take, e, err)
    interior = bracket & (best.abs() < bound * theta)
    return best, interior, err


class CalibratedResponse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, delta, active, T, theta, tau, compensate, soft, trace=None):
        base = base_potential(x, T, theta)
        p = (base + delta.reshape(1, 1, -1, 1, 1)) - theta
        sig = torch.sigmoid(p / (tau * theta))
        ctx.save_for_backward(sig, active)
        ctx.T, ctx.theta, ctx.tau = T, theta, tau
        ctx.compensate, ctx.shape = compensate, x.shape
        ctx.trace, ctx.argument_count = trace, len(ctx.needs_input_grad)
        y = theta * (sig if soft else (p >= 0).to(p.dtype))
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        sig, active = ctx.saved_tensors
        T, theta, tau = ctx.T, ctx.theta, ctx.tau
        g = grad_output.reshape(T, -1, *grad_output.shape[1:])
        w = sig * (1 - sig) / (tau * theta)
        num, den = (g * w).sum(REDUCE), w.sum(REDUCE)
        lam = torch.zeros_like(num)
        if ctx.compensate:
            use = active & (w.mean(REDUCE) >= 1e-6)
            average = g.mean(REDUCE) if ctx.compensate == 'center' else num / den.clamp_min(torch.finfo(w.dtype).tiny)
            lam = torch.where(use, average, lam)
        rank = torch.arange(1, T + 1, device=g.device, dtype=g.dtype)
        dz = (theta * (T / rank).reshape(T, 1, 1, 1, 1) * w *
              (g - lam.reshape(1, 1, -1, 1, 1))).sum(0)
        dx = (dz / T).unsqueeze(0).expand(T, *dz.shape).reshape(ctx.shape)
        # Free-delta control; calibrated deltas are detached and need no gradient.
        dd = theta * num
        if ctx.trace is not None:
            a = (T / rank).reshape(T,1,1,1,1) * w
            raw = theta * (a * g).sum(REDUCE)
            correction = -theta * lam * a.sum(REDUCE)
            ctx.trace.update(local_raw_beta=raw.detach().cpu(),
                             local_correction_beta=correction.detach().cpu(),
                             active=active.detach().cpu())
        return (dx, dd, None, None, None, None, None, None, None)[:ctx.argument_count]


def calibrated(x, target, T=2, theta=1., tau=.1, compensate=True, soft=False, steps=32, trace=None):
    with torch.no_grad():
        d, active, err = solve(base_potential(x, T, theta), target, theta, tau, steps, soft)
    out = CalibratedResponse.apply(x, d, active, T, theta, tau, compensate, soft, trace)
    return out, d, active, err
