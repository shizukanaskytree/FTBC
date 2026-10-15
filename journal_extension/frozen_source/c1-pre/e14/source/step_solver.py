"""E14 exact per-time-step solver for the FTBC objective under the deployed serial IF dynamics.

For one layer with cached inputs x (T,N,C,H,W) and fixed earlier steps, choose b_t per channel (sequentially t=1..T)
so that the channel mean output at step t, theta*count_t/(N*H*W), matches the target (teacher channel mean).
With b_1..b_{t-1} fixed, the step-t count is non-increasing in b_t (b_t does not touch earlier states and the FP32 ops are
monotone); later single steps are not monotone in b_t, hence the time order. The membrane recurrence replicates IF.forward:
mem=0.5*thre; mem-=b_t; mem+=x_t; spike=((mem-thre)>=0)*thre; mem-=spike. Finite bisection, no optimality claim.
"""
import torch


@torch.no_grad()
def _step_counts(mem_prev, xt, bt, thre, chunk=32):
    shape = (1, -1, 1, 1) if xt.dim() == 4 else (1, -1)
    cnt = torch.zeros(xt.shape[1], dtype=torch.int64, device=xt.device)
    dims = (0, 2, 3) if xt.dim() == 4 else (0,)
    for lo in range(0, xt.shape[0], chunk):
        mem = mem_prev[lo:lo + chunk] - bt.view(shape)
        mem = mem + xt[lo:lo + chunk]
        cnt += ((mem - thre) >= 0).to(torch.int64).sum(dims)
    return cnt


@torch.no_grad()
def _advance(mem_prev, xt, bt, thre, chunk=32):
    shape = (1, -1, 1, 1) if xt.dim() == 4 else (1, -1)
    out = torch.empty_like(mem_prev); n = torch.empty(xt.shape, dtype=torch.uint8, device=xt.device)
    for lo in range(0, xt.shape[0], chunk):
        mem = mem_prev[lo:lo + chunk] - bt.view(shape)
        mem = mem + xt[lo:lo + chunk]
        s = (mem - thre) >= 0
        out[lo:lo + chunk] = mem - s.to(xt.dtype) * thre; n[lo:lo + chunk] = s
    return out, n


@torch.no_grad()
def solve_per_step(x, thre, target, steps=32, bound=0.5, chunk=32):
    """Returns b (T,C) float32 and per-step info; also the total per-element counts after all steps."""
    T, C = x.shape[0], x.shape[2]; th = float(thre); n_c = x[0].numel() // C
    target = target.to(device=x.device, dtype=torch.float64); assert target.shape == (C,) and torch.isfinite(target).all()
    mem = (0.5 * thre).expand(x.shape[1:]).to(x.dtype).clone()    # IF.forward starts every neuron at 0.5*thre
    b = torch.zeros(T, C, dtype=torch.float32, device=x.device); info = []; total = torch.zeros(x.shape[1:], dtype=torch.uint8, device=x.device)
    mean = lambda c: c.double() * th / n_c
    for t in range(T):
        lo = torch.full((C,), -bound * th, device=x.device, dtype=torch.float32); hi = -lo
        f = lambda bt: mean(_step_counts(mem, x[t], bt, thre, chunk))
        at_lo, at_hi = f(lo), f(hi)          # bigger b -> fewer spikes: at_lo is the maximum reachable step mean
        for _ in range(steps):
            mid = (lo + hi) / 2; too_many = f(mid) > target
            lo = torch.where(too_many, mid, lo); hi = torch.where(too_many, hi, mid)
        zero = torch.zeros_like(lo); blo = torch.full_like(lo, -bound * th); bhi = -blo
        best = lo.clone(); bm = f(best); err = (bm - target).abs()
        for d in [hi, zero, blo, bhi]:
            m = f(d); e = (m - target).abs()
            take = (e < err) | ((e == err) & ((d.abs() < best.abs()) | ((d.abs() == best.abs()) & (d < best))))
            best = torch.where(take, d, best); bm = torch.where(take, m, bm); err = torch.where(take, e, err)
        z = f(zero); assert torch.all(err <= (z - target).abs()) and torch.all(best.abs() <= bound * th)
        b[t] = best
        mem, s = _advance(mem, x[t], best, thre, chunk); total += s
        info.append({'t': t + 1, 'b': best.cpu(), 'mean_before': z.cpu(), 'mean_after': bm.cpu(), 'normalized_residual': ((bm - target) / th).cpu(),
                     'max_reachable_mean': at_lo.cpu(), 'min_reachable_mean': at_hi.cpu(), 'above_range': (target > at_lo).cpu(),
                     'below_range': (target < at_hi).cpu(), 'at_bound': (best.abs() == bound * th).cpu()})
    return b, {'steps': info, 'target_mean': target.cpu(), 'theta': th, 'observations_per_channel_per_step': n_c}, total
