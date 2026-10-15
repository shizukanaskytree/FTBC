"""E13 serial IF replay and finite solvers. Mirrors source-snapshot/serial/Models/layer.py IF.forward op order.

replay(): per-element spike counts for given per-step biases (subtracted from the membrane, as in IF.forward).
deferral(): exact linear deferral = cumulative threshold offsets C_t=(T-t)/(2T)*theta (Deng & Gu 2021 bias shift
            theta/(2T) with zero initial potential, expressed through the existing time_based_bias).
solve_mean(): per-channel bisection on a time-shared bias current so the channel mean output matches a target.
fit_schedule(): layer-shared cumulative-offset schedule minimizing mean |count - local T-level twin count|.
No global-optimality claim; finite grids and bisection only.
"""
import torch


def biases_from_offsets(C, T):
    """Cumulative offsets C (T,) with C[T-1]=0 -> per-step biases b_1=C_1, b_t=C_t-C_{t-1} (same units)."""
    b = C.clone(); b[1:] = C[1:] - C[:-1]
    return b


def deferral(T, kappa=1.0, dtype=torch.float64):
    return torch.tensor([kappa * (T - t) / (2 * T) for t in range(1, T + 1)], dtype=dtype)


@torch.no_grad()
def replay(x, b, thre, chunk=32):
    """x: (T,N,C,H,W) or (T,N,C) float32 currents; b: (T,C) float32 biases; thre: IF.thresh.data (shape [1]).
    Returns per-element counts (uint8) with exactly the IF.forward arithmetic (mem=0.5*thre; mem-=b_t; mem+=x_t;
    spike=((mem-thre)>=0)*thre; mem-=spike)."""
    assert x.dtype == torch.float32 and b.dtype == torch.float32 and b.shape == (x.shape[0], x.shape[2])
    T = x.shape[0]; shape = (1, -1, 1, 1) if x.dim() == 5 else (1, -1)
    out = torch.empty(x.shape[1:], dtype=torch.uint8, device=x.device)
    for lo in range(0, x.shape[1], chunk):
        xs = x[:, lo:lo + chunk]
        mem = 0.5 * thre; n = torch.zeros(xs.shape[1:], dtype=torch.uint8, device=x.device)
        for t in range(T):
            mem = mem - b[t].view(shape)
            mem = mem + xs[t]
            s = (mem - thre) >= 0
            mem = mem - s.to(xs.dtype) * thre
            n += s
        out[lo:lo + chunk] = n
    return out


def channel_counts(n):
    dims = (0, 2, 3) if n.dim() == 4 else (0,)
    return n.to(torch.int64).sum(dims)


@torch.no_grad()
def local_twin(x, b, thre, chunk=32):
    """T-level twin count of each neuron's own effective input: clamp(floor((sum_t (x_t-b_t) + thre/2)/thre),0,T), FP64."""
    T = x.shape[0]; shape = (1, -1, 1, 1) if x.dim() == 5 else (1, -1); th = float(thre)
    out = torch.empty(x.shape[1:], dtype=torch.uint8, device=x.device)
    total_b = b.double().sum(0).view(shape)
    for lo in range(0, x.shape[1], chunk):
        s = x[:, lo:lo + chunk].double().sum(0) - total_b
        out[lo:lo + chunk] = torch.clamp(torch.floor((s + 0.5 * th) / th), 0, T).to(torch.uint8)
    return out


@torch.no_grad()
def solve_mean(x, b0, thre, target, steps=32, bound=0.5, chunk=32):
    """Time-shared bias current beta per channel (b_t = b0_t - beta) so that thre*count_c/n_c matches target (FP64).
    Candidates: last lo/hi of 32 bisections, zero, both bounds; min |residual|, then smaller |beta|, then smaller beta."""
    T, C = x.shape[0], x.shape[2]; th = float(thre); n_c = x.numel() // C
    target = target.to(device=x.device, dtype=torch.float64); assert target.shape == (C,) and torch.isfinite(target).all()
    def mean(beta):
        return channel_counts(replay(x, b0 - beta.view(1, C), thre, chunk)).double() * th / n_c
    lo = torch.full((C,), -bound * th, device=x.device, dtype=torch.float32); hi = -lo
    lower, upper = mean(lo), mean(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2; less = mean(mid) < target
        lo = torch.where(less, mid, lo); hi = torch.where(less, hi, mid)
    zero = torch.zeros_like(lo); blo = torch.full_like(lo, -bound * th); bhi = -blo
    best = lo.clone(); bm = mean(best); err = (bm - target).abs()
    for d in [hi, zero, blo, bhi]:
        m = mean(d); e = (m - target).abs()
        take = (e < err) | ((e == err) & ((d.abs() < best.abs()) | ((d.abs() == best.abs()) & (d < best))))
        best = torch.where(take, d, best); bm = torch.where(take, m, bm); err = torch.where(take, e, err)
    z = mean(zero); assert torch.all(err <= (z - target).abs()) and torch.all(best.abs() <= bound * th)
    return best, {'target_mean': target.cpu(), 'mean_before': z.cpu(), 'mean_after': bm.cpu(), 'beta': best.cpu(),
                  'normalized_residual': ((bm - target) / th).cpu(), 'lower_mean': lower.cpu(), 'upper_mean': upper.cpu(),
                  'below_range': (target < lower).cpu(), 'above_range': (target > upper).cpu(),
                  'at_bound': (best.abs() == bound * th).cpu(), 'observations_per_channel': n_c, 'theta': th}


@torch.no_grad()
def fit_schedule(x, thre, T, grid_den=32, grid_max=16, sweeps=3, chunk=32):
    """Layer-shared cumulative offsets C_1..C_{T-1} in theta units on {0,1/32,...,16/32}, C_T=0, initialised at the
    exact linear deferral; coordinate descent on mean |count - local twin| (twin uses the zero-sum schedule: unchanged)."""
    C = x.shape[2]; th = float(thre)
    twin = local_twin(x, torch.zeros(T, C, dtype=torch.float32, device=x.device), thre, chunk)
    def objective(off):
        b = (biases_from_offsets(off, T) * th).to(torch.float32).to(x.device).view(T, 1).expand(T, C).contiguous()
        return float((replay(x, b, thre, chunk).to(torch.int16) - twin.to(torch.int16)).abs().double().mean())
    off = deferral(T); grid = torch.arange(0, grid_max + 1, dtype=torch.float64) / grid_den
    history = [{'offsets': off.tolist(), 'objective': objective(off)}]
    for _ in range(sweeps):
        for t in range(T - 1):
            vals = []
            for g in grid:
                trial = off.clone(); trial[t] = g; vals.append(objective(trial))
            vals = torch.tensor(vals, dtype=torch.float64)
            best = vals.min(); idx = [i for i in range(len(grid)) if vals[i] == best]
            pick = min(idx, key=lambda i: (abs(float(grid[i]) - float(off[t])), float(grid[i])))
            off[t] = grid[pick]
        history.append({'offsets': off.tolist(), 'objective': objective(off)})
    return off, {'history': history, 'baseline_IF_objective': objective(torch.zeros(T, dtype=torch.float64)),
                 'grid': [0, grid_max / grid_den, 1 / grid_den]}
