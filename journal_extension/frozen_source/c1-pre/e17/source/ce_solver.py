"""E17 per-step solvers of a cumulative per-unit error objective under the deployed serial IF dynamics.

Same setting and recurrence as e14/source/step_solver.py (imported, frozen): one layer, cached inputs x (T,N,C,H,W) with earlier layers
already calibrated; mem=0.5*thre; for t: mem-=b_t; mem+=x_t; spike=((mem-thre)>=0)*thre; mem-=spike. Per channel and in time order
(b_1..b_{t-1} fixed), b_t in [-bound*theta, bound*theta]:

CE   minimizes E_t(b) = sum_i (R_{t,i}(b) - t*a_i)^2, the squared error of the cumulative output R (theta x spikes so far) against t times
     the unit's reference activation a_i (the teacher output at this interface for the same image and position). With s in {0, theta},
     E_t(b) = const + theta * sum_{i spikes at t} (theta - 2 d_i), d_i = t*a_i - R_{t-1,i}. In real arithmetic unit i spikes iff
     b <= m_i = mem_{t-1,i} + x_{t,i} - theta, so the spike set is a top set in m: the exact minimizer is the best prefix of the in-range
     margins sorted in decreasing order, over the cuts a threshold can realize (between distinct margin values; units with equal margins
     always spike together). The objective is summed in int64 units of q = 2^(e_theta - 26): the reference is a QCFS L=8 output, so a is 0 or
     at least theta/8 and every cost theta - 2d is an integer multiple of q (asserted), which makes the prefix sums exact and deterministic on
     CUDA (float cumsum is not allowed under torch.use_deterministic_algorithms). b_t is the midpoint of the optimal margin interval, or 0
     if that interval contains 0; among equally optimal cuts, b = 0 first, then the smallest |b|, then the smaller b (the P0/M0 order).
CE-A uses CE for steps 1..T-1; the last step instead matches the cumulative channel mean, i.e. the mean count over all T steps times theta
     equals the target (FTBC's channel mean target): the P0 single-step bisection of step_solver with target T*target - sum_{t<T} mean_t.

After each step the float32 IF arithmetic (step_solver._advance) is applied; units whose float32 spike differs from the float64 top-set
prediction (near ties) are counted and reported.
"""
import math
import torch


def _perm(z, lo, hi):
    """(N,C,...) channel slice -> (c, units) float64."""
    return z[:, lo:hi].transpose(0, 1).reshape(hi - lo, -1).double()


@torch.no_grad()
def best_cut(m, w, B, q):
    """m, w: (c, U) float64 margins and spike costs; w must be integer multiples of q (asserted), so all objective sums are exact int64.
    Returns b (c,) float64 minimizing sum_{m_i >= b} w_i over b in [-B, B] among the cuts a threshold can realize, the minimum (float64),
    the number of in-range units and whether the chosen cut lies at an end of the in-range set (k* = 0 or k* = n_in). Units with m >= B
    spike for every admissible b; units with m < -B never do. Ties among optimal cuts: an interval containing 0 (b = 0) first, then the
    smallest |b| (b = interval midpoint), then the smaller b."""
    wi = torch.round(w / q).to(torch.int64)
    assert torch.equal(wi.to(torch.float64) * q, w), 'spike costs are not integer multiples of q'
    hi_const = torch.where(m >= B, wi, torch.zeros_like(wi)).sum(1)
    inr = (m >= -B) & (m < B); nin = inr.sum(1)
    key = torch.where(inr, m, torch.full_like(m, -float('inf')))
    srt, idx = torch.sort(key, dim=1, descending=True, stable=True); del key
    ws = torch.gather(torch.where(inr, wi, torch.zeros_like(wi)), 1, idx); del idx, wi, inr
    assert ws.dtype == torch.int64                                                     # never a float cumsum (CUDA determinism)
    pref = torch.cat([torch.zeros_like(ws[:, :1]), torch.cumsum(ws, 1)], 1); del ws     # int64: deterministic and exact on CUDA
    U1 = pref.shape[1]; k = torch.arange(U1, device=m.device).view(1, -1)
    # a threshold can only cut between distinct margins: the top-k set is realizable iff k = 0, k = n_in or m_(k) > m_(k+1)
    distinct = torch.ones_like(pref, dtype=torch.bool); distinct[:, 1:-1] = srt[:, :-1] > srt[:, 1:]
    valid = (k <= nin.view(-1, 1)) & distinct; del distinct
    val = torch.where(valid, pref, torch.full_like(pref, torch.iinfo(torch.int64).max)); del pref
    vmin = val.min(1).values; opt = valid & (val == vmin.view(-1, 1)); del val, valid
    # b giving exactly the top-k set lies in (m_(k+1), m_(k)], with m_(0) = B and, for k = n_in, the lower end -B (closed)
    pad = torch.cat([torch.full_like(srt[:, :1], B), srt, torch.full_like(srt[:, :1], -float('inf'))], 1); del srt
    upper = pad[:, :U1].clamp(max=B); lower = pad[:, 1:U1 + 1]; del pad
    last = k >= nin.view(-1, 1)
    lower = torch.where(last, torch.full_like(lower, -B), lower)
    has_zero = torch.where(last, upper >= 0, (lower < 0) & (upper >= 0)); del last
    bk = torch.where(has_zero, torch.zeros_like(upper), (lower + upper) / 2); del upper, lower, has_zero
    absb = torch.where(opt, bk.abs(), torch.full_like(bk, float('inf'))); amin = absb.min(1).values
    cand = opt & (absb == amin.view(-1, 1)); del absb, opt
    bsel = torch.where(cand, bk, torch.full_like(bk, float('inf'))).min(1).values
    kstar = torch.where(cand & (bk == bsel.view(-1, 1)), k, torch.full_like(k, U1)).min(1).values; del cand, bk
    at_end = (kstar == 0) | (kstar == nin)
    return bsel, (hi_const + vmin).to(torch.float64) * q, nin, at_end


@torch.no_grad()
def _mean_step(ps, mem, xt, thre, target, steps, bound, chunk=32):
    """The P0 single-step rule of step_solver.solve_per_step (bisection, candidates lo/hi/0/+-bound, min |residual| then |b| then b)."""
    C = xt.shape[1]; th = float(thre); n_c = xt.numel() // C
    mean = lambda c: c.double() * th / n_c
    lo = torch.full((C,), -bound * th, device=xt.device, dtype=torch.float32); hi = -lo
    f = lambda bt: mean(ps._step_counts(mem, xt, bt, thre, chunk))
    at_lo, at_hi = f(lo), f(hi)
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
    return best, {'mean_after': bm.cpu(), 'normalized_residual': ((bm - target) / th).cpu(), 'above_range': (target > at_lo).cpu(),
                  'below_range': (target < at_hi).cpu(), 'at_bound': (best.abs() == bound * th).cpu(), 'anchor_target': target.cpu()}


@torch.no_grad()
def solve_per_step_ce(ps, x, a, thre, bound=0.5, anchor_target=None, steps=32, units_per_chunk=2 ** 26):
    """x: (T,N,C,H,W) float32 inputs; a: (N,C,H,W) float32 reference outputs of the same units. anchor_target=None gives CE; a (C,) channel
    mean target gives CE-A. Returns b (T,C) float32, per-step info and the per-element total spike counts (uint8)."""
    assert x.dtype == torch.float32 and a.dtype == torch.float32 and a.shape == x.shape[1:]
    T, C = x.shape[0], x.shape[2]; th = float(thre); B = bound * th; n_c = x[0].numel() // C; q = math.ldexp(1.0, math.frexp(th)[1] - 27)
    U = n_c; chunk = max(1, min(C, units_per_chunk // max(U, 1)))
    mem = (0.5 * thre).expand(x.shape[1:]).to(x.dtype).clone()
    count = torch.zeros(x.shape[1:], dtype=torch.uint8, device=x.device)
    b_all = torch.zeros(T, C, dtype=torch.float32, device=x.device); info = []; cum_means = torch.zeros(C, dtype=torch.float64, device=x.device)
    for t in range(T):
        anchored = anchor_target is not None and t == T - 1
        if anchored:
            target_t = T * anchor_target.to(device=x.device, dtype=torch.float64) - cum_means
            bt, extra = _mean_step(ps, mem, x[t], thre, target_t, steps, bound); pred = None; nin = None; best = None
        else:
            bt64 = torch.empty(C, dtype=torch.float64, device=x.device); best = torch.empty_like(bt64); nin = torch.empty(C, dtype=torch.int64, device=x.device)
            at_end = torch.empty(C, dtype=torch.bool, device=x.device)
            for lo in range(0, C, chunk):
                hi = min(C, lo + chunk)
                m = _perm(mem, lo, hi) + _perm(x[t], lo, hi) - th
                d = (t + 1) * _perm(a, lo, hi) - th * _perm(count, lo, hi)
                bt64[lo:hi], best[lo:hi], nin[lo:hi], at_end[lo:hi] = best_cut(m, th - 2 * d, B, q); del m, d
            bt = bt64.to(torch.float32).clamp(-B, B); extra = {}
        new_mem, s = ps._advance(mem, x[t], bt, thre)
        near_ties = 0
        if not anchored:                          # float32 IF spike versus the float64 top-set prediction at the chosen b
            for lo in range(0, C, chunk):
                hi = min(C, lo + chunk)
                pred = (_perm(mem, lo, hi) + _perm(x[t], lo, hi) - th) >= bt64[lo:hi].view(-1, 1)
                near_ties += int((pred != _perm(s, lo, hi).bool()).sum()); del pred
        step_mean = s.sum(dim=tuple(i for i in range(s.dim()) if i != 1), dtype=torch.float64) * th / n_c
        cum_means += step_mean; count += s; mem = new_mem; b_all[t] = bt
        rec = {'t': t + 1, 'b': bt.cpu(), 'rule': 'mean-anchor' if anchored else 'cumulative-error', 'mean_after': step_mean.cpu(),
               'near_tie_mismatches': near_ties, 'at_bound': (bt.abs() == B).cpu(), **extra}
        if not anchored:
            rec['in_range_units'] = nin.cpu(); rec['objective_min_offset'] = best.cpu(); rec['cut_at_range_end'] = at_end.cpu()
        info.append(rec)
    return b_all, info, count
