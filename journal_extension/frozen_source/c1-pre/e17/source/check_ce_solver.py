"""CPU checks of ce_solver.py on small random layers (4D and 2D interfaces, T=2 and T=4, several thresholds). Never initializes CUDA.
0. The exact int64 objective path of best_cut equals a float64-cumsum reference with the same cut and tie rules, bitwise.
1. CE steps: the objective at the chosen b is not worse than the minimum over a 4001-point grid, b within the bound.
2. CE-A last step: |mean - anchor target| is minimal over the same grid.
3. The frozen E13 IF replay (serial_solver.replay) reproduces the solver's total counts bitwise; near ties are reported.
4. With the units split across several chunks, the result equals the single-chunk result bitwise."""
import importlib.util, json, sys
from pathlib import Path
import torch
P = Path(__file__).resolve().parents[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ps = load('t_ps', P / 'e14/source/step_solver.py'); ss = load('t_ss', P / 'e13/source/serial_solver.py'); ce = load('t_ce', Path(__file__).resolve().parent / 'ce_solver.py')
import hashlib, math


def best_cut_float(m, w, B):
    """Reference for the check only (CPU): the same realizable-cut and tie rules as ce_solver.best_cut, summed with a float64 cumsum."""
    hi_const = torch.where(m >= B, w, torch.zeros_like(w)).sum(1)
    inr = (m >= -B) & (m < B); nin = inr.sum(1)
    srt, idx = torch.sort(torch.where(inr, m, torch.full_like(m, -float('inf'))), dim=1, descending=True, stable=True)
    pref = torch.cat([torch.zeros_like(m[:, :1]), torch.cumsum(torch.gather(torch.where(inr, w, torch.zeros_like(w)), 1, idx), 1)], 1)
    U1 = pref.shape[1]; k = torch.arange(U1).view(1, -1)
    distinct = torch.ones_like(pref, dtype=torch.bool); distinct[:, 1:-1] = srt[:, :-1] > srt[:, 1:]
    valid = (k <= nin.view(-1, 1)) & distinct
    val = torch.where(valid, pref, torch.full_like(pref, float('inf'))); vmin = val.min(1).values; opt = valid & (val == vmin.view(-1, 1))
    pad = torch.cat([torch.full_like(srt[:, :1], B), srt, torch.full_like(srt[:, :1], -float('inf'))], 1)
    upper = pad[:, :U1].clamp(max=B); lower = pad[:, 1:U1 + 1]; last = k >= nin.view(-1, 1)
    lower = torch.where(last, torch.full_like(lower, -B), lower); has_zero = torch.where(last, upper >= 0, (lower < 0) & (upper >= 0))
    bk = torch.where(has_zero, torch.zeros_like(upper), (lower + upper) / 2)
    absb = torch.where(opt, bk.abs(), torch.full_like(bk, float('inf'))); cand = opt & (absb == absb.min(1).values.view(-1, 1))
    bsel = torch.where(cand, bk, torch.full_like(bk, float('inf'))).min(1).values
    kstar = torch.where(cand & (bk == bsel.view(-1, 1)), k, torch.full_like(k, U1)).min(1).values
    return bsel, hi_const + vmin, nin, (kstar == 0) | (kstar == nin)


torch.manual_seed(0); report = {'cases': [], 'ce_solver_sha256': hashlib.sha256((Path(__file__).resolve().parent / 'ce_solver.py').read_bytes()).hexdigest()}
cut_equal = []
grid_of = lambda th: torch.linspace(-0.5 * th, 0.5 * th, 4001).tolist() + [0.0]
for case in range(12):
    T = [2, 4][case % 2]; four_d = case < 6 or case >= 8; thre = torch.tensor([1.0 + 0.5 * (case % 8)], dtype=torch.float32); th = float(thre)
    shape = (T, 3, 4, 5, 5) if four_d else (T, 40, 6)
    x = torch.randn(*shape) * 0.6 * th + 0.3 * th; a = (torch.rand(*shape[1:]) * th); a[a < 0.2 * th] = 0
    if case >= 8:                                     # tie-heavy inputs: coarse grid values and repeated (zero-padding-like) entries
        x = torch.round(x / th * 4) / 4 * th; x[..., :2, :] = 0.25 * th; a = torch.round(a / th * 8) / 8 * th
    target = a.double().mean(dim=tuple(i for i in range(a.dim()) if i != 1))
    rec = {'T': T, 'interface': '4D' if four_d else '2D', 'theta': th, 'tie_heavy': case >= 8}
    # the exact int64 path equals the float64 reference on every step state of this layer (CE without anchor)
    mem0 = (0.5 * thre).expand(x.shape[1:]).clone(); cnt = torch.zeros(x.shape[1:], dtype=torch.uint8); B0 = 0.5 * th; q = math.ldexp(1.0, math.frexp(th)[1] - 27)
    for t in range(T):
        C = x.shape[2]; mm = ce._perm(mem0, 0, C) + ce._perm(x[t], 0, C) - th; ww = th - 2 * ((t + 1) * ce._perm(a, 0, C) - th * ce._perm(cnt, 0, C))
        r_int = ce.best_cut(mm, ww, B0, q); r_flt = best_cut_float(mm, ww, B0)
        cut_equal.append(all(torch.equal(u, v) for u, v in zip(r_int, r_flt)))
        mem0, s0 = ps._advance(mem0, x[t], r_int[0].to(torch.float32), thre); cnt += s0
    for anchored in [False, True]:
        b, info, total = ce.solve_per_step_ce(ps, x, a, thre, 0.5, target if anchored else None)
        b1, _, total1 = ce.solve_per_step_ce(ps, x, a, thre, 0.5, target if anchored else None, units_per_chunk=7)   # many tiny chunks
        mem = (0.5 * thre).expand(x.shape[1:]).clone(); R = torch.zeros(x.shape[1:], dtype=torch.float64); ok = True; gap = 0.0
        dims = tuple(i for i in range(a.dim()) if i != 1); cum = torch.zeros(x.shape[2], dtype=torch.float64)
        for t in range(T):
            mm, s = ps._advance(mem, x[t], b[t], thre); s = s.double()
            if anchored and t == T - 1:
                tgt = T * target - cum
                got = ((s.mean(dims) * th) - tgt).abs()
                best = None
                for g in grid_of(th):
                    _, sg = ps._advance(mem, x[t], torch.full((x.shape[2],), g), thre); e = ((sg.double().mean(dims) * th) - tgt).abs()
                    best = e if best is None else torch.minimum(best, e)
            else:
                got = ((R + s * th - (t + 1) * a.double()) ** 2).sum(dims); best = None
                for g in grid_of(th):
                    _, sg = ps._advance(mem, x[t], torch.full((x.shape[2],), g), thre)
                    e = ((R + sg.double() * th - (t + 1) * a.double()) ** 2).sum(dims); best = e if best is None else torch.minimum(best, e)
            gap = max(gap, float((got - best).max())); ok &= bool((got <= best + 1e-9).all())
            cum += s.mean(dims) * th; R = R + s * th; mem = mm
        rec['CE-A' if anchored else 'CE'] = {'not_worse_than_grid': ok, 'max_gap': gap, 'bound_ok': bool((b.abs() <= 0.5 * th + 1e-6).all()),
                                             'frozen_replay_equal': bool(torch.equal(ss.replay(x, b, thre), total)),
                                             'chunking_invariant': bool(torch.equal(b, b1) and torch.equal(total, total1)),
                                             'near_ties': [i['near_tie_mismatches'] for i in info]}
    report['cases'].append(rec)
report['passed'] = all(v['not_worse_than_grid'] and v['bound_ok'] and v['frozen_replay_equal'] and v['chunking_invariant']
                       for c in report['cases'] for k, v in c.items() if k in ('CE', 'CE-A'))
report['int64_path_equals_float64_reference'] = all(cut_equal); report['cut_comparisons'] = len(cut_equal)
report['tie_heavy_near_ties'] = sum(sum(c[k]['near_ties']) for c in report['cases'] if c['tie_heavy'] for k in ('CE', 'CE-A'))
report['cuda_initialized'] = torch.cuda.is_initialized(); assert report['passed'] and not report['cuda_initialized'] and report['tie_heavy_near_ties'] == 0 and report['int64_path_equals_float64_reference']
(Path(__file__).resolve().parents[1] / 'cache/check-ce-solver.json').write_text(json.dumps(report, indent=1) + '\n')
print(json.dumps({'passed': report['passed'], 'int64_equals_float64': report['int64_path_equals_float64_reference'], 'comparisons': report['cut_comparisons'], 'cases': [(c['T'], c['interface'], c['CE']['max_gap'], c['CE-A']['max_gap'], c['CE']['near_ties']) for c in report['cases']]}))
