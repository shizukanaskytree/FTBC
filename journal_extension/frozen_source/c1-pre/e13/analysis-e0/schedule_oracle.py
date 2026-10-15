"""CPU exploratory: can a time-varying threshold schedule reduce the local count error of serial IF neurons?
Fixed saved inputs (E0 plain), target = local T-level twin of each neuron's own input. Fit on images 0-31, evaluate on 32-63."""
import json, sys, torch
torch.set_num_threads(16)
D = torch.load('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008/serial/diagnostics-plain.pt', map_location='cpu', weights_only=False)
T = D['metadata']['T']; P = 4  # positions per image
grid = torch.arange(-16, 17, dtype=torch.float64) / 32          # offsets in units of theta
def count(x, th, C):  # x (T,S,K) fp64, th (K,), C (T,K) in theta units -> counts (S,K)
    mem = (0.5 * th).expand(x.shape[1], -1).clone(); n = torch.zeros_like(mem)
    for t in range(T):
        mem = mem + x[t]; s = (mem - th * (1 + C[t]) >= 0).double(); mem = mem - s * th; n = n + s
    return n
def err(x, th, C, tgt): return (count(x, th, C) - tgt).abs().mean(0)   # per-channel mean |count error|
def fit(x, th, tgt, shared):
    K = x.shape[2]; C = torch.zeros(T, K, dtype=torch.float64)
    for sweep in range(3):
        for t in range(T):
            best = None
            for g in grid:
                C2 = C.clone(); C2[t] = g
                e = err(x, th, C2, tgt); e = e.mean().expand(K) if shared else e
                best = e.clone().unsqueeze(0) if best is None else torch.cat([best, e.unsqueeze(0)])
            idx = best.argmin(0) if not shared else best[:, 0].argmin().expand(K)
            C[t] = grid[idx]
    return C
rows = []; pool_tr, pool_te = [], []
names = list(D['layers'])[1:]          # conv1 has constant input: zero local error
for name in names:
    L = D['layers'][name]; x = L['snn_input'].double(); th = L['theta'].double()
    S = x.sum(0); tgt = torch.clamp(torch.floor((S + 0.5 * th) / th), 0, T)
    ntr = 32 * P; xtr, xte, ttr, tte = x[:, :ntr], x[:, ntr:], tgt[:ntr], tgt[ntr:]
    base_te = err(xte, th, torch.zeros(T, x.shape[2], dtype=torch.float64), tte).mean()
    Cc = fit(xtr, th, ttr, shared=False); Cl = fit(xtr, th, ttr, shared=True)
    rows.append({'layer': name, 'base_test': float(base_te), 'per_channel_test': float(err(xte, th, Cc, tte).mean()),
                 'layer_shared_test': float(err(xte, th, Cl, tte).mean()), 'layer_shared_schedule': Cl[:, 0].tolist()})
    pool_tr.append((xtr / th, ttr)); pool_te.append((xte / th, tte))
# One global schedule for all layers (inputs normalized by theta)
xg = torch.cat([a for a, _ in pool_tr], 2); tg = torch.cat([b for _, b in pool_tr], 1); one = torch.ones(xg.shape[2], dtype=torch.float64)
Cg = fit(xg, one, tg, shared=True)[:, :1]
xge = torch.cat([a for a, _ in pool_te], 2); tge = torch.cat([b for _, b in pool_te], 1); onee = torch.ones(xge.shape[2], dtype=torch.float64)
g_base = float(err(xge, onee, torch.zeros(T, xge.shape[2], dtype=torch.float64), tge).mean()); g_fit = float(err(xge, onee, Cg.expand(T, xge.shape[2]), tge).mean())
out = {'rows': rows, 'global_schedule_theta_units': Cg[:, 0].tolist(), 'global_test_base': g_base, 'global_test_fit': g_fit}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for r in rows: print(f"{r['layer'][:30]:30s} base={r['base_test']:.4f} per-chan={r['per_channel_test']:.4f} layer-shared={r['layer_shared_test']:.4f} sched={[round(v,3) for v in r['layer_shared_schedule']]}")
print('GLOBAL schedule', [round(v, 4) for v in Cg[:, 0].tolist()], 'test |err| base', round(g_base, 4), '->', round(g_fit, 4))
