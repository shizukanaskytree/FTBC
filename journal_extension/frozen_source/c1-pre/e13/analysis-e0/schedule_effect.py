"""CPU exploratory: effect of the global decreasing schedule on over/under-firing, drift, and constant-input exactness."""
import torch, itertools
torch.set_num_threads(8)
D = torch.load('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008/serial/diagnostics-plain.pt', map_location='cpu', weights_only=False)
T = 4; C = torch.tensor([0.3125, 0.25, 0.15625, 0.0], dtype=torch.float64)
def count(x, th, C):
    mem = (0.5 * th).expand(x.shape[1], -1).clone(); n = torch.zeros_like(mem); times = []
    for t in range(T):
        mem = mem + x[t]; s = (mem - th * (1 + C[t]) >= 0).double(); mem = mem - s * th; n = n + s; times.append(s)
    return n, torch.stack(times), mem
# 1) constant inputs over the whole representable range: counts must equal the twin
z = torch.linspace(-0.5, 1.5, 20001, dtype=torch.float64).view(1, -1, 1).expand(T, -1, 1); one = torch.ones(1, dtype=torch.float64)
tw = torch.clamp(torch.floor((z.sum(0) + 0.5) / 1.0), 0, T)
for name, sched in [('IF', torch.zeros(T, dtype=torch.float64)), ('sched', C)]:
    n, _, _ = count(z, one, sched.view(T, 1)); print(name, 'constant-input count mismatches:', int((n != tw).sum()), 'of', tw.numel())
# 2) real inputs (images 32-63 = held-out half), pooled over layers 2..33
tot = {k: [] for k in ['IF', 'sched']}
for name, L in list(D['layers'].items())[1:]:
    x = L['snn_input'].double()[:, 128:]; th = L['theta'].double()
    tgt = torch.clamp(torch.floor((x.sum(0) + 0.5 * th) / th), 0, T)
    for k, sched in [('IF', torch.zeros(T, dtype=torch.float64)), ('sched', C)]:
        n, sp, mem = count(x, th, sched.view(T, 1).expand(T, x.shape[2]))
        e = n - tgt; tot[k].append((e, sp))
for k, v in tot.items():
    e = torch.cat([a.flatten() for a, _ in v]); sp = torch.cat([b.reshape(T, -1) for _, b in v], 1)
    print(k, 'mean|e|', round(float(e.abs().mean()), 4), 'signed', round(float(e.mean()), 4), 'over', round(float((e > 0).double().mean()), 4), 'under', round(float((e < 0).double().mean()), 4),
          'spike share by step', [round(float(s), 3) for s in (sp.sum(1) / sp.sum())])
