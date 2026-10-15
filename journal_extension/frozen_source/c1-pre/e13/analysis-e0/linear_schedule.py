"""CPU exploratory: linear deferral schedule C_t = kappa*(T-t)/(2T)*theta; exactness for constant input and held-out local error."""
import torch
torch.set_num_threads(8)
D = torch.load('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008/serial/diagnostics-plain.pt', map_location='cpu', weights_only=False)
def count(x, th, C, T):
    mem = (0.5 * th).expand(x.shape[1], -1).clone(); n = torch.zeros_like(mem)
    for t in range(T):
        mem = mem + x[t]; s = (mem - th * (1 + C[t]) >= 0).double(); mem = mem - s * th; n = n + s
    return n
for T in [2, 4, 8]:
    z = torch.linspace(-0.5, 1.5, 40001, dtype=torch.float64).view(1, -1, 1).expand(T, -1, 1); one = torch.ones(1, dtype=torch.float64)
    tw = torch.clamp(torch.floor(z.sum(0) + 0.5), 0, T)
    res = []
    for kappa in [0, 0.5, 0.9, 1.0, 1.1]:
        C = torch.tensor([kappa * (T - t) / (2 * T) for t in range(1, T + 1)], dtype=torch.float64).view(T, 1)
        res.append((kappa, int((count(z, one, C, T) != tw).sum())))
    print('T', T, 'constant-input mismatches by kappa', res)
T = 4
layers = list(D['layers'].items())[1:]
for kappa in [0, 0.25, 0.5, 0.75, 1.0]:
    C = torch.tensor([kappa * (T - t) / (2 * T) for t in range(1, T + 1)], dtype=torch.float64)
    es = []
    for name, L in layers:
        x = L['snn_input'].double()[:, 128:]; th = L['theta'].double()
        tgt = torch.clamp(torch.floor((x.sum(0) + 0.5 * th) / th), 0, T)
        es.append((count(x, th, C.view(T, 1).expand(T, x.shape[2]), T) - tgt).flatten())
    e = torch.cat(es)
    print(f'kappa={kappa:.2f} sched={[round(float(c),4) for c in C]} held-out mean|e|={float(e.abs().mean()):.4f} signed={float(e.mean()):+.4f} over={float((e>0).double().mean()):.4f} under={float((e<0).double().mean()):.4f}')
