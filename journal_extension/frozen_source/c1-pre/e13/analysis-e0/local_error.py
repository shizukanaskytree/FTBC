"""CPU-only exploratory analysis of E0 serial diagnostics (saved FP32 observations). No CUDA."""
import json, sys, torch
torch.set_num_threads(8)
D = torch.load('/data5/ftbc_neural_networks/runs/c1pre-order-e0-20261008/serial/diagnostics-plain.pt', map_location='cpu', weights_only=False)
T = D['metadata']['T']
rows = []
def simulate(x, th, C=None):
    # x: (T,S,C) fp32 currents; th: (C,) ; C: (T,C) cumulative threshold offsets (fp32) or None
    mem = (0.5 * th).expand(x.shape[1], -1).clone()
    spikes = []
    for t in range(T):
        mem = mem + x[t]
        thr = th if C is None else th + C[t]
        s = (mem - thr >= 0).to(x.dtype) * th
        mem = mem - s
        spikes.append(s)
    return torch.stack(spikes)
for name, L in D['layers'].items():
    x = L['snn_input'].float(); sp = L['spikes'].float(); th = L['theta'].float()
    sim = simulate(x, th)
    replay = torch.equal(sim, sp)
    N = (sp / th).sum(0)                                   # serial counts (S,C)
    S = x.double().sum(0)
    Nloc = torch.clamp(torch.floor((S + 0.5 * th.double()) / th.double()), 0, T)   # local T-level twin of own input
    Nprox = L['proxy'].double() / th.double() * T          # proxy (Q4 twin) on proxy's own input
    e = (N.double() - Nloc)
    ein = Nloc - Nprox
    sse = float((e ** 2).sum()); cm = e.mean(0)
    rows.append({'layer': name, 'replay_exact': replay, 'mean_abs_local': float(e.abs().mean()), 'frac_local_nonzero': float((e != 0).double().mean()),
                 'mean_signed_local': float(e.mean()), 'frac_over': float((e > 0).double().mean()), 'frac_under': float((e < 0).double().mean()),
                 'channel_mean_share_of_sse': float((cm ** 2).sum() * e.shape[0] / sse) if sse else None,
                 'mean_abs_input_divergence': float(ein.abs().mean()), 'mean_abs_total_vs_proxy': float((N.double() - Nprox).abs().mean()),
                 'mean_count_snn': float(N.double().mean()), 'mean_count_local_twin': float(Nloc.mean()), 'mean_count_proxy': float(Nprox.mean())})
json.dump(rows, open(sys.argv[1], 'w'), indent=1)
for r in rows:
    print(f"{r['layer'][:30]:30s} replay={r['replay_exact']!s:5s} |e_loc|={r['mean_abs_local']:.3f} nz={r['frac_local_nonzero']:.2f} over={r['frac_over']:.2f} under={r['frac_under']:.2f} chanshare={r['channel_mean_share_of_sse'] if r['channel_mean_share_of_sse'] is None else round(r['channel_mean_share_of_sse'],2)} |e_in|={r['mean_abs_input_divergence']:.3f} |tot|={r['mean_abs_total_vs_proxy']:.3f} cnt snn/loc/prox={r['mean_count_snn']:.2f}/{r['mean_count_local_twin']:.2f}/{r['mean_count_proxy']:.2f}")
