"""CPU evidence for the exact |delta| <= 0.5*theta comparison (review round 2, item a). No CUDA."""
import hashlib, json
from pathlib import Path
import torch
E = Path(__file__).resolve().parents[1]
out = {}
for which in ['sanity', 'main']:
    c = json.loads((E / 'config' / f'{which}.json').read_text())
    th = torch.load(c['p1']['parent_deployment'], map_location='cpu', weights_only=False)['interface_thresholds']
    assert th == torch.load(c['p0']['parent_deployment'], map_location='cpu', weights_only=False)['interface_thresholds'] and len(th) == 33
    exact = {n: float(torch.tensor(.5 * t, dtype=torch.float32)) == .5 * t and float(torch.tensor(t, dtype=torch.float32)) == t for n, t in th.items()}
    assert all(exact.values())
    files = {'kl': Path(c['kl_bias']), 'e7_finite_solve': Path(c['e7_reference']['output']) / 'bias-finite-solve.pt'}
    row = {'half_theta_exact_in_fp32': True}
    for k, p in files.items():
        b = torch.load(p, map_location='cpu', weights_only=False)['biases']
        ratio = max(float(v.abs().max()) / th[n] for n, v in b.items())
        assert all(float(v.abs().max()) <= .5 * th[n] for n, v in b.items()), k
        row[k] = {'file': str(p), 'max_abs_over_theta': ratio, 'exact_bound_holds': True}
    out[which] = row
assert not torch.cuda.is_initialized()
out.update(passed=True, cuda_initialized=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(E / 'cache/bound-check.json').write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out, indent=1))
