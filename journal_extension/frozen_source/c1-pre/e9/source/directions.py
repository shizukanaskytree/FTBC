"""Ordinary low-dimensional bias parametrization; no novel optimizer claim."""
import torch

def construct(root, thresholds, mode, seed, bound_ratio=.5):
    assert mode in ('root','permuted','constant','full-kl','full-ce')
    gen=torch.Generator(device='cpu').manual_seed(seed)
    basis={}; bounds={}; meta={}
    for name,bias in root.items():
        assert torch.equal(bias,bias[:1].expand_as(bias))
        r=bias[0].flatten().double()/thresholds[name]
        assert torch.isfinite(r).all() and r.abs().max()<=bound_ratio+1e-7
        rms=r.square().mean().sqrt()
        if rms==0:
            v=torch.zeros_like(r); limit=0.
        else:
            vr=r/rms; limit=float(bound_ratio/vr.abs().max())
            permutation=torch.randperm(len(r),generator=gen)
            v=vr if mode=='root' else vr[permutation] if mode=='permuted' else torch.ones_like(r)
        basis[name]=v.float(); bounds[name]=limit
        meta[name]={'root_rms':float(rms),'coefficient_limit':limit,'channels':len(r),'active':bool(rms>0),
                    'limit_over_root_rms':limit/float(rms) if rms>0 else None,
                    'root_at_bound':bool(rms>0 and abs(limit/float(rms)-1)<1e-6)}
    return basis,bounds,meta

def bind(modules,parameters,basis,thresholds,timesteps,full=False):
    for name,p in parameters.items():
        value=p if full else p*basis[name]
        modules[name].output_bias_correction=(value*thresholds[name]).view(1,1,-1,1,1).expand(timesteps,1,-1,1,1).contiguous()

@torch.no_grad()
def clamp(parameters,bounds,full=False,bound_ratio=.5):
    for name,p in parameters.items():
        limit=bound_ratio if full else bounds[name]
        p.clamp_(-limit,limit)
