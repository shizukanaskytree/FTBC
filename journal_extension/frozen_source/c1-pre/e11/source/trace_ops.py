"""Count/trace layout helpers; reuse actual frozen activation forward."""
import torch

def local_input(x, module, spiking):
    return x.reshape(module.T, x.shape[0]//module.T, *x.shape[1:]).mean(0) if spiking else x

def counts(y, module, spiking):
    assert torch.isfinite(y).all()
    if spiking:
        th = float(module.v_threshold)
        assert ((y == 0) | (y == th)).all(), 'Non-binary native spikes'
        return (y != 0).reshape(module.T, y.shape[0]//module.T, *y.shape[1:]).sum(0).to(torch.uint8)
    assert not module.is_cab and not module.cab_inf and not module.is_relu
    q = module.t; assert q == 8
    k = (y/module.up*q).round()
    assert ((k>=0)&(k<=q)).all() and torch.equal(k/q*module.up,y), 'Invalid Q8 levels'
    return k.to(torch.uint8)

@torch.no_grad()
def replay(module, z, spiking, batch, device):
    outputs=[]; identical_mean=True
    for part in z.split(batch):
        part=part.to(device)
        x=part.repeat(module.T,*([1]*(part.ndim-1))) if spiking else part
        identical_mean &= torch.equal(local_input(x,module,spiking),part)
        outputs.append(counts(module(x),module,spiking).cpu())
    return torch.cat(outputs), identical_mean

def trace_difference(a,b):
    x,y=a['input'],b['input']; k,l=a['count'],b['count']
    assert x.shape==y.shape==k.shape==l.shape
    d=(x-y).double().abs(); changed=k!=l
    return {'input_max_abs':float(d.max()),'input_mean_abs':float(d.mean()),
            'input_changed_elements':int((x!=y).sum()),'count_changed_elements':int(changed.sum()),
            'count_affected_images':int(changed.flatten(1).any(1).sum()),
            'count_changed_with_equal_input_elements':int((changed&(x==y)).sum()),
            'nonfinite_input_elements':int((~torch.isfinite(x)|~torch.isfinite(y)).sum()),
            'total_count_elements':k.numel()}
