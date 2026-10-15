"""CPU independent derivative checks, not evidence of hard-network accuracy."""
import json,hashlib
from pathlib import Path
import torch
from calibrated_neuron import calibrated, base_potential, response, CalibratedResponse

torch.set_num_threads(2)
torch.manual_seed(731)
torch.set_default_dtype(torch.float64)


def main():
    x = .25 + .1 * torch.randn(2, 7, 3, 2, 2)
    x = x.reshape(14, 3, 2, 2)
    a = .15 * torch.randn(3, 3)
    b = .2 * torch.randn(3, 3)
    task = torch.randn_like(x)
    target1 = response(base_potential(x, 2, 1.), torch.zeros(3), 1., .1, True).mean((0,1,3,4))
    def mix(z, m): return torch.einsum('oc,bchw->bohw', m, z)
    mid = response(base_potential(x, 2, 1.), torch.zeros(3), 1., .1, True).reshape_as(x)
    v = .25 + mix(mid, a) + mix(x, b)
    target2 = response(base_potential(v, 2, 1.), torch.zeros(3), 1., .1, True).mean((0,1,3,4))
    def implicit_graph(beta, comp=True):
        z = x + beta[:3].reshape(1,3,1,1)
        y, d1, act1, err1 = calibrated(z, target1, compensate=comp, soft=True, steps=64)
        v = .25 + mix(y,a) + mix(z,b) + beta[3:].reshape(1,3,1,1)
        y2, d2, act2, err2 = calibrated(v, target2, compensate=comp, soft=True, steps=64)
        assert act1.all() and act2.all()
        return (y2*task).sum(), torch.cat([d1,d2]), max(err1.max().item(),err2.max().item())
    p = (.01*torch.randn(6)).requires_grad_()
    loss, root, err = implicit_graph(p)
    local = torch.autograd.grad(loss,p)[0]
    eps = 1e-5
    fd = torch.stack([(implicit_graph(p.detach()+torch.eye(6)[i]*eps)[0]-
                       implicit_graph(p.detach()-torch.eye(6)[i]*eps)[0])/(2*eps) for i in range(6)])
    def explicit(beta, d):
        z=x+beta[:3].reshape(1,3,1,1)
        y=response(base_potential(z,2,1.),d[:3],1.,.1,True)
        v=.25+mix(y.reshape_as(x),a)+mix(z,b)+beta[3:].reshape(1,3,1,1)
        y2=response(base_potential(v,2,1.),d[3:],1.,.1,True)
        F=torch.cat([y.mean((0,1,3,4))-target1,y2.mean((0,1,3,4))-target2])
        return (y2.reshape_as(x)*task).sum(), F
    d=root.detach().requires_grad_()
    Jp,Jd=torch.autograd.functional.jacobian(lambda beta,delta:explicit(beta,delta)[1],(p,d))
    gp,gd=torch.autograd.grad(explicit(p,d)[0],(p,d))
    dense=gp-gd@torch.linalg.solve(Jd,Jp)
    assert torch.allclose(local,fd,atol=1e-7,rtol=1e-6), (local,fd)
    assert torch.allclose(local,dense,atol=1e-10,rtol=1e-9), (local,dense)
    stop=torch.autograd.grad(implicit_graph(p,False)[0],p)[0]
    # T=1 global per-channel shift is completely absorbed at an interior root.
    xx=x[:7]; q=torch.zeros(3,requires_grad=True)
    t=response(base_potential(xx,1,1.),torch.zeros(3),1.,.1,True).mean((0,1,3,4))
    yy,_,ac,_=calibrated(xx+q.reshape(1,3,1,1),t,T=1,soft=True,steps=64)
    tq=torch.autograd.grad((yy*task[:7]).sum(),q)[0]
    assert ac.all() and tq.abs().max()<1e-10
    # Hard FP32: equality events, negative/zero drive, unbracketed targets, tied support.
    hard_checks=0
    for T in (1,2,4):
        xx=torch.tensor([-.5,0.,.25,.5,1.],dtype=torch.float32).reshape(5,1,1,1).repeat(T,1,1,1)
        for target in (-.1,0.,.3,1.,1.1):
            yy,dd,ac,ee=calibrated(xx,torch.tensor([target],dtype=torch.float32),T=T)
            manual=response(base_potential(xx,T,1.),dd,1.,.1,False).reshape_as(xx)
            assert torch.equal(yy,manual) and dd.abs().max()<=.5
            assert torch.isin(yy,torch.tensor([0.,1.])).all()
            assert torch.isfinite(ee).all()
            if target<0 or target>1: assert not ac.any()
            hard_checks+=1
    # At a bound, compensation is disabled; backward equals the ordinary smooth surrogate.
    xx=x.clone().requires_grad_(); d=torch.full((3,),.5)
    hard=CalibratedResponse.apply(xx,d,torch.zeros(3,dtype=torch.bool),2,1.,.1,True,False)
    g1=torch.autograd.grad((hard*task).sum(),xx)[0]
    sm=response(base_potential(xx,2,1.),d,1.,.1,True).reshape_as(xx)
    g2=torch.autograd.grad((sm*task).sum(),xx)[0]
    assert torch.allclose(g1,g2,atol=1e-12,rtol=1e-12)
    dd=torch.tensor([-.1,0.,.12],requires_grad=True)
    yy=CalibratedResponse.apply(x,dd,torch.zeros(3,dtype=torch.bool),2,.7,.1,False,False)
    actual_dd=torch.autograd.grad((yy*task).sum(),dd)[0]
    smooth=response(base_potential(x,2,.7),dd,.7,.1,True).reshape_as(x)
    expected_dd=torch.autograd.grad((smooth*task).sum(),dd)[0]
    assert torch.allclose(actual_dd,expected_dd,atol=1e-11,rtol=1e-11)
    assert not torch.cuda.is_initialized()
    return {'passed':True,'cuda_initialized':False,'smooth_graph_root_residual':err,
            'smooth_graph_vs_finite_difference_max_abs':float((local-fd).abs().max()),
            'smooth_graph_vs_dense_ift_max_abs':float((local-dense).abs().max()),
            'compensated_gradient':local.tolist(),'stop_gradient':stop.tolist(),
            't1_uniform_beta_max_abs':float(tq.abs().max()),'hard_edge_checks':hard_checks,
            'free_delta_gradient_max_error':float((actual_dd-expected_dd).abs().max()),
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('calibrated_neuron.py')]},
            'claim_scope':'smooth interior roots only; hard network uses surrogate and remains untested'}


if __name__=='__main__':
    out=main(); print(json.dumps(out,indent=2))
    (Path(__file__).resolve().parents[1]/'cache/cpu-math.json').write_text(json.dumps(out,indent=2)+'\n')
