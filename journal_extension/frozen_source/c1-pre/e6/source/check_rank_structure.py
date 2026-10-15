"""Local closed forms for time-invariant incoming gradient; not a whole-graph attribution."""
import json
from pathlib import Path
import torch
from calibrated_neuron import CalibratedResponse,base_potential
torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(731)
checks=[]
for T in (1,2,4):
 B,C,H,W=29,64,3,3
 theta=.731;tau=.1
 x=(.4+.3*torch.randn(T*B,C,H,W)).requires_grad_()
 g=torch.randn(B,C,H,W).unsqueeze(0).expand(T,B,C,H,W)
 delta=torch.zeros(C);active=torch.ones(C,dtype=torch.bool)
 y=CalibratedResponse.apply(x,delta,active,T,theta,tau,True,True)
 actual=torch.autograd.grad((y*g.reshape_as(y)).sum(),x)[0].reshape(T,B,C,H,W).sum((0,1,3,4))
 sig=torch.sigmoid((base_potential(x.detach(),T,theta)-theta)/(tau*theta))
 w=sig*(1-sig)/(tau*theta);wi=w.sum(0)
 rank=torch.arange(1,T+1).reshape(T,1,1,1,1)
 rho=(w*T/rank).sum(0)/wi
 weight=wi.sum((0,2,3));eg=(wi*g[0]).sum((0,2,3))/weight
 er=(wi*rho).sum((0,2,3))/weight
 egr=(wi*g[0]*rho).sum((0,2,3))/weight
 closed=theta*weight*(egr-eg*er)
 assert torch.allclose(actual,closed,atol=1e-10,rtol=1e-9)
 rec={'T':T,'channels':C,'covariance_max_abs_error':float((actual-closed).abs().max())}
 if T==1:assert actual.abs().max()<1e-10
 if T==2:
  A=(wi*g[0]).sum((0,2,3));Bv=(w[0]*g[0]).sum((0,2,3));pi=w[0].sum((0,2,3))/weight
  u=theta*(A+Bv);c=theta*(Bv-pi*A)
  assert torch.allclose(c,actual,atol=1e-10,rtol=1e-9)
  ratio=Bv/A
  assert torch.equal(u*c<0,(ratio>-1)&(ratio<pi))
  rec.update(local_sign_reversals=int((u*c<0).sum()),sign_reversal_rule_exact=True)
 checks.append(rec)
out={'passed':True,'checks':checks,'cuda_initialized':torch.cuda.is_initialized(),
     'scope':'same incoming g for local stop/comp decomposition; not all multilevel gradient changes attributed to local lambda'}
assert not out['cuda_initialized']
(Path(__file__).resolve().parents[1]/'cache/cpu-rank-structure.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
