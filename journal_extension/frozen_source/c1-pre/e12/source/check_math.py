import json,sys
from pathlib import Path
import torch
from chunked_solver import solver,full_counts,chunked_counts,install
E=Path(__file__).resolve().parents[1];torch.set_num_threads(1);torch.manual_seed(20261009)
examples=[]
for n in [8,17,32]:
 base=torch.randn(2,n,3,4,4);base[:,0]=torch.tensor([.5,1.,1.5]).view(1,3,1,1)
 theta=1.;target=torch.tensor([-.1,.37,1.1],dtype=torch.float64)
 expected=solver.solve(base,target,theta)
 for delta in [torch.zeros(3),torch.tensor([-.5,0.,.5])]:assert torch.equal(full_counts(base,delta,theta),chunked_counts(base,delta,theta))
 install();actual=solver.solve(base,target,theta)
 assert torch.equal(actual[0],expected[0]) and all(torch.equal(actual[1][k],v) if isinstance(v,torch.Tensor) else actual[1][k]==v for k,v in expected[1].items())
 solver.counts=full_counts;examples.append(n)
# Direct transfer is equivalent only if the two response-increment maps coincide at that delta.
# Finite scalar count witness, fixed positive input affine transformation; no general classification claim.
def response(x,d):
 base=torch.stack([2*x+.5,x+.25])
 return ((base+d)>=1).double().mean()
witness=None
values=torch.arange(-32,49,dtype=torch.float32)/32
for k in range(300):
 x=values[torch.randint(len(values),(16,))];z=.75*x+.125
 a=torch.clamp(torch.floor(x*8+.5)/8,0,1).double().mean()
 p0=response(x,0);p1=response(z,0);m=a+p1-p0
 ds=torch.arange(-64,65,dtype=torch.float32)/128
 d0=min(ds.tolist(),key=lambda d:(float(abs(response(x,d)-a)),abs(d),d))
 d1=min(ds.tolist(),key=lambda d:(float(abs(response(z,d)-m)),abs(d),d))
 if abs(response(z,d1)-m)<abs(response(z,d0)-m):
  e0=response(x,d0)-a;e1=response(z,d1)-m
  lhs=(response(z,d1)-response(x,d0))-(p1-p0)
  assert lhs==e1-e0
  witness={'x':x.tolist(),'z':z.tolist(),'a':float(a),'p0':float(p0),'p1':float(p1),'m':float(m),'source_delta':d0,'relative_delta':d1,'transfer_residual':float(response(z,d0)-m),'relative_residual':float(e1),'source_residual':float(e0),'identity_lhs':float(lhs),'identity_rhs':float(e1-e0),'scope':'finite FP32 scalar witness, not full-network superiority'};break
assert witness is not None and not torch.cuda.is_initialized()
(E/'cache/math-check.json').write_text(json.dumps({'passed':True,'chunk_sizes_tested':examples,'full_vs_chunked_solver_exact':True,'residual_identity_witness':witness,'cuda_initialized':False},indent=2)+'\n')
print(json.dumps(witness))
