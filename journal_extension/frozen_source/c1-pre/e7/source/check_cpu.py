import hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare
from solver import native_base,hard,counts,solve
c=json.loads((E/'config/main.json').read_text());parent=prepare(c,cpu=True)
from modules import ParaInfNeuron
g=torch.Generator().manual_seed(20261009);trials=0
for T in [1,2,4,8]:
 for theta in [.125,1.,2.5]:
  m=ParaInfNeuron(T,th=theta,init_mem=.5,dim=5)
  x=torch.randn(T*3,4,2,2,generator=g)*theta
  x[0,0]=0;x[1,1]=-theta
  for d in [-.5,0,.125,.5]:
   delta=torch.full((4,),d*theta)
   m.output_bias_correction=delta.reshape(1,1,4,1,1).repeat(T,1,1,1,1)
   y=m(x);b=native_base(m,x)
   assert torch.equal(hard(b,delta,theta).reshape_as(y),y)
   assert torch.equal(counts(b,delta,theta),torch.count_nonzero(y.reshape(T,3,4,2,2),dim=(0,1,3,4)))
   trials+=1
# Finite dyadic cases including tied events, negative/zero drives and out-of-support targets.
exhaustive=0
for trial in range(30):
 base=torch.randint(-8,25,(2,3,3,2,2),generator=g).float()/16
 target=torch.tensor([-.2,.35,1.2],dtype=torch.float64) if trial%3==0 else torch.rand(3,generator=g).double()
 d,r=solve(base,target,1.)
 for ch in range(3):
  events=(1-base[:,:,ch]).flatten();points=torch.cat([events,torch.nextafter(events,torch.full_like(events,-float('inf'))),torch.tensor([-.5,0,.5])]).clamp(-.5,.5)
  errors=[]
  for p in points:
   cnt=counts(base[:,:,ch:ch+1],p.reshape(1),1.)
   errors.append(float((cnt.double()/r['observations_per_channel']-target[ch]).abs()))
  assert float(r['normalized_residual'][ch].abs())<=min(errors)+1e-15
  exhaustive+=1
base=torch.zeros(2,3,2,2,2);d,r=solve(base,torch.zeros(2),1.)
assert torch.equal(d,torch.zeros(2)) and r['at_lower_mean'].all() and not r['bracketed'].any()
assert not torch.cuda.is_initialized()
out={'passed':True,'parent_check':parent,'native_parameterization_cases':trials,'small_enumeration_channels':exhaustive,
 'zero_tie_boundary_checked':True,'not_global_optimality_proof':True,'cuda_initialized':False,
 'files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),E/'source/solver.py',E/'source/run_diagnostic.py']}}
(E/'cache/cpu-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
