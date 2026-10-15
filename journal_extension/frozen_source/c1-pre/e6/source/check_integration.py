"""Reproduce the CPU parent identity and nonzero-bias native FP32 checks."""
import hashlib,json
from pathlib import Path
import torch
from run_method import common
from calibrated_neuron import CalibratedResponse
E=Path(__file__).resolve().parents[1]
c=json.loads((E/'configs/sanity.json').read_text());r=common.prepare(c,cpu=True)
from modules import ParaInfNeuron
checks=[]
for T in (1,2,4):
 for theta in (.2,.731,1.,3.4156):
  m=ParaInfNeuron(T,th=theta,dim=5)
  x=torch.linspace(-theta,2*theta,24*T*3*2*2).reshape(24*T,3,2,2)
  d=torch.tensor([-.5*theta,0.,.3*theta])
  m.output_bias_correction=d.reshape(1,1,3,1,1).repeat(T,1,1,1,1)
  z=CalibratedResponse.apply(x,d,torch.zeros(3,dtype=torch.bool),T,theta,.1,False,False)
  assert torch.equal(m(x),z),(T,theta)
  checks.append({'T':T,'theta':theta,'nonzero_delta':d.tolist(),'exact':True})
assert not torch.cuda.is_initialized()
paths=[Path(__file__),E/'source/calibrated_neuron.py',E/'source/run_method.py',E.parent/'e4/source/common.py',E.parent/'source-snapshot/parallel/modules.py']
out={'runtime':r,'real_neuron_fp32_checks':checks,'cuda_initialized':False,
     'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(E/'cache/cpu-integration.json').write_text(json.dumps(out,indent=2)+'\n');print('CPU parent and 12 native nonzero-bias FP32 checks passed')
