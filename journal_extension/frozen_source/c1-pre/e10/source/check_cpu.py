import ast,copy,hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 c=json.loads((E/'config/main-native.json').read_text());r=prepare(c,cpu=True)
 from modules import ParaInfNeuron,Heaviside
 from surrogate import Surrogate,install
 original_path=Path(sys.modules['modules'].__file__)
 original=ast.parse(original_path.read_text())
 cls=next(x for x in original.body if isinstance(x,ast.ClassDef) and x.name=='ParaInfNeuron')
 old=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
 new=next(x for x in ast.parse((E/'source/surrogate.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='forward')
 class Revert(ast.NodeTransformer):
  calls=0
  def visit_Call(self,node):
   if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='Surrogate':
    self.calls+=1;return ast.parse('Heaviside.apply(potential - self.v_threshold, 1.0, "triangle")',mode='eval').body
   return self.generic_visit(node)
 rev=Revert();restored=rev.visit(copy.deepcopy(new));assert rev.calls==2 and ast.dump(old)==ast.dump(restored)
 cases=0
 for theta in [.1,.6923413872718811,1.,2.2850592136383057,10.458191871643066]:
  for mode in ['native','relative-area','relative-peak']:
   x=torch.tensor([-2.,-1.,-.5,0.,.5,1.,2.],dtype=torch.float64)*theta;x.requires_grad_(True)
   weight=torch.arange(1,8,dtype=x.dtype)
   y=Surrogate.apply(x,theta,mode);(y*weight).sum().backward()
   expected=(1-x.detach().abs()/(1 if mode=='native' else theta)).clamp(min=0)
   if mode=='relative-area':expected/=theta
   assert torch.equal(x.grad,weight*expected) and torch.equal(y,(x>=0).to(x.dtype))
   if theta==1:
    z=x.detach().clone().requires_grad_(True);(Heaviside.apply(z,1.,'triangle')*weight).sum().backward();assert torch.equal(x.grad,z.grad)
   for dim,shape in [(5,(4,3,2,2)),(3,(4,3))]:
    a=ParaInfNeuron(2,th=theta,dim=dim);b=copy.deepcopy(a);install({'n':b},['n'],mode)
    # Both temporal events and neighbouring FP32 representable inputs; negative/zero drives too.
    vals=torch.tensor([-1.,0.,.25,.5,.75,1.],dtype=torch.float32)*theta
    vals=torch.cat([vals,torch.nextafter(vals,torch.full_like(vals,float('inf'))),torch.nextafter(vals,torch.full_like(vals,-float('inf')))])
    for v in vals:
     inp=torch.full(shape,float(v));assert torch.equal(a(inp),b(inp));cases+=1
 for f in (E/'source').glob('*.py'):ast.parse(f.read_text())
 for f in (E/'config').glob('*.json'):
  d=json.loads(f.read_text());assert d['mode']=='full-kl' and d['lr_ratios']==[.001] and d['epochs']==(1 if d['sanity'] else 4)
 r.update(actual_imported_source=str(original_path),actual_imported_sha256=sha(original_path),ast_only_two_calls_changed=True,hard_forward_cases=cases,analytic_backward_checked=True,theta_one_matches=True,cuda_initialized=torch.cuda.is_initialized())
 assert not r['cuda_initialized']
 files=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))+[E/'EXPERIMENT_PLAN.md']
 r['files_sha256']={str(f):sha(f) for f in files};(E/'cache/cpu-check.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ['files_sha256','interfaces']}))
if __name__=='__main__':check()
