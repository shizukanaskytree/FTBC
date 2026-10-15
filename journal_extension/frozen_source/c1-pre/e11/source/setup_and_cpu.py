import ast,hashlib,json,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[1];P=E.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(P/'e4/source'))
from common import prepare
import torch
from trace_ops import counts,local_input,replay
base=json.loads((P/'e10/config/main-native.json').read_text())
keys=['workspace','parent_config','parent_runner','parent_proxy','parent_deployment','parent_predictions','calibrated_deployment','calibrated_predictions','seed','allowed_uuid']
base={k:base[k] for k in keys};raw=Path('/data7/projects/ftbc_neural_networks_outputs/c1pre-e10-20261009/main-native-v1')
base.update(prior_freeze=str(P/'e10/protocol-freeze.json'),freeze_manifest=str(E/'protocol-freeze.json'),dev_count=0,diagnostic_count=8,
    kl_bias=str(raw/'bias-full-kl-lr0.001.pt'),anchor_predictions=str(raw/'predictions.pt'),teacher_scoring=str(raw/'scoring-teacher.pt'),teacher_fit=str(raw/'fit-teacher-logits.pt'))
for phase,n in [('sanity',8),('main',256)]:
 for state in ['teacher','bn','kl']:
  c={**base,'fit_count':n,'state':state,'mode':state,'sanity':phase=='sanity','output':f'/data7/projects/ftbc_neural_networks_outputs/c1pre-e11-20261009/{phase}-{state}-v1'}
  path=E/f'config/{phase}-{state}.json';path.write_text(json.dumps(c,indent=2)+'\n')
c=json.loads((E/'config/main-teacher.json').read_text());parents=prepare(c,cpu=True)
from modules import QCFS,ParaInfNeuron
cases=0
for dim,shape in [(5,(8,3,2,2)),(3,(8,3))]:
 x=(torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32)/32-1).reshape(shape)
 ann=QCFS(up=1.25,t=8,dim=dim);k=counts(ann(x),ann,False)
 k4,_=replay(ann,x,False,4,'cpu');k8,_=replay(ann,x,False,8,'cpu');assert torch.equal(k,k4) and torch.equal(k4,k8);cases+=1
 for bias in [0.,.21]:
  module=ParaInfNeuron(2,th=1.25,init_mem=.5,dim=dim)
  module.output_bias_correction=torch.full((2,1,3,*([1]*(len(shape)-2))),bias)
  # Distinct temporal inputs exercise T-major layout, not accidental image-major repetition.
  temporal=torch.stack([x-.125,x+.125]);flat=temporal.flatten(0,1)
  mean=local_input(flat,module,True);assert torch.equal(mean,x)
  k=counts(module(flat),module,True)
  k4,ok4=replay(module,mean,True,4,'cpu');k8,ok8=replay(module,mean,True,8,'cpu')
  assert ok4 and ok8 and torch.equal(k,k4) and torch.equal(k4,k8);cases+=1
rawpred=torch.load(base['anchor_predictions'],map_location='cpu',weights_only=False)
kb=torch.load(base['kl_bias'],map_location='cpu',weights_only=False)
bn=torch.load(base['parent_deployment'],map_location='cpu',weights_only=False)
assert kb['parent_sha256']==hashlib.sha256(Path(base['parent_deployment']).read_bytes()).hexdigest()
assert kb['thresholds']==bn['interface_thresholds'] and list(kb['biases'])==parents['interfaces']
ts=torch.load(base['teacher_scoring'],map_location='cpu',weights_only=False)['fit']
tf=torch.load(base['teacher_fit'],map_location='cpu',weights_only=False)
assert len(tf['ids'])==256 and torch.equal(ts['ids'],tf['ids'])
for key in ['bn-only','full-kl-lr0.001']:
 assert torch.equal(rawpred[key]['fit']['ids'],tf['ids'])
assert not torch.cuda.is_initialized()
for p in (E/'source').glob('*.py'):ast.parse(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))
report={'passed':True,'actual_native_toy_cases':cases,'parents':parents,'cuda_initialized':False,'source_module':sys.modules['modules'].__file__,'files_sha256':{str(p):sha(p) for p in files}}
(E/'cache/cpu-check.json').write_text(json.dumps(report,indent=2)+'\n')
w=Path(c['workspace'])/'manuscript';before={key:subprocess.check_output(['git','-C',str(w),*args],text=True) for key,args in [('HEAD',['rev-parse','HEAD']),('status',['status','--porcelain=v1','--untracked-files=all'])]};before['index_sha256']=sha(w/'.git/index')
t=E/'preservation-before.json'
if t.exists():assert json.loads(t.read_text())==before
else:t.write_text(json.dumps(before,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'}))
