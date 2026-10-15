import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare
from norm_match import construct
from accept_results import geometry
import torch
torch.set_num_threads(4)
c=json.loads((E/'config/main-reference.json').read_text());r=prepare(c,cpu=True)
data={n:torch.load(d['bias'],map_location='cpu',weights_only=False) for n,d in c['references'].items()}
root=data['finite-solve'];old=data['G-e8']
assert root['parent_sha256']==old['parent_sha256'] and root['thresholds']==old['thresholds']
checks={}
for mode in ['global','layer']:
 out,meta=construct(root['biases'],old['biases'],root['thresholds'],mode);checks[mode]=meta
 g=geometry(out,data,root['thresholds'])
 assert len(g['layers'])==33 and g['overall']['norm']>0
 if mode=='global':assert abs(g['overall']['cosine_R']-1)<1e-12
 if mode=='layer':assert all(abs(x['norm_ratio_G']-1)<1e-6 for x in g['layers'].values())
 # Independent norm calculation across actual saved first-time channel vectors.
 before=[];after=[]
 for n in out:
  th=root['thresholds'][n];x=out[n][0].double()/th;y=old['biases'][n][0].double()/th
  before.append(y.flatten());after.append(x.flatten())
  if mode=='layer':assert abs(float(x.square().sum().sqrt()/y.square().sum().sqrt())-1)<1e-6
 if mode=='global':assert abs(float(torch.cat(after).square().sum().sqrt()/torch.cat(before).square().sum().sqrt())-1)<1e-6
z=torch.zeros(2,1,2,1,1);o,meta=construct({'a':z},{'a':z},{'a':1.},'layer');assert not o['a'].count_nonzero()
try:construct({'a':z},{'a':z+.1},{'a':1.},'layer')
except AssertionError:pass
else:raise AssertionError('degenerate infeasible direction not rejected')
assert not torch.cuda.is_initialized()
g=geometry(data['bn-only']['biases'],data,root['thresholds']);assert g['overall']['cosine_G'] is None and g['overall']['rms']==0
files=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))+[E/'EXPERIMENT_PLAN.md',E/'analyze.py']
identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(E/'cache/cpu-check.json').write_text(json.dumps({'passed':True,'runtime':r,'geometry':checks,'zero_direction_checked':True,'cuda_initialized':False,'files_sha256':identities},indent=2)+'\n')
print(json.dumps({'passed':True,'channels':r['channels'],'global_alpha':next(iter(checks['global']['scale'].values())),'max_layer_norm_error':max(checks['layer']['per_layer_norm_relative_error'].values()),'cuda_initialized':False}))
