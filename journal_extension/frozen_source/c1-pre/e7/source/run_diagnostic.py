"""One frozen whole-network forward calibration diagnostic; GPU via Slurm only."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare
import torch
from solver import native_base,hard,counts,solve

class PrefixDone(Exception):pass

@torch.no_grad()
def teacher_targets(r,batches):
 totals={};sizes={};handles=[]
 def hook(n):
  def f(_,args,y):
   t=y.double().sum((0,2,3));totals[n]=totals.get(n,0)+t
   sizes[n]=sizes.get(n,0)+y.numel()//y.shape[1]
  return f
 for n in r.names:handles.append(dict(r.teacher.named_modules())[n].register_forward_hook(hook(n)))
 try:
  for x,_,_ in batches:r.e0.logits(r.teacher,x.cuda(),r.parent)
 finally:
  for h in handles:h.remove()
 assert set(totals)==set(r.names)
 return {n:totals[n]/sizes[n] for n in r.names}

@torch.no_grad()
def measure(r,batches,targets):
 raw={};handles=[];order=[]
 def hook(n):
  def f(m,args,y):
   base=native_base(m,args[0]);d=m.output_bias_correction[0].flatten()
   assert torch.equal(hard(base,d,m.v_threshold).reshape_as(y),y),'native replay mismatch'
   cnt=torch.count_nonzero(y.reshape(m.T,-1,*y.shape[1:]),dim=(0,1,3,4))
   size=y.numel()//y.shape[1]
   if n not in raw:raw[n]={'count':cnt,'n':size};order.append(n)
   else:raw[n]['count']+=cnt;raw[n]['n']+=size
  return f
 for n in r.names:handles.append(r.modules[n].register_forward_hook(hook(n)))
 try:
  for x,_,_ in batches:r.e0.logits(r.model,x.cuda(),r.parent,True)
 finally:
  for h in handles:h.remove()
 assert order==r.names,'state order differs from actual forward traversal'
 return {n:{'count':v['count'].cpu(),'observations_per_channel':v['n'],
  'theta':r.bn['interface_thresholds'][n],'target_mean':targets[n].cpu(),
  'normalized_residual':(v['count'].double()/v['n']-targets[n]/r.bn['interface_thresholds'][n]).cpu()}
  for n,v in raw.items()}

@torch.no_grad()
def fit_one(r,batches,targets,name,c):
 m=r.modules[name];cache=None;offset=0;T=m.T;theta=m.v_threshold
 assert m.__class__.__name__=='ParaInfNeuron' and m.dim==5
 assert not torch.count_nonzero(m.output_bias_correction)
 def hook(_,args,y):
  nonlocal cache,offset
  base=native_base(m,args[0]);d=m.output_bias_correction[0].flatten()
  assert torch.equal(hard(base,d,theta).reshape_as(y),y),'prefix native replay mismatch'
  if cache is None:
   shape=(T,c['fit_count'],*base.shape[2:]);nbytes=T*c['fit_count']*base[0,0].numel()*4
   assert 6*nbytes+8*2**30<.85*r.hardware['total_memory'],'projected memory exceeds conservative envelope'
   cache=torch.empty(shape,device=base.device,dtype=base.dtype)
  b=base.shape[1];cache[:,offset:offset+b].copy_(base);offset+=b
  raise PrefixDone()
 h=m.register_forward_hook(hook);start=time.perf_counter()
 try:
  for x,_,_ in batches:
   try:r.e0.logits(r.model,x.cuda(),r.parent,True)
   except PrefixDone:pass
   else:raise AssertionError('prefix hook did not fire')
 finally:h.remove()
 assert offset==c['fit_count']
 d,info=solve(cache,targets[name],theta,c['bisection_steps'],c['bound_ratio'])
 m.output_bias_correction.copy_(d.reshape(1,1,-1,1,1).expand_as(m.output_bias_correction))
 torch.cuda.synchronize();info['seconds']=time.perf_counter()-start
 info['base_shape']=list(cache.shape);info['cache_bytes']=cache.numel()*cache.element_size()
 del cache
 return info

def verify_prediction_reference(r,ident,path,state):
 old=torch.load(path,map_location='cpu',weights_only=False)[state]
 report={}
 for role in ['fit','development']:
  p=r.predictions[ident][role];o=old[role];n=len(p['ids'])
  assert torch.equal(p['ids'],o['ids'][:n]) and torch.equal(p['labels'],o['labels'][:n])
  assert torch.equal(p['logits'],o['logits'][:n]),'same-card/batch historical replay changed'
  report[role]={'ids_labels_and_logits_bitwise':True,'n':n}
 return report

@torch.no_grad()
def run(c):
 start=time.perf_counter();r=prepare(c);batches=list(r.e0.loader(r.rows['fit'],r.parent,c['train_batch']))
 assert torch.equal(torch.cat([x[2] for x in batches]),torch.tensor([v['index'] for v in r.rows['fit']]))
 assert all(len(x[2])==c['train_batch'] for x in batches)
 targets=teacher_targets(r,batches)
 r.e0.atomic_torch(r.output/'targets.pt',{'means':{n:x.cpu() for n,x in targets.items()},'ids':torch.cat([x[2] for x in batches]),
   'labels':torch.cat([x[1] for x in batches]),'teacher_digest':r.teacher_digest})
 measures={};replays={}
 for ident,item in c['references'].items():
  if ident=='bn-only':bias=r.bn['runtime_biases']
  else:
   saved=torch.load(item['bias'],map_location='cpu',weights_only=False)
   assert saved['parent_sha256']==r.e0.sha256(c['parent_deployment']) and saved['thresholds']==r.bn['interface_thresholds']
   bias=saved['biases']
  r.score(ident,bias,{'reference_only':True,'source':item['bias']},diagnostics=False)
  replays[ident]=verify_prediction_reference(r,ident,item['predictions'],item['state'])
  measures[ident]=measure(r,batches,targets)
  r.e0.atomic_torch(r.output/'measurements.pt',measures)
 r.assign(r.bn['runtime_biases']);fits={}
 for i,name in enumerate(r.names):
  fits[name]=fit_one(r,batches,targets,name,c)
  r.e0.atomic_torch(r.output/'fit-progress.pt',{'layers':fits,'biases':r.snapshot(),'completed_interfaces':i+1})
  r.progress('fitted_layer',index=i+1,name=name,mean_residual=float(fits[name]['normalized_residual'].abs().mean()),seconds=fits[name]['seconds'])
 final=r.snapshot();r.score('finite-solve',final,{'bisection_steps':c['bisection_steps'],'fit_count':c['fit_count'],'bound_ratio':c['bound_ratio']},diagnostics=False)
 measures['finite-solve']=measure(r,batches,targets)
 for n in r.names:
  assert torch.equal(measures['finite-solve'][n]['count'],fits[n]['after_count']),'fitted vs full-network response differs'
  assert measures['finite-solve'][n]['observations_per_channel']==fits[n]['observations_per_channel']
  assert torch.equal(final[n][0].flatten(),fits[n]['delta'])
 r.e0.atomic_torch(r.output/'measurements.pt',measures);r.e0.atomic_json(r.output/'reference-replays.json',replays)
 out=r.finish();out.update(seconds_total=time.perf_counter()-start,full_fit_replay_equal=True,fit_interfaces=len(fits),finite_bisection_not_exact=True)
 r.e0.atomic_json(r.output/'result.json',out)
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=json.loads(Path(a.config).read_text())
 try:print(json.dumps(run(c)))
 except Exception:
  Path(c['output']).mkdir(exist_ok=True,parents=True);(Path(c['output'])/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
