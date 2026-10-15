"""E8 saved-endpoint replay; no training. GPU work only under Slurm."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e7/source'))
from run_diagnostic import prepare,teacher_targets,measure,verify_prediction_reference
import torch
from norm_match import construct

@torch.no_grad()
def run(c):
 start=time.perf_counter();r=prepare(c)
 batches=list(r.e0.loader(r.rows['fit'],r.parent,c['train_batch']))
 target=teacher_targets(r,batches)
 if not c['sanity']:
  old=torch.load(c['targets'],map_location='cpu',weights_only=False)
  assert torch.equal(old['ids'],torch.cat([b[2] for b in batches]))
  assert old['teacher_digest']==r.teacher_digest
  assert all(torch.equal(target[n].cpu(),old['means'][n]) for n in r.names)
 r.e0.atomic_torch(r.output/'targets.pt',{'means':{n:t.cpu() for n,t in target.items()},'ids':torch.cat([b[2] for b in batches])})
 loaded={}
 for name,ref in c['references'].items():
  d=torch.load(ref['bias'],map_location='cpu',weights_only=False)
  assert d['parent_sha256']==r.e0.sha256(c['parent_deployment']) and d['thresholds']==r.bn['interface_thresholds']
  assert list(d['biases'])==r.names
  loaded[name]=d['biases']
 ids=['bn-only'] if c['mode']!='reference' or c['sanity'] else list(c['references'])
 measurements={};replays={}
 for name in ids:
  r.score(name,loaded[name],{'reference_only':True},diagnostics=False)
  ref=c['references'][name]
  replays[name]=verify_prediction_reference(r,name,ref['predictions'],ref['state'])
  measurements[name]=measure(r,batches,target)
 if c['mode']!='reference':
  bias,meta=construct(loaded['finite-solve'],loaded['G-e8'],r.bn['interface_thresholds'],c['mode'])
  name='root-'+c['mode']+'-Gnorm'
  r.score(name,bias,meta,diagnostics=False)
  measurements[name]=measure(r,batches,target)
 r.e0.atomic_torch(r.output/'measurements.pt',measurements)
 r.e0.atomic_json(r.output/'reference-replays.json',replays)
 out=r.finish();out['seconds_total']=time.perf_counter()-start
 r.e0.atomic_json(r.output/'result.json',out);return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=json.loads(Path(a.config).read_text())
 try:print(json.dumps(run(c)))
 except Exception:
  Path(c['output']).mkdir(parents=True,exist_ok=True);(Path(c['output'])/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
