import sys,json,hashlib
from pathlib import Path
import torch
from norm_match import construct
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def geometry(bias,refs,theta):
 def v(d,n):return d[n][0].flatten().double()/theta[n]
 def stat(x,g,r):
  norm=float(x.norm());gn=float(g.norm());rn=float(r.norm())
  return {'norm':norm,'rms':float(x.square().mean().sqrt()),'norm_ratio_G':norm/gn if gn else None,
   'cosine_G':float(x.dot(g))/(norm*gn) if norm*gn else None,'cosine_R':float(x.dot(r))/(norm*rn) if norm*rn else None}
 names=list(bias);xx=[v(bias,n) for n in names];gg=[v(refs['G-e8']['biases'],n) for n in names];rr=[v(refs['finite-solve']['biases'],n) for n in names]
 return {'overall':stat(torch.cat(xx),torch.cat(gg),torch.cat(rr)),'layers':{n:stat(x,g,r) for n,x,g,r in zip(names,xx,gg,rr)}}
def check(phase):
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 states={};mem=[];hardware=[];cross={}
 for mode in ['reference','global','layer']:
  c=json.loads((E/'config'/f'{phase}-{mode}.json').read_text());out=Path(c['output'])
  r=json.loads((out/'result.json').read_text());assert r['status']=='completed'
  assert r['records']==(4 if phase=='main' and mode=='reference' else (1 if mode=='reference' else 2))
  assert r['teacher_unchanged'] and r['non_bias_unchanged'] and r['all_strict_reloads_passed']
  mem.append(r['max_memory_reserved']);hardware.append(r['hardware'])
  pred=torch.load(out/'predictions.pt',map_location='cpu',weights_only=False)
  meas=torch.load(out/'measurements.pt',map_location='cpu',weights_only=False)
  target=torch.load(out/'targets.pt',map_location='cpu',weights_only=False)
  rows=json.loads((out/'metrics-progress.json').read_text())
  roles=json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text());roles=roles.get('roles',roles)
  refs={n:torch.load(d['bias'],map_location='cpu',weights_only=False) for n,d in c['references'].items()}
  for row in rows:
   name=row['id'];saved=torch.load(out/f'bias-{name}.pt',map_location='cpu',weights_only=False)
   assert saved['parent_sha256']==sha(c['parent_deployment']) and saved['thresholds']==refs['finite-solve']['thresholds']
   expected=refs[name]['biases'] if name in refs else construct(refs['finite-solve']['biases'],refs['G-e8']['biases'],saved['thresholds'],mode)[0]
   assert all(torch.equal(saved['biases'][n],expected[n]) for n in expected)
   stats={}
   gpred=torch.load(c['references']['G-e8']['predictions'],map_location='cpu',weights_only=False)['G-e8']
   for role in ['fit','development']:
    q=pred[name][role];n=c['fit_count'] if role=='fit' else c['dev_count']
    ids=torch.tensor([x['index'] for x in roles[role][:n]]);labels=torch.tensor([x['label'] for x in roles[role][:n]])
    assert torch.equal(q['ids'],ids) and torch.equal(q['labels'],labels) and torch.isfinite(q['logits']).all()
    y=q['logits'].argmax(1);assert torch.equal(y,q['predictions'])
    correct=int((y==labels).sum());ce=float(torch.nn.functional.cross_entropy(q['logits'].double(),labels))
    assert correct==row['metrics'][role]['correct'] and abs(ce-row['metrics'][role]['cross_entropy'])<1e-12
    if name in refs:
     old=torch.load(c['references'][name]['predictions'],map_location='cpu',weights_only=False)[name][role]
     assert torch.equal(q['logits'],old['logits'][:n]) and torch.equal(q['ids'],old['ids'][:n])
    if name=='bn-only':
     if role in cross:assert torch.equal(q['logits'],cross[role])
     else:cross[role]=q['logits']
    bn=pred['bn-only'][role]['predictions']==labels;good=y==labels
    gp=gpred[role];assert torch.equal(gp['ids'][:n],ids) and torch.equal(gp['labels'][:n],labels)
    oldg=gp['predictions'][:n]==labels
    stats[role]={'correct':correct,'n':n,'top1':100*correct/n,'ce':ce,'repair_vs_bn':int((good&~bn).sum()),'damage_vs_bn':int((bn&~good).sum()),'repair_vs_G':int((good&~oldg).sum()),'damage_vs_G':int((~good&oldg).sum())}
   assert torch.equal(target['ids'],pred[name]['fit']['ids'])
   residuals=[]
   for n,d in meas[name].items():
    assert torch.equal(d['target_mean'],target['means'][n])
    rr=d['count'].double()/d['observations_per_channel']-d['target_mean']/d['theta']
    assert torch.allclose(rr,d['normalized_residual'],atol=1e-15,rtol=0);residuals.append(rr.abs())
   if phase=='main' and name in refs:
    old=torch.load(Path(c['targets']).with_name('measurements.pt'),map_location='cpu',weights_only=False)[name]
    assert all(torch.equal(d['count'],old[n]['count']) and torch.equal(d['target_mean'],old[n]['target_mean']) for n,d in meas[name].items())
   assert sum(len(x) for x in residuals)==7616
   states[name]={'metrics':stats,'M':float(torch.cat(residuals).mean()),'geometry':geometry(saved['biases'],refs,saved['thresholds'])}
 total_memory=hardware[0]['total_memory']
 for x in hardware:
  assert all(x[k]==hardware[0][k] for k in ['slurm_job_id','gpu_uuid','name','total_memory','torch','torchvision','cuda_runtime','cudnn']),'worker hardware differs'
 result={'passed':True,'phase':phase,'states':states,'max_memory_reserved':mem,'total_memory':total_memory,'hardware':hardware[0],'cross_worker_BN_equal':True,'saved_bias_construct_exact':True,'all_strict_reloads_passed':True,'cuda_initialized':torch.cuda.is_initialized()}
 if phase=='main':
  gains={};bn=states['bn-only'];g=states['G-e8']
  for name in ['root-global-Gnorm','root-layer-Gnorm']:
   gains[name]={}
   for role in ['fit','development']:
    gap=bn['metrics'][role]['correct']-g['metrics'][role]['correct'];assert gap>0
    gains[name][role]=(states[name]['metrics'][role]['correct']-g['metrics'][role]['correct'])/gap
  result['recovered_G_to_BN_gap']=gains
  result['E8_C1_descriptive']=all(x>=.5 for x in gains['root-layer-Gnorm'].values())
 assert not result['cuda_initialized']
 (E/f'{phase}-acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':check(sys.argv[1])
