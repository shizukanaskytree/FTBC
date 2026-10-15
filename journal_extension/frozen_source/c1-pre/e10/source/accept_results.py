"""CPU acceptance: identities, actual bias, exact native reproduction, raw outcomes."""
import hashlib,json,sys
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
MODES=['native','relative-area','relative-peak']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check(phase):
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 states={};preds={};mem=[];hw=[];anchor=None;teacherfit=None
 for mode in MODES:
  c=json.loads((E/f'config/{phase}-{mode}.json').read_text());out=Path(c['output'])
  result=json.loads((out/'result.json').read_text());assert result['status']=='completed' and result['surrogate']==mode
  assert result['teacher_unchanged'] and result['non_bias_unchanged'] and result['all_strict_reloads_passed']
  mem.append(result['max_memory_reserved']);hw.append(result['hardware'])
  pred=torch.load(out/'predictions.pt',map_location='cpu',weights_only=False);assert set(pred)=={'bn-only','full-kl-lr0.001'} and result['records']==2
  rows=json.loads((out/'metrics-progress.json').read_text());assert {x['id'] for x in rows}==set(pred)
  fit_teacher=torch.load(out/'fit-teacher-logits.pt',map_location='cpu',weights_only=False)
  if teacherfit is None:teacherfit=fit_teacher
  else:assert teacherfit['teacher_digest']==fit_teacher['teacher_digest'] and torch.equal(teacherfit['ids'],fit_teacher['ids']) and torch.equal(teacherfit['logits'],fit_teacher['logits'])
  parent=json.loads(Path(c['parent_config']).read_text());roles=json.loads(Path(parent['dataset_manifest']).read_text());roles=roles.get('roles',roles)
  name='full-kl-lr0.001';params=torch.load(out/f'parameters-{name}.pt',map_location='cpu',weights_only=False)
  actual=torch.load(out/f'bias-{name}.pt',map_location='cpu',weights_only=False);bn=torch.load(out/'bias-bn-only.pt',map_location='cpu',weights_only=False)
  assert actual['parent_sha256']==bn['parent_sha256']==sha(c['parent_deployment']) and actual['thresholds']==bn['thresholds']
  steps=c['fit_count']//8*c['epochs'];assert params['steps']==steps and params['full'] and params['objective']=='kl' and params['lr']==.001
  assert sum(x.numel() for x in params['parameters'].values())==7616 and set(params['parameters'])==set(actual['biases'])==set(bn['biases'])
  for n,p in params['parameters'].items():
   assert torch.isfinite(p).all() and p.abs().max()<=.5
   want=(p*actual['thresholds'][n]).view(1,1,-1,1,1).expand_as(actual['biases'][n]).contiguous()
   assert torch.equal(want,actual['biases'][n]) and not torch.count_nonzero(bn['biases'][n])
  trace=json.loads((out/'traces.json').read_text());assert set(trace)=={name} and len(trace[name])==steps
  assert [t['step'] for t in trace[name]]==list(range(1,steps+1))
  assert all(torch.isfinite(torch.tensor([t['loss'],t['gradient_norm']])).all() for t in trace[name])
  assert any(t['gradient_norm']>0 for t in trace[name])
  if anchor is None:anchor=pred['bn-only']
  for role,num in [('fit',c['fit_count']),('development',c['dev_count'])]:
   ids=torch.tensor([v['index'] for v in roles[role][:num]]);labels=torch.tensor([v['label'] for v in roles[role][:num]])
   for row in rows:
    q=pred[row['id']][role];assert torch.equal(q['ids'],ids) and torch.equal(q['labels'],labels) and torch.isfinite(q['logits']).all()
    assert torch.equal(q['predictions'],q['logits'].argmax(1))
    good=q['predictions']==labels;ce=float(torch.nn.functional.cross_entropy(q['logits'].double(),labels))
    assert int(good.sum())==row['metrics'][role]['correct'] and abs(ce-row['metrics'][role]['cross_entropy'])<1e-12
   assert torch.equal(anchor[role]['logits'],pred['bn-only'][role]['logits'])
  if mode=='native' and phase=='main':
   old=Path('/data7/projects/ftbc_neural_networks_outputs/c1pre-e9-20261009/main-full-kl-v1')
   bias_old=torch.load(old/f'bias-{name}.pt',map_location='cpu',weights_only=False)['biases']
   pred_old=torch.load(old/'predictions.pt',map_location='cpu',weights_only=False)[name]
   assert all(torch.equal(v,bias_old[n]) for n,v in actual['biases'].items())
   assert all(torch.equal(q['logits'],pred_old[r]['logits']) for r,q in pred[name].items())
  preds[mode]=pred[name];assert set(params['nonzero_gradient_seen'])==set(actual['biases']) and set(params['first_gradient_norms'])==set(actual['biases'])
  assert all(torch.isfinite(torch.tensor(v)) for v in params['first_gradient_norms'].values())
  states[mode]={'nonzero_gradient_interfaces':sum(params['nonzero_gradient_seen'].values()),'solver_seconds':next(x for x in rows if x['id']==name)['detail']['solver_seconds']}
  if mode=='native':teacher=torch.load(out/'scoring-teacher.pt',map_location='cpu',weights_only=False)
 for h in hw:assert all(h[k]==hw[0][k] for k in ['slurm_job_id','gpu_uuid','name','total_memory','torch','torchvision','cuda_runtime','cudnn'])
 preds['bn-only']=anchor
 for mode,p in preds.items():
  states.setdefault(mode,{})['metrics']={}
  for role,q in p.items():
   t=teacher[role];assert torch.equal(q['ids'],t['ids']) and torch.equal(q['labels'],t['labels'])
   good=q['predictions']==q['labels'];stats={'n':len(good),'correct':int(good.sum()),'top1':100*float(good.double().mean()),'ce':float(torch.nn.functional.cross_entropy(q['logits'].double(),q['labels'])),'kl':float(torch.nn.functional.kl_div(torch.log_softmax(q['logits'].double(),1),torch.softmax(t['logits'].double(),1),reduction='batchmean'))}
   for other in ['bn-only','native']:
    g=preds[other][role]['predictions']==q['labels'];stats['paired_vs_'+other]={'repair':int((good&~g).sum()),'damage':int((~good&g).sum())}
   states[mode]['metrics'][role]=stats
 rel=states['relative-area']['metrics']['development'];base=states['native']['metrics']['development']
 result={'passed':True,'phase':phase,'states':states,'C1':None if phase=='sanity' else rel['correct']>=base['correct']+3 and rel['ce']<=base['ce'],'native_bitwise_reproduction':phase=='main','hard_forward_BN_identity':True,'actual_bias_reconstruction':True,'strict_reload':True,'max_memory_reserved':mem,'total_memory':hw[0]['total_memory'],'hardware':hw[0],'cuda_initialized':torch.cuda.is_initialized()}
 assert not result['cuda_initialized'];(E/f'{phase}-acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'passed':True,'phase':phase,'C1':result['C1'],'dev':{k:v['metrics']['development']['correct'] for k,v in states.items()},'max_reserved_GiB':[v/2**30 for v in mem]}))
if __name__=='__main__':check(sys.argv[1])
