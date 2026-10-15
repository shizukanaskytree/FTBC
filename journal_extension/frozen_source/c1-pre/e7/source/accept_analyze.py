"""CPU-only validation and fixed descriptive interpretation."""
import json,sys,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def check(which):
 c=json.loads((E/'config'/f'{which}.json').read_text());o=Path(c['output'])
 for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
 r=json.loads((o/'result.json').read_text());assert r['status']=='completed' and r['fit_interfaces']==33 and r['records']==5
 assert r['teacher_unchanged'] and r['non_bias_unchanged'] and r['all_strict_reloads_passed'] and r['full_fit_replay_equal']
 p=torch.load(o/'predictions.pt',map_location='cpu',weights_only=False);m=torch.load(o/'measurements.pt',map_location='cpu',weights_only=False)
 fits=torch.load(o/'fit-progress.pt',map_location='cpu',weights_only=False);targets=torch.load(o/'targets.pt',map_location='cpu',weights_only=False)
 records=json.loads((o/'metrics-progress.json').read_text());roles=json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text())
 if 'roles' in roles:roles=roles['roles']
 rows=[];base=p['bn-only'];N=c['fit_count']
 for record in records:
  ident=record['id'];assert record['strict_reload'] and record['non_bias_frozen']
  saved=torch.load(o/f'bias-{ident}.pt',map_location='cpu',weights_only=False)
  assert saved['parent_sha256']==sha(c['parent_deployment'])
  assert len(saved['biases'])==len(m[ident])==33
  for n,v in saved['biases'].items():
   assert torch.isfinite(v).all() and torch.equal(v,v[:1].expand_as(v))
   assert float(v.abs().max())<=.500001*saved['thresholds'][n]
  metrics={}
  for role in ['fit','development']:
   q=p[ident][role];n=c['fit_count'] if role=='fit' else c['dev_count']
   expected=torch.tensor([x['index'] for x in roles[role][:n]])
   labels=torch.tensor([x['label'] for x in roles[role][:n]])
   assert torch.equal(q['ids'],expected) and torch.equal(q['labels'],labels)
   pred=q['logits'].argmax(1);assert torch.equal(pred,q['predictions']) and torch.isfinite(q['logits']).all()
   correct=int((pred==labels).sum());ce=float(torch.nn.functional.cross_entropy(q['logits'].double(),labels))
   assert correct==record['metrics'][role]['correct'] and abs(ce-record['metrics'][role]['cross_entropy'])<1e-12
   old=base[role]['predictions']==labels;new=pred==labels
   metrics[role]={'correct':correct,'n':n,'top1':100*correct/n,'ce':ce,'repair_vs_bn':int((new&~old).sum()),'damage_vs_bn':int((old&~new).sum())}
  residuals=[];layer_means=[];near=[];per_layer={}
  for name,d in m[ident].items():
   assert torch.equal(d['target_mean'],targets['means'][name])
   rr=d['count'].double()/d['observations_per_channel']-d['target_mean']/d['theta']
   assert torch.allclose(rr,d['normalized_residual'],atol=1e-15,rtol=0)
   residuals.append(rr.abs());layer_means.append(float(rr.abs().mean()));near.extend((rr.abs()<=1/d['observations_per_channel']+1e-7).tolist())
   per_layer[name]={'channels':len(rr),'M':float(rr.abs().mean()),'max_absolute_residual':float(rr.abs().max())}
   if ident=='finite-solve':
    f=fits['layers'][name];assert torch.equal(d['count'],f['after_count'])
    assert torch.equal(saved['biases'][name][0].flatten(),f['delta'])
    outside=f['below_range']|f['above_range'];jump=(~outside)&f['bracketed']&f['float_adjacent']&((f['hi_count']-f['lo_count'])>1)
    other=~(outside|jump)
    per_layer[name].update(residual_partition={key:{'channels':int(mask.sum()),'absolute_residual_sum':float(rr[mask].abs().sum())} for key,mask in [('outside_bound_support',outside),('bracketed_adjacent_count_jump',jump),('other_unresolved',other)]},
      selected_zero=int((f['delta']==0).sum()),selected_boundary=int((f['delta'].abs()==.5*f['theta']).sum()),
      maximum_lo_hi_count_gap=int((f['hi_count']-f['lo_count']).max()),
      mean_local_count_change=float((f['after_count']-f['before_count']).abs().double().mean()/f['observations_per_channel']),
      mean_normalized_abs_delta=float(f['delta'].double().abs().mean()/f['theta']),
      max_normalized_abs_delta=float(f['delta'].double().abs().max()/f['theta']))
  z=torch.cat(residuals);assert len(z)==7616
  rows.append({'state':ident,'metrics':metrics,'M_channel':float(z.mean()),'M_layer':sum(layer_means)/33,'within_quantum_fraction':sum(near)/len(near),'per_layer':per_layer})
 assert torch.equal(targets['ids'],base['fit']['ids']) and torch.equal(targets['labels'],base['fit']['labels'])
 bn=next(x for x in rows if x['state']=='bn-only');new=next(x for x in rows if x['state']=='finite-solve')
 ratio=new['M_channel']/bn['M_channel'];sufficient=ratio<=.1 and new['within_quantum_fraction']>=.95
 gains={role:new['metrics'][role]['top1']-bn['metrics'][role]['top1'] for role in ['fit','development']}
 result={'passed':True,'scope':'sanity only' if c['sanity'] else 'single setting exposed-development diagnostic, not independent confirmation',
 'rows':rows,'M_ratio':ratio,'substantial_mean_fit':sufficient,'accuracy_gain_pp':gains,
 'description':('substantial_fit' if sufficient else 'insufficient_fit')+'__'+('fit_and_dev_damage_ge5pp' if all(x<=-5 for x in gains.values()) else ('dev_within1pp_of_BN' if abs(gains['development'])<=1 else 'mixed_accuracy')),
 'support':{key:sum(int(x[key].sum()) for x in fits['layers'].values()) for key in ['below_range','above_range','at_lower_mean','at_upper_mean','bracketed','float_adjacent']},
 'max_memory_reserved':r['max_memory_reserved'],'hardware':r['hardware'],'finite_solver_not_exact':True,'cuda_initialized':torch.cuda.is_initialized()}
 assert not result['cuda_initialized']
 (E/f'{which}-acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({**result,'rows':[{k:v for k,v in row.items() if k!='per_layer'} for row in rows]}))
if __name__=='__main__':check(sys.argv[1])
