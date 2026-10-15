"""CPU-only independent acceptance and descriptive objective analysis."""
import argparse
import hashlib
import json
import csv
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
MEAN_KEYS=['target_mean','actual_mean','zero_current_mean','teacher_input_mean','zero_teacher_mean']
KEYS=['deployed','zero_current','teacher_input','zero_teacher','local_response_abs','local_changed_fraction']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def statistics(layers,key):
 profiles={};signed=[];floors=[];batch=[];sdall=[]
 for n,layer in layers.items():
  x=layer[key].double();R=x.sum(0)/x.shape[0]
  sd=((x-R).square().sum(0)/(len(x)-1)).sqrt() if len(x)>1 else torch.zeros_like(R)
  se=sd/len(x)**.5
  profiles[n]={'M':R.abs().mean().item(),'F':se.mean().item(),'mean_abs_batch':x.abs().mean().item(),'mean_batch_sd':sd.mean().item(),'channels':len(R)}
  signed.append(R);floors.append(se);batch.append(x);sdall.append(sd)
 return {'M':torch.cat(signed).abs().mean().item(),'M_layer_equal':sum(p['M'] for p in profiles.values())/len(profiles),
         'F':torch.cat(floors).mean().item(),'mean_abs_batch':torch.cat(batch,1).abs().mean().item(),'mean_batch_sd':torch.cat(sdall).mean().item(),'layers':profiles}

def accept(phase):
 names=[('sanity-' if phase=='sanity' else '')+g for g in 'SGL']; anchors=[];results={};count=0;peaks=[];runmeta=[]
 freeze=json.loads((E/'protocol-freeze.json').read_text())
 for p,h in freeze['files_sha256'].items():assert sha(p)==h,p
 for name in names:
  c=json.loads((E/'configs'/f'{name}.json').read_text());out=Path(c['output']);r=json.loads((out/'result.json').read_text())
  assert r['status']=='completed' and not (out/'failure.json').exists()
  assert all(r[k] for k in ['non_bias_frozen','teacher_frozen','all_strict_reloads'])
  assert r['hardware']['gpu_uuid']==c['allowed_uuid'] and 'slurm' in r['hardware']['cgroup'].lower()
  data=torch.load(out/'measurements.pt',map_location='cpu',weights_only=False)
  assert list(data)==list(c['states'])==r['states'];assert r['batches']==c['fit_count']//8
  manifest=json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text())
  fit=manifest['fit'][:c['fit_count']];expected=torch.tensor([v['index'] for v in fit]);labels=torch.tensor([v['label'] for v in fit])
  parent=torch.load(c['parent_deployment'],map_location='cpu',weights_only=False)
  for ident,d in data.items():
   assert d['source']==c['states'][ident] and d['source_sha256']==sha(d['source'])
   b=torch.load(d['source'],map_location='cpu',weights_only=False);actual=torch.load(d['reload']['saved'],map_location='cpu',weights_only=False)
   assert d['reload']['strict_reload'] and d['reload']['fit_image_count']==4 and d['reload']['saved_sha256']==sha(d['reload']['saved'])
   assert actual['parent_sha256']==sha(c['parent_deployment']) and actual['thresholds']==parent['interface_thresholds']
   assert set(actual['biases'])==set(parent['runtime_biases'])==set(d['layers'])
   assert all(torch.equal(b['biases'][n],actual['biases'][n]) for n in b['biases'])
   assert torch.equal(d['ids'],expected) and torch.equal(d['labels'],labels)
   assert d['logits'].shape==(len(fit),1000) and torch.isfinite(d['logits']).all()
   assert float((d['logits'].argmax(1)==labels).double().mean()*100)==d['top1']
   assert abs(float(torch.nn.functional.cross_entropy(d['logits'].double(),labels))-d['ce'])<1e-12
   assert d['local_replays_verified']==33*r['batches']
   for n,layer in d['layers'].items():
    assert set(layer)==set(KEYS+MEAN_KEYS)
    for k,v in layer.items():assert v.shape==(r['batches'],actual['biases'][n].shape[2]) and torch.isfinite(v).all()
    for key,mean_key in zip(KEYS[:4],MEAN_KEYS[1:]):
     reconstructed=(layer['target_mean']-layer[mean_key])/parent['interface_thresholds'][n]
     assert torch.allclose(reconstructed,layer[key],atol=2e-7,rtol=2e-7), (ident,n,key)
    assert torch.equal(layer['target_mean'],data['bn-only']['layers'][n]['target_mean'])
    assert torch.equal(layer['zero_teacher'],data['bn-only']['layers'][n]['teacher_input'])
    if phase=='main' and ident in ['bn-only',f'{c["mode"]}-e8']:
     sc=json.loads((E/'configs'/f'sanity-{c["mode"]}.json').read_text())
     sanity=torch.load(Path(sc['output'])/'measurements.pt',map_location='cpu',weights_only=False)
     assert all(torch.equal(v[:1],sanity[ident]['layers'][n][k]) for k,v in layer.items())
    if ident=='bn-only':
     assert torch.equal(layer['deployed'],layer['zero_current']) and torch.equal(layer['teacher_input'],layer['zero_teacher'])
     assert layer['local_response_abs'].sum()==layer['local_changed_fraction'].sum()==0
    assert torch.all((layer['local_changed_fraction']>=0)&(layer['local_changed_fraction']<=1))
    assert torch.all(layer['local_response_abs']>=0)
   summary={'fit_top1':d['top1'],'fit_ce':d['ce'],'source':d['source']}
   for k in KEYS[:4]:summary[k]=statistics(d['layers'],k)
   for k in KEYS[4:]:summary[k]=torch.cat([v[k] for v in d['layers'].values()],1).double().mean().item()
   if ident in results:assert results[ident]==summary
   results[ident]=summary;count+=1
  anchors.append(data['bn-only']);peaks.append(r['max_memory_reserved']);runmeta.append(r)
 for a in anchors[1:]:
  assert torch.equal(a['logits'],anchors[0]['logits'])
  for n,layer in a['layers'].items():
   for k,v in layer.items():assert torch.equal(v,anchors[0]['layers'][n][k])
 receipt={'passed':True,'records':count,'unique_states':len(results),'sum_process_peak_reserved_bytes':sum(peaks),'source_files_verified':len(freeze['files_sha256']),
          'fit_count':len(fit),'same_card_bn_bitwise':True,'local_replay_and_strict_reload_witnesses_passed':True,'protected_evaluation':False,'runs':runmeta}
 assert sum(peaks)<48*2**30
 (E/f'{phase}-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
 (E/f'{phase}-analysis.json').write_text(json.dumps(results,indent=2)+'\n')
 if phase=='main':
  rule={}
  for group,key in [('S','deployed'),('G','deployed'),('L','deployed'),('L-own','teacher_input')]:
   mode=group[0];base=results['bn-only'][key];seq=[base['M']]+[results[f'{mode}-e{i}'][key]['M'] for i in [1,4,8]]
   ratio=seq[-1]/seq[0] if seq[0]>1e-8 else None
   progress=ratio is not None and ratio<=.5 and all(b<=a for a,b in zip(seq,seq[1:])) and base['M']>2*base['F']
   verdict='clear_descriptive_progress' if progress else 'large_remaining_residual' if ratio is not None and ratio>=.8 else 'mixed_or_inconclusive'
   layer_ratios={n:(results[f'{mode}-e8'][key]['layers'][n]['M']/p['M'] if p['M']>1e-8 else None) for n,p in base['layers'].items()}
   rule[group]={'reading':verdict,'M_sequence':seq,'final_to_bn_ratio':ratio,'baseline_F':base['F'],'layer_ratios':layer_ratios}
  (E/'reading-rule.json').write_text(json.dumps(rule,indent=2)+'\n')
  with (E/'measurement-table.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['state','fit_top1','M_deployed','F_deployed','M_teacher_input','M_zero_current','local_response_abs','local_changed_fraction'])
   for n,d in results.items():w.writerow([n,d['fit_top1'],d['deployed']['M'],d['deployed']['F'],d['teacher_input']['M'],d['zero_current']['M'],d['local_response_abs'],d['local_changed_fraction']])
 print(json.dumps({k:v for k,v in receipt.items() if k!='runs'}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['sanity','main']);a=p.parse_args();accept(a.phase)
