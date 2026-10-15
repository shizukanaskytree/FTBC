"""CPU acceptance and paired summaries, not independent confirmation statistics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
E=Path(__file__).resolve().parents[1]

def check(arm):
 c=json.loads((E/'configs'/f'{arm}.json').read_text());out=Path(c['output'])
 r=json.loads((out/'result.json').read_text());assert r['status']=='completed'
 assert r['steps']==(0 if r['gate_passed'] is False else c['steps']) and r['strict_reload'] and r['teacher_frozen']
 assert r['hardware']['gpu_uuid']==c['allowed_uuid']
 p=torch.load(out/'predictions.pt',map_location='cpu',weights_only=False)
 records=json.loads((out/'metrics-progress.json').read_text())
 parent=json.loads(Path(c['parent_config']).read_text())
 roles=json.loads(Path(parent['dataset_manifest']).read_text())
 bn=torch.load(c['parent_deployment'],map_location='cpu',weights_only=False)
 sha=hashlib.sha256(Path(c['parent_deployment']).read_bytes()).hexdigest()
 for rec in records:
  ident=rec['id'];assert rec['strict_reload'] and rec['immutable_state_verified'] and rec['teacher_verified']
  for role,pred in p[ident].items():
   count=c['fit_count'] if role=='fit' else c['dev_count'];rows=roles[role][:count]
   assert torch.equal(pred['ids'],torch.tensor([z['index'] for z in rows]))
   assert torch.equal(pred['labels'],torch.tensor([z['label'] for z in rows]))
   assert torch.isfinite(pred['logits']).all()
   correct=int((pred['logits'].argmax(1)==pred['labels']).sum())
   ce=float(torch.nn.functional.cross_entropy(pred['logits'].double(),pred['labels']))
   assert correct==rec['metrics'][role]['correct']
   assert abs(ce-rec['metrics'][role]['cross_entropy'])<1e-12
  dep=torch.load(out/f'deployment-{ident}.pt',map_location='cpu',weights_only=False)
  assert dep['parent_sha256']==sha and dep['thresholds']==bn['interface_thresholds']
  assert len(dep['mutable_state'])==4
  for k,v in dep['mutable_state'].items():
   assert torch.isfinite(v).all()
   if k.endswith('output_bias_correction'):
    assert torch.equal(v,v[:1].expand_as(v));th=dep['thresholds'][k.removesuffix('.output_bias_correction')]
    assert float(v.abs().max())<=.5*th+1e-6
   else:
    sp='conv5_x.2.residual_function.2' if '.1.bias' in k else 'conv5_x.2.relu'
    assert float((v-bn['state_dict'][k]).abs().max())<=.5*dep['thresholds'][sp]+1e-6
 return c,r,p,records

def paired(a,b):
 assert torch.equal(a['ids'],b['ids']) and torch.equal(a['labels'],b['labels'])
 la=torch.nn.functional.cross_entropy(a['logits'].double(),a['labels'],reduction='none').numpy()
 lb=torch.nn.functional.cross_entropy(b['logits'].double(),b['labels'],reduction='none').numpy()
 d=la-lb;rng=np.random.default_rng(20261009)
 bootstrap=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
 ca=a['logits'].argmax(1)==a['labels'];cb=b['logits'].argmax(1)==b['labels']
 repair=int((ca&~cb).sum());damage=int((~ca&cb).sum())
 return {'accuracy_gain_pp':100*(repair-damage)/len(d),'repair':repair,'damage':damage,
         'mean_paired_ce_difference':float(d.mean()),'descriptive_bootstrap95':np.quantile(bootstrap,[.025,.975]).tolist(),
         'nominal_mcnemar_p':float(binomtest(repair,repair+damage,.5).pvalue) if repair+damage else 1.,
         'scope':'exposed development; intervals and p are descriptive, not confirmation or independent seed evidence'}

def run(phase):
 arms=['sanity'] if phase=='sanity' else ['comp','stop','free-ce','free-kl','beta-ce']
 checked={a:check(a) for a in arms}
 c,r,p,rec=checked[arms[0]]
 result={'passed':True,'arms':arms,'max_memory_reserved':max(z[1]['max_memory_reserved'] for z in checked.values()),
         'total_memory':r['hardware']['total_memory'],'all_metrics_recomputed':True}
 result['P0_fit']={k:p['calibration-only']['fit']['metrics'][k]-p['bn']['fit']['metrics'][k]
                   for k in ['accuracy_percent','cross_entropy']}
 if phase=='sanity':
  gate=json.loads((Path(c['output'])/'direction-gate.json').read_text())
  result['gate_passed']=gate['passed'];result['main_released']=gate['passed']
  if gate['passed']:
   z=torch.load(Path(c['output'])/'train-logits-step-1.pt',map_location='cpu',weights_only=False)
   ce=float(torch.nn.functional.cross_entropy(z.double(),p['step-1']['fit']['labels']))
   expected=next(z['ce'] for z in gate['replays'] if z['mode']=='comp' and z['scope']=='both' and z['learning_rate']==c['learning_rate'])
   assert ce==expected,'Actual Adam training step differs from G0 replay'
   result['g0_actual_first_training_step_ce_exact']=True
 if phase=='main':
  b=checked['comp'][2]['bn']
  for a in arms:
   for role in ('fit','development'):
    assert torch.equal(checked[a][2]['bn'][role]['logits'],b[role]['logits']),a
    assert torch.equal(checked[a][2]['calibration-only'][role]['logits'],checked['comp'][2]['calibration-only'][role]['logits']),a
  comp=checked['comp'][2]['step-16']
  comparisons={'bn':b,'stop':checked['stop'][2]['step-16'],'free-ce':checked['free-ce'][2]['step-16'],
               'free-kl':checked['free-kl'][2]['step-16'],'beta-ce':checked['beta-ce'][2]['step-16'],
               'beta-ce-recalibrated':checked['beta-ce'][2]['recalibrated-step-16']}
  result['paired_development']={k:paired(comp['development'],v['development']) for k,v in comparisons.items()}
  result['development_minimum_effect_met']=all(z['accuracy_gain_pp']>=.5 and z['mean_paired_ce_difference']<0 for z in result['paired_development'].values())
  result['fit_improved']=comp['fit']['metrics']['cross_entropy']<checked['comp'][2]['step-0']['fit']['metrics']['cross_entropy']
  result['candidate_screen_passed']=result['fit_improved'] and result['development_minimum_effect_met']
 (E/f'{phase}-acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['sanity','main']);run(p.parse_args().phase)
