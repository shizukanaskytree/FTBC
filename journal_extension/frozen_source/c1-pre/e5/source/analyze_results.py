"""Preregistered E5 development analysis; no protected data and no GPU calls."""
import csv,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
E=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
rows=[];preds={};temperatures={};raw={};group_metrics={}
for mode in ['S','G','L','vector']:
    assert json.loads((E/f'{mode}-acceptance.json').read_text())['passed']
    c=json.loads((E/f'config/{mode}.json').read_text());o=Path(c['output'])
    p=torch.load(o/'predictions.pt',map_location='cpu',weights_only=True);preds[mode]=p
    t=torch.load(o/'fit-teacher-logits.pt',map_location='cpu',weights_only=True)
    metrics=json.loads((o/'metrics-progress.json').read_text());raw[mode]=json.loads((o/'result.json').read_text())
    for rec in metrics:
        ident=rec['id'];q=p[ident]
        row={'mode':mode,'condition':ident}
        for role in ['fit','development']:
            row[role+'_top1']=rec['metrics'][role]['accuracy_percent'];row[role+'_ce']=rec['metrics'][role]['cross_entropy']
        row.update(dev_repair=rec['paired_vs_bn']['development']['repair'],dev_damage=rec['paired_vs_bn']['development']['damage'])
        diag=rec['local_fit_diagnostics'];row['sampled_local_mse']=float(np.mean([x['normalized_mse'] for x in diag.values()]));row['sampled_local_mean_abs_residual']=float(np.mean([x['mean_abs_channel_residual'] for x in diag.values()]))
        b=torch.load(o/f'bias-{ident}.pt',map_location='cpu',weights_only=False)
        norm=torch.cat([x[0].flatten().double()/b['thresholds'][n] for n,x in b['biases'].items()])
        row.update(bias_relative_rms=float(norm.square().mean().sqrt()),bound_channels=int((norm.abs()>=.5-1e-7).sum()))
        rows.append(row)
        assert torch.equal(q['fit']['ids'],t['ids'])
        z=q['fit']['logits'].double();y=q['fit']['labels'];teacher=F.softmax(t['logits'].double(),1)
        for objective in ['label_ce','teacher_kl']:
            def loss(beta):return float(F.cross_entropy(z*beta,y)) if objective=='label_ce' else float(F.kl_div(F.log_softmax(z*beta,1),teacher,reduction='batchmean'))
            fit=minimize_scalar(loss,bounds=(.01,100.),method='bounded',options={'xatol':1e-11});assert fit.success
            beta=min([.01,100.,1.,float(fit.x)],key=loss)
            dev=q['development'];v=dev['logits'].double()*beta
            assert torch.equal(v.argmax(1),dev['logits'].argmax(1))
            temperatures[f'{mode}:{ident}:{objective}']={'temperature':1/beta,'fit_objective':loss(beta),'development_ce':float(F.cross_entropy(v,dev['labels'])),'development_top1':float((v.argmax(1)==dev['labels']).double().mean()*100)}
# Same-card anchors must be exact; cross-hardware equality is recorded, not assumed.
anchors={}
for mode in ['G','L','vector']:
    for ident in ['bn-only','ftbc-e1','kl-e4','e1-scale-0.125']:
        a=preds['S'][ident]['development'];b=preds[mode][ident]['development']
        assert torch.equal(a['ids'],b['ids']) and torch.equal(a['labels'],b['labels'])
        delta=float((a['logits']-b['logits']).abs().max());anchors[f'{mode}:{ident}']={'max_logit_difference':delta,'prediction_disagreements':int((a['logits'].argmax(1)!=b['logits'].argmax(1)).sum())}
        if mode!='vector':assert delta==0

def compare(am,ai,bm,bi):
    a=preds[am][ai]['development'];b=preds[bm][bi]['development'];assert torch.equal(a['ids'],b['ids']) and torch.equal(a['labels'],b['labels'])
    y=a['labels'];ca=a['logits'].argmax(1)==y;cb=b['logits'].argmax(1)==y;d=(ca.int()-cb.int()).double()
    ce=F.cross_entropy(a['logits'].double(),y,reduction='none')-F.cross_entropy(b['logits'].double(),y,reduction='none')
    def interval(x,scale):
        avg=float(x.mean())*scale;se=float(x.std(unbiased=True))/math.sqrt(len(x))*scale
        return {'mean':avg,'paired_normal_95_interval':[avg-1.96*se,avg+1.96*se]}
    return {'a':am+':'+ai,'b':bm+':'+bi,'top1_pp':interval(d,100),'ce_nats':interval(ce,1),'repair':int((ca&~cb).sum()),'damage':int((~ca&cb).sum()),'n':len(y)}
# First interface has identical fixed upstream inputs in G and S.
for k in [1,4,8]:
    values=[]
    for m in ['S','G']:
        c=json.loads((E/f'config/{m}.json').read_text())
        values.append(torch.load(Path(c['output'])/f'bias-{m}-e{k}.pt',map_location='cpu',weights_only=False)['biases']['conv1.2'])
    assert torch.equal(*values), 'First-layer G/S biases differ'
comparisons=[]
for k in [1,4,8]:
    comparisons += [compare('G',f'G-e{k}','S',f'S-e{k}'),compare('L',f'L-e{k}','G',f'G-e{k}')]
    for mode in ['S','G','L']:
        comparisons += [compare(mode,f'{mode}-e{k}',mode,'bn-only'),compare(mode,f'{mode}-e{k}',mode,'kl-e4')]
for ident in preds['vector']:
    if ident not in ['bn-only','ftbc-e1','kl-e4','e1-scale-0.125']:comparisons.append(compare('vector',ident,'vector','ftbc-e1' if ident.startswith('e1-') else 'kl-e4'))
with (E/'results-table.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result={'scope':'one calibration sample, prospective research-internal development; exploratory, no multiplicity-adjusted confirmation','rows':rows,'paired_comparisons':comparisons,'temperature_fits':temperatures,'hardware_anchors':anchors,'per_process_cost_seconds':{m:r['whole_process_seconds'] for m,r in raw.items()},'resource_note':'Concurrent child seconds are not additive GPU hours. Use Slurm allocation wall windows separately.','new_method_established':False}
(E/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'rows':rows,'primary_e8_comparisons':[x for x in comparisons if x['a'].endswith('e8')],'anchors':anchors},indent=2))
