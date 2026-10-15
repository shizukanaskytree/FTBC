#!/usr/bin/env python3
"""CPU-only acceptance from saved real predictions, serialized buffers and manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    torch.set_num_threads(4)
    config=json.loads(Path(args.config).read_text());out=Path(config['output'])
    result=json.loads((out/'result.json').read_text());checks=json.loads((out/'deployment-checks.json').read_text())
    freeze=json.loads(Path(config['freeze_manifest']).read_text())
    for path,digest in freeze['files_sha256'].items():assert sha(path)==digest,path
    for name,item in json.loads((out/'artifacts.json').read_text()).items():
        assert sha(out/name)==item['sha256'],name
        assert (out/name).stat().st_size==item['bytes']
    assert result['status']=='completed' and result['all_deployment_checks_passed']
    assert result['hardware']['gpu_uuid']==config['allowed_uuid']
    assert result['config_sha256']==sha(args.config)
    assert result['freeze_sha256']==sha(config['freeze_manifest'])
    assert result['input_counts']=={'fit':config['fit_count'],'development':config['dev_count']}
    expected={'bn-only'}
    expected|={f'ce-{ratio:g}-s{s}' for ratio in config['ce_lr_ratios'] for s in config['ce_checkpoints']}
    expected|={f'{f}-p{s}' for f in ['fullscan']+['grid'+str(g) for g in config['grid_points']] for s in range(1,config['sweeps']+1)}
    assert set(r['id'] for r in result['records'])==expected==set(checks)
    data=torch.load(out/'predictions.pt',map_location='cpu',weights_only=False)
    inputs=json.loads(Path(config['input_manifest']).read_text())
    for role in ('fit','development'):
        assert data[role+'_ids'].tolist()==[r['index'] for r in inputs[role]]
        assert data[role+'_labels'].tolist()==[r['label'] for r in inputs[role]]
    baseline=data['candidates']['bn-only']['actual_development']['predictions']
    labels=data['development_labels'];base_ok=baseline==labels
    table=[]
    for r in result['records']:
        name=r['id'];prediction=data['candidates'][name];c=checks[name]
        assert c['exact_reload'] and c['non_bias_state_unchanged'] and c['top1_equal']
        assert c['actual_integer_counts_equal_fit_and_dev'] and c['strict_model_state_reload']
        saved=torch.load(out/f'bias-{name}.pt',map_location='cpu',weights_only=True)
        assert sha(out/f'bias-{name}.pt')==c['bias_sha256']
        assert sha(saved['parent_deployment'])==saved['parent_sha256']
        buffer=saved['output_bias_correction']
        wanted=torch.zeros_like(buffer)
        wanted[:,:,config['channels'],0,0]=torch.tensor(r['delta'],dtype=buffer.dtype).view(1,1,-1)
        assert torch.equal(buffer,wanted)
        assert float(buffer.abs().max())<=float(torch.tensor(.5*result['theta'],dtype=buffer.dtype))
        for role in ('fit','development'):
            actual=prediction['actual_'+role];y=data[role+'_labels'];z=actual['logits']
            assert torch.equal(actual['ids'],data[role+'_ids']) and torch.equal(actual['labels'],y)
            assert torch.equal(actual['predictions'],z.argmax(1))
            accuracy=float((z.argmax(1)==y).double().mean()*100)
            ce=float(F.cross_entropy(z.double(),y))
            assert abs(accuracy-r['actual_'+role]['top1'])<1e-12
            assert abs(ce-r['actual_'+role]['ce'])<1e-12
            assert torch.equal(z.argmax(1),prediction[role+'_logits'].argmax(1))
        ok=prediction['actual_development']['predictions']==labels
        table.append({'id':name,'family':r['family'],'fit_ce':r['actual_fit']['ce'],
                      'dev_ce':r['actual_development']['ce'],'dev_top1':r['actual_development']['top1'],
                      'repair':int((ok&~base_ok).sum()),'damage':int((~ok&base_ok).sum()),
                      'spikes_fit':r['selected_interface_spikes_fit'],'solver_seconds':r['solver_seconds'],
                      'calibration_cost_seconds':r['calibration_cost_seconds'],
                      'capture_solver_replay_seconds':r['with_capture_and_replay_seconds']})
    key=lambda r:(-r['dev_top1'],r['dev_ce'],r['solver_seconds'],r['id'])
    controls=[r for r in table if r['family']=='ce' or r['family'].startswith('grid')]
    events=[r for r in table if r['family']=='fullscan'];best_control=min(controls,key=key)
    event_best=min(events,key=key);base=next(r for r in table if r['id']=='bn-only')
    budget_checks=[]
    for r in events:
        comparable=[x for x in controls if x['solver_seconds']<=r['solver_seconds']]
        overlap=min(x['solver_seconds'] for x in controls)<=r['solver_seconds']<=max(x['solver_seconds'] for x in controls)
        bc=min(comparable,key=key) if comparable else None
        budget_checks.append({'event':r['id'],'overlapping_measured_budget':overlap,'control':None if bc is None else bc['id'],
          'quality_screen':bool(overlap and bc and r['dev_top1']>=bc['dev_top1']+.5-1e-12 and r['fit_ce']<bc['fit_ce'] and r['dev_top1']>=base['dev_top1'])})
    cost_checks=[]
    for r in events:
        equivalent=[x for x in controls if x['dev_top1']>=r['dev_top1']]
        cheapest=min(equivalent,key=lambda x:x['calibration_cost_seconds']) if equivalent else None
        passed=bool(cheapest and r['dev_top1']>=best_control['dev_top1'] and r['dev_top1']>=base['dev_top1'] and
                    r['calibration_cost_seconds']<=.8*cheapest['calibration_cost_seconds'])
        cost_checks.append({'event':r['id'],'cheapest_equal_or_better_accuracy_control':None if cheapest is None else cheapest['id'],'passed':passed})
    cost_screen=any(c['passed'] for c in cost_checks)
    b=result['bound']
    bound_screen=abs(b['loss_sum_delta_vs_fullscan'])<=1e-7 and b['loss_query_ratio']<1 and b['total_time_ratio']<=.8
    acceptance={'passed':True,'candidate_count':len(table),'config_sha256':sha(args.config),
      'frozen_files_verified':len(freeze['files_sha256']),'actual_predictions_recomputed':True,
      'serialized_real_bias_verified':True,'gpu_execution_reload_witness':str(out/'deployment-checks.json'),
      'scope':'CPU independently recomputed metrics/identities/buffers; exact forward reload witnessed in GPU run',
      'protected_evaluation_opened':False,'cuda_initialized':torch.cuda.is_initialized()}
    analysis={'scope':result['scope'],'table':table,'best_control':best_control,'event_best':event_best,
      'baseline':base,'budget_checks':budget_checks,'cost_screen':cost_screen,'cost_checks':cost_checks,
      'candidate_screen_passed':cost_screen or any(x['quality_screen'] for x in budget_checks),
      'bound_screen_passed':bound_screen,'bound':b,'micro':result['micro'],
      'process_gpu_hours':result['elapsed_seconds_including_setup_preprocessing']/3600,
      'capture_seconds':result['capture_seconds'],
      'limitations':['single sample and setting, eight terminal coordinates','development reused for selection; no confirmation',
                    'inherited BN fit on 4096 images, not a 256-image end-to-end method',
                    'per-candidate cost excludes other hyperparameter trials; entire search cost reported separately',
                    'all costs are calibration/validation runtime; no inference latency claim']}
    (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    (out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps({'acceptance':acceptance,'best_control':best_control,'event_best':event_best,
                      'screen':analysis['candidate_screen_passed'],'bound_screen':bound_screen}))


if __name__=='__main__':main()
