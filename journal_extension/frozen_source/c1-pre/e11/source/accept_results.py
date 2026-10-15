"""CPU raw-tensor acceptance, independent metric/count-difference recomputation."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def accept(phase):
    frozen=json.loads((E/'protocol-freeze.json').read_text())
    for path,h in frozen['files_sha256'].items():assert sha(path)==h,path
    cpu=json.loads((E/'cache/cpu-check.json').read_text());results={};peaks=[];identities=[];memory=[]
    for state in ['teacher','bn','kl']:
        c=json.loads((E/f'config/{phase}-{state}.json').read_text());o=Path(c['output']);r=json.loads((o/'result.json').read_text())
        assert not (o/'failure.json').exists() and r['status']=='completed'
        assert r['state_unchanged'] and r['teacher_unchanged'] and r['non_bias_unchanged']
        assert r['hardware']['gpu_uuid']==c['allowed_uuid'] and 'slurm' in r['hardware']['cgroup']
        assert r['source_module']==str(Path(cpu['source_module']).resolve()) and r['source_module'] in frozen['files_sha256']
        assert r['source_module_sha256']==sha(r['source_module'])
        assert r['dev_count']==0 and r['fit_count']==c['fit_count']
        p=torch.load(o/'predictions.pt',map_location='cpu',weights_only=False)
        ids=torch.load(o/'input-identities.pt',map_location='cpu',weights_only=False)
        # Parent validator checks the same read-only whitelist and no split crossover.
        sys.path.insert(0,str(E.parent/'e4/source'));from common import prepare
        checked=prepare(c,cpu=True);assert checked['roles']=={'fit':c['fit_count'],'development':0,'diagnostic':8}
        actual_parent=json.loads(Path(c['parent_config']).read_text())
        import importlib.util
        spec=importlib.util.spec_from_file_location('e11_check_e0',c['parent_runner']);e0=importlib.util.module_from_spec(spec);spec.loader.exec_module(e0)
        selected=e0.validate_config(actual_parent)['fit'][:c['fit_count']]
        assert ids['ids'].tolist()==[x['index'] for x in selected]
        assert ids['sample_sha256']==[x['sha256'] for x in selected]
        assert ids['labels'].tolist()==[x['label'] for x in selected]
        for key,v in p.items():
            assert torch.equal(v['ids'],ids['ids']) and torch.equal(v['labels'],ids['labels'])
            assert torch.isfinite(v['logits']).all() and torch.equal(v['predictions'],v['logits'].argmax(1))
            assert int((v['predictions']==v['labels']).sum())==v['correct']
            assert abs(float(torch.nn.functional.cross_entropy(v['logits'].double(),v['labels']))-v['ce'])<1e-12
        assert torch.equal(p['b4']['logits'],p['b4-repeat']['logits']) and r['repeat']['bitwise_equal']
        assert all(x['bitwise_equal'] for x in r['anchors'].values())
        assert all(x['bitwise_equal'] for x in r['trace_anchors'].values())
        n=c['fit_count']
        if state=='teacher':
            old=torch.load(c['teacher_scoring'],map_location='cpu',weights_only=False)['fit'];old8=torch.load(c['teacher_fit'],map_location='cpu',weights_only=False)
            assert torch.equal(p['b8']['logits'],old8['logits'][:n])
        else:
            old=torch.load(c['anchor_predictions'],map_location='cpu',weights_only=False)[{'bn':'bn-only','kl':'full-kl-lr0.001'}[state]]['fit']
            saved=torch.load(o/'actual-bias.pt',map_location='cpu',weights_only=False)
            expected=torch.load(c['parent_deployment'] if state=='bn' else c['kl_bias'],map_location='cpu',weights_only=False)
            biases=expected['runtime_biases'] if state=='bn' else expected['biases']
            assert set(saved['biases'])==set(biases) and all(torch.equal(saved['biases'][name],v) for name,v in biases.items())
            assert r['strict_reload']
        assert torch.equal(p['b4']['logits'],old['logits'][:n])
        diff=(p['b4']['logits']-p['b8']['logits']).double().abs();pred_diff=p['b4']['predictions']!=p['b8']['predictions']
        assert int(pred_diff.sum())==r['batch_effect']['prediction_disagreements']
        assert float(diff.max())==r['batch_effect']['max_abs']
        a=torch.load(o/'trace-b4.pt',map_location='cpu',weights_only=False);b=torch.load(o/'trace-b8.pt',map_location='cpu',weights_only=False)
        rr=torch.load(o/'local-replays.pt',map_location='cpu',weights_only=False);diag=json.loads((o/'layer-diagnostics.json').read_text())
        assert list(a['layers'])==list(b['layers'])==checked['interfaces'] and len(diag)==33
        assert torch.equal(a['ids'],ids['ids'][:8]) and torch.equal(b['ids'],ids['ids'][:8])
        assert a['thresholds']==b['thresholds']
        assert torch.equal(a['logits'],p['b4']['logits'][:8]) and torch.equal(b['logits'],p['b8']['logits'][:8])
        layer_changes=[];local_ok=True;failures={};first_input=None
        for name,av in a['layers'].items():
            bv=b['layers'][name];x,y=av['input'],bv['input'];k,l=av['count'],bv['count'];z=diag[name]
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            changed=k!=l;d=(x-y).double().abs()
            expected={'input_max_abs':float(d.max()),'input_mean_abs':float(d.mean()),'input_changed_elements':int((x!=y).sum()),'count_changed_elements':int(changed.sum()),'count_affected_images':int(changed.flatten(1).any(1).sum()),'count_changed_with_equal_input_elements':int((changed&(x==y)).sum()),'total_count_elements':k.numel(),'nonfinite_input_elements':0}
            for key,v in expected.items():assert v==z[key],(name,key)
            t=rr[name]
            assert torch.equal(t['b4-input-b4'],k)==z['local_b4_reconstruct']
            assert torch.equal(t['b8-input-b8'],l)==z['local_b8_reconstruct']
            assert torch.equal(t['b4-input-b4'],t['b4-input-b8'])==z['common_input_b4_b8_equal']
            assert all(t['mean_identity'])==z['repeated_mean_bitwise']
            bad=[key for key in ['local_b4_reconstruct','local_b8_reconstruct','common_input_b4_b8_equal','repeated_mean_bitwise'] if not z[key]]
            if bad:failures[name]=bad
            local_ok &= not bad
            if first_input is None and (x!=y).any():first_input=name
            if changed.any():layer_changes.append(name)
        assert r['first_count_difference']==next(iter(layer_changes),None)
        results[state]={'prediction_disagreements':int(pred_diff.sum()),'different_ids':ids['ids'][pred_diff].tolist(),
            'correct':{key:p[key]['correct'] for key in p},'ce':{key:p[key]['ce'] for key in p},'max_logit_diff':float(diff.max()),
            'mean_logit_diff':float(diff.mean()),'first_count_difference':r['first_count_difference'],'layers_with_count_differences':len(layer_changes),
            'local_replay_failures':failures,'first_input_difference':first_input,'first_count_difference_replay_consistent':None if not layer_changes else layer_changes[0] not in failures,'all_local_replays_consistent':bool(local_ok),'repeat_bitwise':True,'anchors_bitwise':True}
        identities.append(ids['image_tensor_sha256']);peaks.append(r['max_memory_reserved']);memory.append(r['hardware']['total_memory'])
        del a,b,rr,p
    assert len(set(identities))==1 and len(set(memory))==1
    out={'passed':True,'phase':phase,'frozen_files':len(frozen['files_sha256']),'states':results,'max_memory_reserved':peaks,'total_memory':memory[0],
         'common_image_tensor_sha256':identities[0],'protected_evaluation':False,'cuda_initialized':torch.cuda.is_initialized(),
         'E11-C1':any(v['prediction_disagreements']>0 for v in results.values()) if phase=='main' else None}
    assert not out['cuda_initialized'];(E/f'{phase}-acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['sanity','main']);phase=parser.parse_args().phase
    try:accept(phase)
    except Exception:
        import traceback
        (E/f'{phase}-acceptance-failure.json').write_text(json.dumps({'passed':False,'C1':'not established','traceback':traceback.format_exc()},indent=2)+'\n');raise
