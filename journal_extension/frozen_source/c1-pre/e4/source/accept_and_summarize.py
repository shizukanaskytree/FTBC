"""CPU recomputation of all predeclared deployed predictions and bias identities."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from run_interventions import interventions


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('phase');a=p.parse_args()
    e=Path(__file__).resolve().parents[1];c=json.loads((e/f'config/{a.phase}.json').read_text());o=Path(c['output'])
    raw=json.loads((o/'result.json').read_text()); assert raw['status']=='completed' and not (o/'failure.json').exists()
    records=json.loads((o/'metrics-progress.json').read_text());pred=torch.load(o/'predictions.pt',map_location='cpu',weights_only=True)
    parent=torch.load(c['parent_deployment'],map_location='cpu',weights_only=False)
    d=torch.load(c['calibrated_deployment'],map_location='cpu',weights_only=False)['runtime_biases']
    names=list(parent['runtime_biases']);expected=['bn-only','ftbc-e1'];factors={}
    if a.phase=='objective':
        expected += [f'rate256-e{k}' for k in c['checkpoints']]
        expected += [f'{obj}-lr{lr}-e{k}' for obj in c['objectives'] for lr in c['lr_ratios'] for k in c['checkpoints']]
    else:
        cases=list(interventions(names,c['mode']))
        if c.get('sanity'):cases=cases[:2]
        expected += [k for k,v in cases];factors=dict(cases)
    assert [r['id'] for r in records]==expected
    assert set(pred)==set(expected) and raw['records']==len(expected)
    roles=json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text())
    rows=[]
    for rec in records:
        ident=rec['id']; assert rec['strict_reload'] and rec['non_bias_frozen']
        b=torch.load(o/f'bias-{ident}.pt',map_location='cpu',weights_only=False)
        assert b['parent_sha256']==sha(Path(c['parent_deployment']))
        assert b['thresholds']==parent['interface_thresholds'] and set(b['biases'])==set(names)
        for n,v in b['biases'].items():
            assert torch.isfinite(v).all() and torch.equal(v,v[:1].expand_as(v))
            if ident=='bn-only': assert not torch.count_nonzero(v)
            elif ident=='ftbc-e1': assert torch.equal(v,d[n])
            elif ident in factors: assert torch.equal(v,d[n]*factors[ident][n])
            else: assert v.abs().max() <= .500001*parent['interface_thresholds'][n]
        result={'id':ident}
        for role in ['fit','development']:
            v=pred[ident][role];n=c['fit_count'] if role=='fit' else c['dev_count'];r=roles[role][:n]
            assert torch.equal(v['ids'],torch.tensor([x['index'] for x in r]))
            assert torch.equal(v['labels'],torch.tensor([x['label'] for x in r]))
            assert torch.isfinite(v['logits']).all()
            accuracy=float((v['logits'].argmax(1)==v['labels']).double().mean()*100)
            loss=float(F.cross_entropy(v['logits'].double(),v['labels']))
            assert accuracy==rec['metrics'][role]['accuracy_percent']
            assert abs(loss-rec['metrics'][role]['cross_entropy'])<1e-12
            result[role]={'accuracy':accuracy,'ce':loss}
            correct=v['logits'].argmax(1)==v['labels']
            for anchor in ['bn-only','ftbc-e1']:
                z=pred[anchor][role]['logits'].argmax(1)==v['labels']
                result[role]['vs_'+anchor]={'repair':int((correct & ~z).sum()),'damage':int((z & ~correct).sum())}
        rows.append(result)
    for label in ['bn-only','ftbc-e1']:
        anchor=json.loads((o/f'anchor-{label}.json').read_text())
        if not anchor['cross_hardware']:
            assert anchor['max_logit_delta']==0 and anchor['top1_prediction_disagreements']==0
    freeze=json.loads(Path(c['freeze_manifest']).read_text())
    for path,h in freeze['files_sha256'].items(): assert sha(Path(path))==h,path
    assert raw['non_bias_unchanged'] and raw['teacher_unchanged'] and raw['all_strict_reloads_passed']
    accepted={'passed':True,'conditions':len(expected),'all_candidate_ids_match':True,'all_predictions_recomputed':True,'actual_biases_reloaded':True,'frozen_files':len(freeze['files_sha256']),'rows':rows,'whole_process_seconds':raw['whole_process_seconds'],'interpretation':'exploratory development; no novelty or confirmation claim'}
    (o/'acceptance.json').write_text(json.dumps(accepted,indent=2)+'\n')
    (e/f'{a.phase}-acceptance.json').write_text(json.dumps(accepted,indent=2)+'\n')
    artifacts={str(p.relative_to(o)):sha(p) for p in o.iterdir() if p.is_file() and p.name!='artifacts.json'}
    (o/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(json.dumps({'passed':True,'conditions':len(expected),'rows':rows,'seconds':raw['whole_process_seconds']}))


if __name__=='__main__':main()
