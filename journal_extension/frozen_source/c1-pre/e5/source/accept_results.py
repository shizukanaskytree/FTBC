"""Independent CPU metrics and identity audit; forward reload belongs to GPU runner."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
E=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def accept(name):
    c=json.loads((E/f'config/{name}.json').read_text());o=Path(c['output'])
    result=json.loads((o/'result.json').read_text());records=json.loads((o/'metrics-progress.json').read_text())
    assert result['status']=='completed' and not (o/'failure.json').exists()
    assert result['teacher_unchanged'] and result['non_bias_unchanged'] and result['all_strict_reloads_passed']
    assert result['hardware']['gpu_uuid']==c['allowed_uuid'] and 'slurm' in result['hardware']['cgroup'].lower()
    expected=['bn-only','ftbc-e1','kl-e4','e1-scale-0.125']
    if c['mode']=='vector': expected += ['e1-shuffle-1.0','e1-shuffle-0.5','e1-norm-kl','kl-norm-e1','e1-direct_bn','e1-post_add','kl-e4-direct_bn','kl-e4-post_add']
    else:
        expected += [f'{c["mode"]}-e{k}' for k in c['checkpoints']]
        if not c['sanity']: expected += [f'{c["mode"]}-e8-{g}' for g in ['direct_bn','post_add']]
    assert [r['id'] for r in records]==expected and result['records']==len(expected)
    pred=torch.load(o/'predictions.pt',map_location='cpu',weights_only=True);assert set(pred)==set(expected)
    parent=torch.load(c['parent_deployment'],map_location='cpu',weights_only=False)
    e1=torch.load(c['calibrated_deployment'],map_location='cpu',weights_only=False)['runtime_biases']
    kl=torch.load(Path(c['e4_objective_output'])/'bias-kl-lr0.001-e4.pt',map_location='cpu',weights_only=False)['biases']
    manifest=json.loads(Path(json.loads(Path(c['parent_config']).read_text())['dataset_manifest']).read_text())
    rows=[]
    for rec in records:
        ident=rec['id'];assert rec['strict_reload'] and rec['non_bias_frozen']
        saved=torch.load(o/f'bias-{ident}.pt',map_location='cpu',weights_only=False)
        assert saved['parent_sha256']==sha(c['parent_deployment']) and saved['thresholds']==parent['interface_thresholds']
        assert set(saved['biases'])==set(parent['runtime_biases'])
        for n,b in saved['biases'].items():
            assert torch.isfinite(b).all() and b.shape==parent['runtime_biases'][n].shape and torch.equal(b,b[:1].expand_as(b))
            if ident=='bn-only': assert not b.count_nonzero()
            elif ident=='ftbc-e1': assert torch.equal(b,e1[n])
            elif ident=='kl-e4': assert torch.equal(b,kl[n])
            elif ident=='e1-scale-0.125': assert torch.equal(b,e1[n]*.125)
            if ident!='ftbc-e1': assert b.abs().max()<=.500001*parent['interface_thresholds'][n]
            if ident.endswith('-direct_bn') or ident.endswith('-post_add'):
                group='post_add' if ident.endswith('-post_add') else 'direct_bn'
                src=e1 if ident.startswith('e1-') else kl if ident.startswith('kl-') else torch.load(o/f'bias-{c["mode"]}-e8.pt',map_location='cpu',weights_only=False)['biases']
                kept=n.endswith('.relu')==(group=='post_add')
                assert torch.equal(b,src[n] if kept else torch.zeros_like(b))
        row={'id':ident}
        for role,count in [('fit',c['fit_count']),('development',c['dev_count'])]:
            q=pred[ident][role];ref=manifest[role][:count]
            assert torch.equal(q['ids'],torch.tensor([v['index'] for v in ref]))
            assert torch.equal(q['labels'],torch.tensor([v['label'] for v in ref]))
            assert q['logits'].shape==(count,1000) and torch.isfinite(q['logits']).all()
            labels=q['labels'];a=q['logits'].argmax(1)==labels;base=pred['bn-only'][role]['logits'].argmax(1)==labels
            acc=float(a.double().mean()*100);ce=float(F.cross_entropy(q['logits'].double(),labels))
            assert acc==rec['metrics'][role]['accuracy_percent'] and abs(ce-rec['metrics'][role]['cross_entropy'])<1e-12
            repair=int((a & ~base).sum());damage=int((~a & base).sum())
            assert {'repair':repair,'damage':damage}==rec['paired_vs_bn'][role]
            row[role]={'accuracy':acc,'ce':ce,'repair':repair,'damage':damage}
        rows.append(row)
    if c['mode'] in ['G','L']:
        trace=json.loads((o/'traces.json').read_text())
        assert len(trace)==33*len(c['checkpoints'])
        for k in c['checkpoints']:
            group=[x for x in trace if x['epochs']==k]
            assert [x['layer'] for x in group]==list(parent['runtime_biases'])
            assert all(x['updates']==k*c['fit_count']//c['train_batch'] for x in group)
        if c['sanity'] and c['mode']=='G': assert records[-1]['detail']['cached_vs_uncached_checked']
    elif c['mode']=='S' and c.get('check_e4',False):
        for k in c['checkpoints']:
            a=torch.load(o/f'bias-S-e{k}.pt',map_location='cpu',weights_only=False)['biases']
            b=torch.load(Path(c['e4_objective_output'])/f'bias-rate256-e{k}.pt',map_location='cpu',weights_only=False)['biases']
            assert all(torch.equal(a[n],b[n]) for n in a)
    freeze=json.loads((E/'protocol-freeze.json').read_text())
    for p,h in freeze['files_sha256'].items(): assert sha(p)==h,p
    receipt={'passed':True,'conditions':len(expected),'rows':rows,'whole_process_seconds':result['whole_process_seconds'],'max_memory_reserved':result['max_memory_reserved'],'source_files_verified':len(freeze['files_sha256']),'identity_and_recomputed_metrics_passed':True,'reload_scope':'GPU runner four fitting-image strict reload witness, CPU tensor/record checks only','protected_evaluation':False}
    (E/f'{name}-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'phase':name,'passed':True,'conditions':len(expected),'rows':rows,'seconds':result['whole_process_seconds']}))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['sanity','S','G','L','vector']);a=p.parse_args()
    if a.phase=='sanity':
        got=[accept('sanity-'+n) for n in ['S','G','L']]
        receipt={'passed':all(x['passed'] for x in got),'sum_process_peak_reserved_bytes':sum(x['max_memory_reserved'] for x in got),'conditions':sum(x['conditions'] for x in got)}
        assert receipt['sum_process_peak_reserved_bytes']<48*2**30
        (E/'sanity-acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    else: accept(a.phase)
