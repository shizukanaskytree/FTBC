"""Recompute development metrics and fixed screening decisions on CPU."""
import json,sys,hashlib
from pathlib import Path
import torch
torch.set_num_threads(4)
from directions import construct
E=Path(__file__).resolve().parents[1]
MODES=['root','permuted','constant','full-kl','full-ce']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def check(phase):
    for p,h in json.loads((E/'protocol-freeze.json').read_text())['files_sha256'].items():assert sha(p)==h,p
    states={};memory=[];hardware=[];cross={};teacher_ref=None;preds={};solver={}
    for mode in MODES:
        c=json.loads((E/'config'/f'{phase}-{mode}.json').read_text());out=Path(c['output'])
        result=json.loads((out/'result.json').read_text());assert result['status']=='completed'
        assert result['teacher_unchanged'] and result['non_bias_unchanged'] and result['all_strict_reloads_passed']
        memory.append(result['max_memory_reserved']);hardware.append(result['hardware'])
        rows=json.loads((out/'metrics-progress.json').read_text());pred=torch.load(out/'predictions.pt',map_location='cpu',weights_only=False)
        expected=set(['bn-only']+([n for n in c['references'] if n!='bn-only'] if mode=='root' and phase=='main' else [])+[f'{mode}-lr{lr}' for lr in c['lr_ratios']])
        assert set(pred)==expected and result['records']==len(expected)==len(rows)
        targets=torch.load(out/'targets.pt',map_location='cpu',weights_only=False)
        teacher=torch.load(out/'fit-teacher-logits.pt',map_location='cpu',weights_only=False)
        if teacher_ref is None:teacher_ref=teacher
        else:assert teacher['teacher_digest']==teacher_ref['teacher_digest'] and torch.equal(teacher['logits'],teacher_ref['logits']) and torch.equal(teacher['ids'],teacher_ref['ids'])
        rolefile=json.loads(Path(c['parent_config']).read_text())['dataset_manifest'];roles=json.loads(Path(rolefile).read_text());roles=roles.get('roles',roles)
        refroot=torch.load(c['references']['finite-solve']['bias'],map_location='cpu',weights_only=False)
        basis,limits,meta=construct(refroot['biases'],refroot['thresholds'],mode,c['permutation_seed'])
        actualbasis=torch.load(out/'basis.pt',map_location='cpu',weights_only=False);assert actualbasis['bounds']==limits and all(torch.equal(actualbasis['basis'][n],v) for n,v in basis.items())
        measures=torch.load(out/'measurements.pt',map_location='cpu',weights_only=False)
        traces=json.loads((out/'traces.json').read_text());assert set(traces)=={f'{mode}-lr{lr}' for lr in c['lr_ratios']}
        for row in rows:
            name=row['id'];bias=torch.load(out/f'bias-{name}.pt',map_location='cpu',weights_only=False)
            assert bias['parent_sha256']==sha(c['parent_deployment']) and bias['thresholds']==refroot['thresholds']
            if name in c['references']:
                ref=c['references'][name];wanted=torch.load(ref['bias'],map_location='cpu',weights_only=False)['biases']
            else:
                params=torch.load(out/f'parameters-{name}.pt',map_location='cpu',weights_only=False)
                full=mode.startswith('full-');assert params['full']==full and params['steps']==c['epochs']*c['fit_count']//c['train_batch']
                assert params['objective']==('ce' if mode=='full-ce' else 'kl')==row['detail']['objective']
                assert params['lr']==row['detail']['lr'] and name==f'{mode}-lr{params["lr"]}'
                assert set(params['nonzero_gradient_seen'])==set(basis) and set(params['first_gradient_norms'])==set(basis)
                assert all(torch.isfinite(torch.tensor(v)) for v in params['first_gradient_norms'].values())
                assert row['detail']['trainable_parameters']==(7616 if full else 33)
                trace=traces[name];assert len(trace)==params['steps'] and [v['step'] for v in trace]==list(range(1,params['steps']+1))
                assert all(torch.isfinite(torch.tensor([v['loss'],v['gradient_norm']])).all() for v in trace)
                assert any(v['gradient_norm']>0 for v in trace)
                wanted={}
                for n,p in params['parameters'].items():
                    assert p.numel()==(len(basis[n]) if full else 1) and p.abs().max()<=(.5 if full else limits[n])+1e-7
                    value=p if full else p*basis[n]
                    wanted[n]=(value*bias['thresholds'][n]).view(1,1,-1,1,1).expand_as(bias['biases'][n]).contiguous()
                solver[name]=row['detail']['solver_seconds']
                if phase=='main' and name=='full-kl-lr0.001':
                    old=torch.load(c['references']['kl-e4']['bias'],map_location='cpu',weights_only=False)['biases']
                    assert all(torch.equal(v,old[n]) for n,v in wanted.items()),'ordinary KL reproduction changed'
            assert set(wanted)==set(bias['biases'])
            assert all(torch.equal(v,bias['biases'][n]) and v.abs().max()<=.5*bias['thresholds'][n]+1e-6 for n,v in wanted.items())
            stats={}
            for role,count in [('fit',c['fit_count']),('development',c['dev_count'])]:
                q=pred[name][role];ids=torch.tensor([v['index'] for v in roles[role][:count]]);labels=torch.tensor([v['label'] for v in roles[role][:count]])
                assert torch.equal(q['ids'],ids) and torch.equal(q['labels'],labels) and torch.isfinite(q['logits']).all()
                y=q['logits'].argmax(1);assert torch.equal(y,q['predictions']);good=y==labels
                correct=int(good.sum());ce=float(torch.nn.functional.cross_entropy(q['logits'].double(),labels))
                assert correct==row['metrics'][role]['correct'] and abs(ce-row['metrics'][role]['cross_entropy'])<1e-12
                if name in c['references']:
                    old=torch.load(ref['predictions'],map_location='cpu',weights_only=False)[ref['state']][role];assert torch.equal(q['logits'],old['logits'][:count])
                if name=='bn-only':
                    if role in cross:assert torch.equal(q['logits'],cross[role])
                    else:cross[role]=q['logits']
                bn=pred['bn-only'][role]['predictions']==labels
                stats[role]={'n':count,'correct':correct,'top1':100*correct/count,'ce':ce,'repair_vs_bn':int((good&~bn).sum()),'damage_vs_bn':int((~good&bn).sum())}
                preds.setdefault(name,{})[role]=q
            assert torch.equal(targets['ids'],pred[name]['fit']['ids'])
            errs=[]
            for n,m in measures[name].items():
                assert torch.equal(m['target_mean'],targets['means'][n])
                error=m['count'].double()/m['observations_per_channel']-m['target_mean']/m['theta']
                assert torch.allclose(error,m['normalized_residual'],atol=1e-15,rtol=0);errs.append(error.abs())
            assert sum(x.numel() for x in errs)==7616
            profile={}
            if name not in c['references'] and not mode.startswith('full-'):
                for n,p in params['parameters'].items():
                    sr=meta[n]['root_rms'];a=float(p)
                    profile[n]={'coefficient':a,'signed_relative_R':a/sr if sr else None,'absolute_relative_R':abs(a)/sr if sr else None,'at_bound':abs(a)>=limits[n]-1e-7,**meta[n]}
            states[name]={'metrics':stats,'M':float(torch.cat(errs).mean()),'detail':row['detail'],'coefficient_profile':profile,
                'observed_nonzero_gradient_interfaces':None if name in c['references'] else sum(params['nonzero_gradient_seen'].values())}
    for h in hardware:assert all(h[k]==hardware[0][k] for k in ['slurm_job_id','gpu_uuid','name','total_memory','torch','torchvision','cuda_runtime','cudnn'])
    tc=json.loads((E/'config'/f'{phase}-root.json').read_text())
    scoreteacher=torch.load(Path(tc['output'])/'scoring-teacher.pt',map_location='cpu',weights_only=False)
    for name,s in states.items():
        for role in ['fit','development']:
            q=preds[name][role];t=scoreteacher[role]
            assert torch.equal(t['ids'],q['ids']) and torch.equal(t['labels'],q['labels']) and torch.isfinite(t['logits']).all()
            s['metrics'][role]['kl']=float(torch.nn.functional.kl_div(torch.log_softmax(q['logits'].double(),1),torch.softmax(t['logits'].double(),1),reduction='batchmean'))
            b=preds['bn-only'][role]
            s['metrics'][role]['prediction_changes_vs_bn']=int((q['predictions']!=b['predictions']).sum())
            s['metrics'][role]['kl_BN_to_candidate']=float(torch.nn.functional.kl_div(torch.log_softmax(q['logits'].double(),1),torch.softmax(b['logits'].double(),1),reduction='batchmean'))
    if phase=='main':
        for role in ['fit','development']:assert torch.equal(preds['full-kl-lr0.001'][role]['logits'],preds['kl-e4'][role]['logits']),'KL logits reproduction changed'
    pairwise={}
    for n,q in preds.items():
        pairwise[n]={}
        good=q['development']['predictions']==q['development']['labels']
        for other,o in preds.items():
            refgood=o['development']['predictions']==o['development']['labels']
            pairwise[n][other]={'repair':int((good&~refgood).sum()),'damage':int((~good&refgood).sum()),'correctness_disagreements':int((good!=refgood).sum())}
    output={'passed':True,'phase':phase,'states':states,'pairwise_development':pairwise,'solver_seconds':solver,'hardware':hardware[0],'max_memory_reserved':memory,'total_memory':hardware[0]['total_memory'],'strict_reloads':True,'actual_bias_reconstruction':True,'cross_worker_BN_teacher_equal':True,'cuda_initialized':torch.cuda.is_initialized()}
    assert not output['cuda_initialized']
    (E/f'{phase}-acceptance.json').write_text(json.dumps(output,indent=2)+'\n')
    # Persist paired raw predictions only by reference; scientific interpretation is separate.
    print(json.dumps({'passed':True,'phase':phase,'states':{n:s['metrics']['development']['correct'] for n,s in states.items()},'max_memory_GiB':[x/2**30 for x in memory]}))
if __name__=='__main__':check(sys.argv[1])
