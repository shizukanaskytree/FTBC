"""E10 surrogate sensitivity of ordinary full-bias KL. Slurm only."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e7/source'))
from run_diagnostic import prepare,teacher_targets,measure,verify_prediction_reference
from run_objective_probe import task_loss
sys.path.insert(0,str(E.parent/'e9/source'))
from directions import construct,bind,clamp
import torch

def run(c):
    start=time.perf_counter();r=prepare(c)
    from surrogate import install
    install(r.modules,r.names,c['surrogate'])
    for p in r.model.parameters():p.requires_grad_(False)
    for p in r.teacher.parameters():p.requires_grad_(False)
    r.model.eval();r.teacher.eval()
    refs={n:torch.load(d['bias'],map_location='cpu',weights_only=False) for n,d in c['references'].items()}
    for d in refs.values():
        assert d['parent_sha256']==r.e0.sha256(c['parent_deployment']) and d['thresholds']==r.bn['interface_thresholds']
        assert list(d['biases'])==r.names
    refnames=list(refs) if c['mode']=='root' and not c['sanity'] else ['bn-only']
    for name in refnames:
        r.score(name,refs[name]['biases'],{'reference_only':True},diagnostics=False)
        d=c['references'][name];verify_prediction_reference(r,name,d['predictions'],d['state'])
    if c['surrogate']=='native':
        with torch.no_grad():
            scoring_teacher={role:r.e0.evaluate(r.teacher,r.rows[role],r.parent,False) for role in ['fit','development']}
        r.e0.atomic_torch(r.output/'scoring-teacher.pt',scoring_teacher)
    batches=[(x.cuda(),y.cuda(),ids) for x,y,ids in r.e0.loader(r.rows['fit'],r.parent,c['train_batch'])]
    ids=torch.cat([b[2] for b in batches]);assert torch.equal(ids,torch.tensor([x['index'] for x in r.rows['fit']]))
    torch.cuda.synchronize();t=time.perf_counter()
    with torch.no_grad():teacher=[r.e0.logits(r.teacher,x,r.parent).detach() for x,_,_ in batches]
    torch.cuda.synchronize();teacher_seconds=time.perf_counter()-t
    r.e0.atomic_torch(r.output/'fit-teacher-logits.pt',{'ids':ids,'logits':torch.cat(teacher).cpu(),'teacher_digest':r.teacher_digest})
    with torch.no_grad():target=teacher_targets(r,batches)
    if not c['sanity']:
        old=torch.load(c['targets'],map_location='cpu',weights_only=False)
        assert torch.equal(old['ids'],ids) and old['teacher_digest']==r.teacher_digest
        assert all(torch.equal(target[n].cpu(),old['means'][n]) for n in r.names)
    r.e0.atomic_torch(r.output/'targets.pt',{'ids':ids,'means':{n:v.cpu() for n,v in target.items()}})
    basis,bounds,meta=construct(refs['finite-solve']['biases'],r.bn['interface_thresholds'],c['mode'],c['permutation_seed'])
    full=c['mode'].startswith('full-');objective='ce' if c['mode']=='full-ce' else 'kl'
    r.e0.atomic_torch(r.output/'basis.pt',{'basis':basis,'bounds':bounds,'metadata':meta,'mode':c['mode']})
    basis={n:v.cuda() for n,v in basis.items()};traces={};measurements={}
    with torch.no_grad():
        for name in refnames:
            r.assign(refs[name]['biases']);measurements[name]=measure(r,batches,target)
    for lr in c['lr_ratios']:
        r.assign(r.bn['runtime_biases'])
        parameters={n:torch.nn.Parameter(torch.zeros(len(basis[n]) if full else 1,device='cuda')) for n in r.names}
        optimizer=torch.optim.Adam(list(parameters.values()),lr=lr,weight_decay=0.)
        trace=[];solver_seconds=0.;step=0;nonzero_seen={n:False for n in r.names};first_grad=None
        for epoch in range(1,c['epochs']+1):
            torch.cuda.synchronize();t=time.perf_counter()
            for j,(x,y,_) in enumerate(batches):
                bind(r.modules,parameters,basis,r.bn['interface_thresholds'],r.parent['timesteps'],full)
                optimizer.zero_grad(set_to_none=True)
                logits=r.e0.logits(r.model,x,r.parent,True);loss=task_loss(logits,y,teacher[j],objective)
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
                gradient_norms={n:float(p.grad.norm()) for n,p in parameters.items()}
                if first_grad is None:first_grad=gradient_norms
                for n,g in gradient_norms.items():nonzero_seen[n]=nonzero_seen[n] or g>0
                grad=float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.))
                optimizer.step();clamp(parameters,bounds,full,c['bound_ratio']);step+=1
                trace.append({'step':step,'epoch':epoch,'batch':j,'loss':float(loss.detach()),'gradient_norm':grad,
                    'saturated_parameters':sum(int((p.detach().abs() >= (c['bound_ratio'] if full else bounds[n])).sum()) for n,p in parameters.items())})
            torch.cuda.synchronize();solver_seconds+=time.perf_counter()-t
            r.progress('training',mode=c['mode'],lr=lr,epoch=epoch,steps=step,solver_seconds=solver_seconds)
        bind(r.modules,parameters,basis,r.bn['interface_thresholds'],r.parent['timesteps'],full)
        ident=f'{c["mode"]}-lr{lr}'
        actual=r.snapshot()
        for n,v in actual.items():assert v.abs().max()<=c['bound_ratio']*r.bn['interface_thresholds'][n]+1e-6
        r.e0.atomic_torch(r.output/f'parameters-{ident}.pt',{'parameters':{n:p.detach().cpu() for n,p in parameters.items()},'full':full,'lr':lr,'steps':step,'objective':objective,'first_gradient_norms':first_grad,'nonzero_gradient_seen':nonzero_seen})
        r.score(ident,actual,{'objective':objective,'lr':lr,'epochs':c['epochs'],'steps':step,'solver_seconds':solver_seconds,
            'trainable_parameters':sum(p.numel() for p in parameters.values()),'mode':c['mode']},diagnostics=False)
        with torch.no_grad():measurements[ident]=measure(r,batches,target)
        traces[ident]=trace
        r.assign(r.bn['runtime_biases']);del optimizer,parameters
    r.e0.atomic_json(r.output/'traces.json',traces);r.e0.atomic_torch(r.output/'measurements.pt',measurements)
    result=r.finish();result.update(surrogate=c['surrogate'],whole_process_seconds=time.perf_counter()-start,teacher_cache_seconds=teacher_seconds)
    r.e0.atomic_json(r.output/'result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=json.loads(Path(a.config).read_text())
    try:print(json.dumps(run(c)))
    except Exception:
        Path(c['output']).mkdir(parents=True,exist_ok=True);(Path(c['output'])/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
