"""E6 bounded real-BN adaptation. All CUDA use is guarded by inherited Slurm checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
import argparse, copy, importlib.util, json, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from calibrated_neuron import calibrated, CalibratedResponse, base_potential, response
from direction_gate import run_gate

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e6_common',HERE.parents[1]/'e4/source/common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
BLOCK='conv5_x.2'
SPIKES=['residual_function.2','relu']
BNS=['residual_function.1','residual_function.4']


class TrainNeuron(nn.Module):
    def __init__(self, original, target, mode, tau):
        super().__init__()
        self.theta=float(original.v_threshold);self.T=original.T;self.tau=tau;self.mode=mode
        self.register_buffer('target',target)
        self.register_buffer('delta',torch.zeros_like(target))
        self.register_buffer('delta0',torch.zeros_like(target))
        self.register_buffer('active',torch.zeros_like(target,dtype=torch.bool))
        self.free=nn.Parameter(torch.zeros_like(target),requires_grad=mode.startswith('free'))
        self.detail={};self.gradient_trace={};self.anchor=None;self.checked_original=False
        assert type(original).__name__=='ParaInfNeuron'
        assert getattr(original,'learned_calibration',None) is None
        assert getattr(original,'runtime_output_bias_correction',None) is None
        rank=torch.arange(1,self.T+1,device=original.bias.device,dtype=original.bias.dtype).unsqueeze(-1)
        assert torch.equal(original.TxT,self.T/rank)
        assert torch.equal(original.bias,.5*self.theta/rank)
        self.reference=copy.deepcopy(original)

    def forward(self,x):
        if self.mode in ('comp','stop','center'):
            compensation='center' if self.mode=='center' else self.mode=='comp'
            y,d,active,err=calibrated(x,self.target,self.T,self.theta,self.tau,compensation,trace=self.gradient_trace)
            self.delta.copy_(d);self.active.copy_(active)
        else:
            if self.mode.startswith('free'):
                d=(self.delta0+self.theta*self.free).clamp(-.5*self.theta,.5*self.theta)
            elif self.mode=='fixed':d=self.delta.detach().clone()
            else:d=torch.zeros_like(self.delta)
            active=torch.zeros_like(self.active)
            y=CalibratedResponse.apply(x,d,active,self.T,self.theta,self.tau,False,False)
            self.delta.copy_(d.detach());self.active.zero_()
            with torch.no_grad():
                err=(y.reshape(self.T,-1,*y.shape[1:]).mean((0,1,3,4))-self.target).abs()
        with torch.no_grad():
            if not self.checked_original:
                self.reference.output_bias_correction=d.detach().reshape(1,1,-1,1,1).repeat(self.T,1,1,1,1)
                assert torch.equal(self.reference(x),y),'Actual original neuron with fitted delta differs'
                self.reference.output_bias_correction.zero_()
                yz=response(base_potential(x,self.T,self.theta),torch.zeros_like(d),self.theta,self.tau).reshape_as(x)
                assert torch.equal(self.reference(x),yz),'Actual original zero neuron differs'
                self.checked_original=True
            base=base_potential(x,self.T,self.theta)
            p=((base+d.detach().reshape(1,1,-1,1,1))-self.theta)/(self.tau*self.theta)
            sig=torch.sigmoid(p);weak=(sig*(1-sig)/(self.tau*self.theta)).mean((0,1,3,4))<1e-6
            self.detail={'mean_normalized_residual':float(err.mean()/self.theta),
                         'interior_channels':int(active.sum()),'weak_channels':int(weak.sum()),
                         'compensated_channels':int((active&~weak).sum()) if self.mode=='comp' else 0,
                         'bound_channels':int((d.abs()>=.5*self.theta).sum()),
                         'delta_abs_max_normalized':float(d.abs().max()/self.theta)}
            if self.anchor is not None and self.anchor.shape==y.shape:
                self.detail['spike_changed_fraction_vs_initial_calibration']=float(((y>0)!=self.anchor).float().mean())
            if self.anchor is None and self.mode=='stop':self.anchor=(y>0).detach().clone()
        return y


@torch.no_grad()
def fit_cache(r):
    """Frozen prefix; preserves T-major batch order; source-ANN targets on original images."""
    block=dict(r.model.named_modules())[BLOCK]
    prefix=[];targets={n:[] for n in SPIKES};images=[];labels=[];ids=[];teacher_logits=[]
    captured={}
    hook=block.register_forward_pre_hook(lambda m, inp:captured.update(prefix=inp[0].detach().cpu()))
    handles=[]
    for name in SPIKES:
        def collect(m,inp,out,key=name): targets[key].append(out.detach().double().sum((0,2,3)).cpu())
        handles.append(dict(r.teacher.named_modules())[BLOCK+'.'+name].register_forward_hook(collect))
    sizes=[]
    try:
        for x,y,i in r.e0.loader(r.rows['fit'],r.parent,4):
            xx=x.cuda();r.e0.logits(r.model,xx,r.parent,True)
            p=captured['prefix'].reshape(2,len(x),*captured['prefix'].shape[1:])
            prefix.append(p);sizes.append(len(x)*p.shape[-1]*p.shape[-2])
            teacher_logits.append(r.e0.logits(r.teacher,xx,r.parent).cpu())
            images.append(x);labels.append(y);ids.append(i)
    finally:
        hook.remove()
        for h in handles:h.remove()
    cache={'prefix':torch.cat(prefix,dim=1).flatten(0,1),
           'teacher_means':{n:(torch.stack(v).sum(0)/sum(sizes)).float() for n,v in targets.items()},
           'teacher_logits':torch.cat(teacher_logits),'images':torch.cat(images),
           'labels':torch.cat(labels),'ids':torch.cat(ids),
           'teacher_state_digest':r.teacher_digest,'parent_sha256':r.e0.sha256(r.c['parent_deployment'])}
    assert torch.equal(cache['ids'],torch.tensor([z['index'] for z in r.rows['fit']]))
    r.e0.atomic_torch(r.output/'fit-cache.pt',cache)
    return cache


def tail_forward(tail,x,params,T=2):
    out=torch.func.functional_call(tail,params,(x,),strict=False)
    return out.reshape(T,-1,1000).mean(0)


def run(c):
    began=time.perf_counter();r=common.prepare(c)
    torch.set_num_threads(4)
    assert c['bound_ratio']==.5
    cache=fit_cache(r);N=c['fit_count'];T=r.parent['timesteps'];assert T==2
    tail=nn.Sequential(copy.deepcopy(dict(r.model.named_modules())[BLOCK]),
                       copy.deepcopy(r.model.avg_pool),nn.Flatten(),copy.deepcopy(r.model.fc)).eval()
    for p in tail.parameters():p.requires_grad_(False)
    for n in BNS:assert not dict(tail[0].named_modules())[n].training
    x=cache['prefix'].cuda();y=cache['labels'].cuda();teach=cache['teacher_logits'].cuda()
    # Cache and split are checked against the *real* full network, before replacements.
    with torch.no_grad():
        small=x.reshape(T,N,*x.shape[1:])[:,:4].flatten(0,1)
        baseline_full=r.e0.logits(r.model,cache['images'][:4].cuda(),r.parent,True)
        baseline_tail=tail_forward(tail,small,{})
        assert torch.equal(baseline_full,baseline_tail),'Prefix/tail FP32 mismatch at evaluation batch=4'
    wrappers=[];base_beta=[];scales=[]
    for bnname,spname in zip(BNS,SPIKES):
        original=dict(tail[0].named_modules())[spname];scale=float(original.v_threshold)
        scales.append(scale);base_beta.append(dict(tail[0].named_modules())[bnname].bias.detach().clone())
        w=TrainNeuron(original,cache['teacher_means'][spname].cuda(),c['arm'],c['tau'])
        parent,_,leaf=spname.rpartition('.')
        target=dict(tail[0].named_modules())[parent] if parent else tail[0]
        setattr(target,leaf,w);wrappers.append(w)
    beta=nn.ParameterList([nn.Parameter(torch.zeros_like(b)) for b in base_beta])
    def params():return {'0.'+n+'.bias':b+s*p for n,b,s,p in zip(BNS,base_beta,scales,beta)}
    assert set(params())<=set(dict(tail.named_parameters()))
    def forward():return tail_forward(tail,x,params())
    def set_mode(mode):
        for w in wrappers:w.mode=mode
    # All calibrated and free controls start from identical two-interface roots.
    with torch.no_grad():
        set_mode('stop');forward()
        for w in wrappers:w.delta0.copy_(w.delta);w.free.zero_()
        set_mode(c['arm'])
    mutable={BLOCK+'.'+n+'.bias' for n in BNS}|{BLOCK+'.'+n+'.output_bias_correction' for n in SPIKES}
    immutable=[k for k in r.model.state_dict() if k not in mutable]
    original_digest=r.e0.state_digest(r.model,immutable)
    records=[];predictions={};trace=[]

    @torch.no_grad()
    def score(ident,zero=False):
        # Solve on all fit observations once; deploy static beta and delta afterwards.
        if not zero:fit_logits=forward().cpu()
        for j,(bnname,spname) in enumerate(zip(BNS,SPIKES)):
            r.modules[BLOCK+'.'+bnname].bias.copy_(base_beta[j] if zero else base_beta[j]+scales[j]*beta[j])
            d=torch.zeros_like(wrappers[j].delta) if zero else wrappers[j].delta
            r.modules[BLOCK+'.'+spname].output_bias_correction=d.reshape(1,1,-1,1,1).expand(T,1,-1,1,1).clone()
        p={role:r.e0.evaluate(r.model,r.rows[role],r.parent,True) for role in ('fit','development') if r.rows[role]}
        saved={k:v.detach().cpu().clone() for k,v in r.model.state_dict().items() if k in mutable}
        path=r.output/('deployment-'+ident+'.pt')
        r.e0.atomic_torch(path,{'mutable_state':saved,'thresholds':r.bn['interface_thresholds'],
                              'parent_sha256':r.e0.sha256(c['parent_deployment'])})
        reload=torch.load(path,map_location='cpu',weights_only=False)
        state=dict(r.bn['state_dict']);state.update(reload['mutable_state'])
        r.restored.load_state_dict(state,strict=True)
        for name in r.names:dict(r.restored.named_modules())[name].v_threshold=reload['thresholds'][name]
        before=r.e0.logits(r.model,r.test_images,r.parent,True).cpu()
        after=r.e0.logits(r.restored,r.test_images,r.parent,True).cpu()
        assert torch.equal(before,after),'Real beta + static delta exact reload failed'
        assert r.e0.state_digest(r.model,immutable)==original_digest
        assert r.e0.state_digest(r.teacher)==r.teacher_digest
        detail={'id':ident,'metrics':{k:v['metrics'] for k,v in p.items()},
                'strict_reload':True,'immutable_state_verified':True,'teacher_verified':True,
                'beta_abs_max_normalized':max(float(z.abs().max()) for z in beta),
                'layers':{n:w.detail for n,w in zip(SPIKES,wrappers)} if not zero else {}}
        if not zero:
            detail['fit_full_population_vs_deploy']={
                'max_logit_delta':float((fit_logits-p['fit']['logits']).abs().max()),
                'top1_disagreements':int((fit_logits.argmax(1)!=p['fit']['predictions']).sum())}
            r.e0.atomic_torch(r.output/('train-logits-'+ident+'.pt'),fit_logits)
        predictions[ident]=p;records.append(detail)
        r.e0.atomic_torch(r.output/'predictions.pt',predictions)
        r.e0.atomic_json(r.output/'metrics-progress.json',records)
        r.progress('scored',candidate=ident,metrics=detail['metrics'])

    score('bn',True)
    saved_mode=c['arm'];set_mode('stop');score('calibration-only');set_mode(saved_mode)
    with torch.no_grad():
        set_mode('fixed')
        got=tail_forward(tail,x.reshape(T,N,*x.shape[1:])[:,:4].flatten(0,1),params())
        want=r.e0.logits(r.model,cache['images'][:4].cuda(),r.parent,True)
        assert torch.equal(got,want),'Wrapped fixed calibration differs from original deployment at batch4'
        set_mode(saved_mode)
    # Direction probe: exact first Adam steps, same initial hard forward, fit only.
    gate=None
    if c['direction_probe']:
        gate=run_gate(forward,beta,wrappers,set_mode,y,c,r)
        set_mode(saved_mode)
    trainable=list(beta)+([w.free for w in wrappers] if c['arm'].startswith('free') else [])
    opt=torch.optim.Adam(trainable,lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    score('step-0')
    initial='bn' if c['arm']=='beta-ce' else 'calibration-only'
    for role in predictions['step-0']:
        assert torch.equal(predictions['step-0'][role]['logits'],predictions[initial][role]['logits']), 'Step-0 anchor mismatch'
    steps=c['steps'] if gate is None or gate['passed'] else 0
    for step in range(1,steps+1):
        t=time.perf_counter();opt.zero_grad(set_to_none=True)
        out=forward()
        loss=F.kl_div(F.log_softmax(out,1),F.softmax(teach,1),reduction='batchmean') if c['arm']=='free-kl' else F.cross_entropy(out,y)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        grad_norm=sum(float(p.grad.detach().square().sum()) for p in trainable)**.5
        opt.step()
        with torch.no_grad():
            for p in beta:p.clamp_(-c['bound_ratio'],c['bound_ratio'])
            if c['arm'].startswith('free'):
                for w in wrappers:w.free.clamp_(min=(-.5*w.theta-w.delta0)/w.theta,max=(.5*w.theta-w.delta0)/w.theta)
        torch.cuda.synchronize()
        trace.append({'step':step,'loss_before_update':float(loss.detach()),'grad_norm':grad_norm,
                      'seconds':time.perf_counter()-t,'layers':{n:w.detail for n,w in zip(SPIKES,wrappers)}})
        r.e0.atomic_json(r.output/'training-trace.json',trace)
        r.progress('trained',step=step,loss=float(loss.detach()),seconds=trace[-1]['seconds'])
        if step in c['checkpoints']:
            score('step-'+str(step))
            if c['arm']=='beta-ce':
                set_mode('stop');score('recalibrated-step-'+str(step));set_mode(c['arm'])
    torch.cuda.synchronize()
    result={'status':'completed','arm':c['arm'],'steps':steps,'planned_steps':c['steps'],
            'gate_passed':gate['passed'] if gate else None,'fit_count':N,
            'seconds':time.perf_counter()-began,'max_memory_reserved':torch.cuda.max_memory_reserved(),
            'max_memory_allocated':torch.cuda.max_memory_allocated(),'hardware':r.hardware,
            'teacher_frozen':True,'static_deployment':True,'strict_reload':True,
            'data_status':'exposed_development_single_fit_sample_not_confirmation','records':len(records)}
    r.e0.atomic_json(r.output/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    try:print(json.dumps(run(c)))
    except Exception:
        out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
