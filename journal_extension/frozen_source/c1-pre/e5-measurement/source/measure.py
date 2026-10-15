"""E5-M: forward-only objective/replay measurements of immutable saved states."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / 'e4/source'))
from common import prepare
import torch


def means(y, T):
    return y.reshape(T, -1, *y.shape[1:]).mean(0).mean((0, 2, 3), keepdim=True)


def aggregate(batch_residuals):
    x = batch_residuals.double()
    R = x.mean(0)
    sd = x.std(0, unbiased=True) if len(x)>1 else torch.zeros_like(R)
    return {'signed_global':R, 'batch_sd':sd, 'batch_sem_scale':sd/len(x)**.5,
            'mean_absolute_batch':x.abs().mean(0)}


def capture(model, names):
    holder, handles = {}, []
    def pre(n):
        def fn(_, args): holder.setdefault(n,{})['x'] = args[0].detach().clone()
        return fn
    def post(n):
        def fn(_, args, y): holder[n]['y'] = y.detach().clone()
        return fn
    modules = dict(model.named_modules())
    for n in names:
        handles.extend([modules[n].register_forward_pre_hook(pre(n)), modules[n].register_forward_hook(post(n))])
    return holder, handles


@torch.no_grad()
def residuals(module, x, y, tx, ty, T, theta):
    """Never changes upstream. Hooks have been removed before these replays."""
    original = module.output_bias_correction
    assert torch.equal(module(x), y), 'Full model vs local neuron replay mismatch'
    target = ty.mean((0,2,3), keepdim=True)
    actual = means(y,T)
    teacher_x = tx.repeat(T,1,1,1)
    own = means(module(teacher_x),T)
    try:
        module.output_bias_correction = torch.zeros_like(original)
        zy = module(x)
        zown = means(module(teacher_x),T)
    finally:
        module.output_bias_correction = original
    zero = means(zy,T)
    delta = (y.reshape(T,-1,*y.shape[1:]).mean(0) - zy.reshape(T,-1,*zy.shape[1:]).mean(0))/theta
    return {'target_mean':target.flatten().cpu(), 'actual_mean':actual.flatten().cpu(),
            'zero_current_mean':zero.flatten().cpu(), 'teacher_input_mean':own.flatten().cpu(), 'zero_teacher_mean':zown.flatten().cpu(),
            'deployed':((target-actual)/theta).flatten().cpu(),
            'zero_current':((target-zero)/theta).flatten().cpu(),
            'teacher_input':((target-own)/theta).flatten().cpu(),
            'zero_teacher':((target-zown)/theta).flatten().cpu(),
            'local_response_abs':delta.abs().mean((0,2,3)).cpu(),
            'local_changed_fraction':(delta!=0).float().mean((0,2,3)).cpu()}


@torch.no_grad()
def strict_reload(r, ident, biases, source):
    r.assign(biases)
    assert all(torch.equal(r.snapshot()[n], biases[n]) for n in r.names)
    before = r.e0.logits(r.model,r.test_images,r.parent,True).cpu()
    path=r.output/f'measurement-state-{ident}.pt'
    r.e0.atomic_torch(path,{'biases':r.snapshot(),'source':source,'parent_sha256':r.e0.sha256(r.c['parent_deployment']), 'thresholds':r.bn['interface_thresholds']})
    loaded=torch.load(path,map_location='cpu',weights_only=False)
    state=dict(r.bn['state_dict']); state.update({n+'.output_bias_correction':v for n,v in loaded['biases'].items()})
    r.restored.load_state_dict(state,strict=True)
    mods=dict(r.restored.named_modules())
    for n in r.names: mods[n].v_threshold=loaded['thresholds'][n]
    after=r.e0.logits(r.restored,r.test_images,r.parent,True).cpu()
    assert torch.equal(before,after), 'Actual bias strict reload mismatch'
    return {'saved':str(path),'saved_sha256':r.e0.sha256(path),'strict_reload':True,'fit_image_count':4}


def cpu_check(c):
    out=prepare(c,cpu=True)
    from modules import ParaInfNeuron
    m=ParaInfNeuron(2, th=1.,init_mem=.5,dim=5)
    m.output_bias_correction=torch.tensor([0.,.15,-.1]).reshape(1,1,3,1,1).repeat(2,1,1,1,1)
    x=torch.linspace(-.4,1.7,48).reshape(4,3,2,2)
    tx=x[:2].clone(); ty=tx.clamp(0,1)
    y=m(x); saved=m.output_bias_correction.clone()
    rr=residuals(m,x,y,tx,ty,2,1.)
    assert torch.equal(saved,m.output_bias_correction)
    manual=y.double().reshape(2,2,3,2,2).sum((0,1,3,4))/16
    assert torch.allclose(means(y,2).flatten().double(),manual,atol=1e-7,rtol=0)
    m.output_bias_correction.zero_(); r0=residuals(m,x,m(x),tx,ty,2,1.)
    assert torch.equal(r0['deployed'],r0['zero_current']) and r0['local_changed_fraction'].sum()==0
    a=aggregate(torch.tensor([[1.,-2.],[3.,2.]]))
    assert torch.equal(a['signed_global'],torch.tensor([2.,0.],dtype=torch.double))
    assert torch.equal(a['mean_absolute_batch'],torch.tensor([2.,2.],dtype=torch.double))
    assert not torch.cuda.is_initialized()
    for source in c['states'].values():
        d=torch.load(source,map_location='cpu',weights_only=False)
        assert len(d['biases'])==33 and all(torch.isfinite(v).all() for v in d['biases'].values())
    return {**out,'measurement_formula_checks':True,'state_files_checked':len(c['states'])}


@torch.no_grad()
def run(c):
    started=time.perf_counter(); r=prepare(c)
    T=r.parent['timesteps']; records={}; expected=torch.tensor([row['index'] for row in r.rows['fit']])
    assert len(expected)==c['fit_count'] and c['fit_count']%c['train_batch']==0 and len(r.rows['development'])==0
    batches=list(r.e0.loader(r.rows['fit'],r.parent,c['train_batch']))
    assert torch.equal(torch.cat([z[2] for z in batches]),expected)
    for ident,source in c['states'].items():
        state=torch.load(source,map_location='cpu',weights_only=False)
        assert state['parent_sha256']==r.e0.sha256(c['parent_deployment'])
        assert state['thresholds']==r.bn['interface_thresholds']
        biases=state['biases']; reload_info=strict_reload(r,ident,biases,source)
        raw={n:{} for n in r.names}; logits=[]; labels=[]; ids=[]
        for j,(x,y,i) in enumerate(batches):
            x=x.cuda(); th,handles=capture(r.teacher,r.names)
            try: r.e0.logits(r.teacher,x,r.parent)
            finally:
                for h in handles: h.remove()
            sh,handles=capture(r.model,r.names)
            try: pred=r.e0.logits(r.model,x,r.parent,True).cpu()
            finally:
                for h in handles: h.remove()
            assert set(th)==set(sh)==set(r.names)
            for n in r.names:
                values=residuals(r.modules[n],sh[n]['x'],sh[n]['y'],th[n]['x'],th[n]['y'],T,r.bn['interface_thresholds'][n])
                for key,value in values.items(): raw[n].setdefault(key,[]).append(value)
            logits.append(pred);labels.append(y);ids.append(i)
            del th,sh,x
        raw={n:{k:torch.stack(v) for k,v in layer.items()} for n,layer in raw.items()}
        assert all(v.shape[0]==len(batches) for layer in raw.values() for v in layer.values())
        assert all(torch.equal(r.snapshot()[n],biases[n]) for n in r.names)
        assert r.e0.state_digest(r.model,r.immutable)==r.initial_digest
        assert r.e0.state_digest(r.teacher)==r.teacher_digest
        pred=torch.cat(logits); lab=torch.cat(labels)
        records[ident]={'source':source,'source_sha256':r.e0.sha256(source),'reload':reload_info,'layers':raw,
                        'ids':torch.cat(ids),'labels':lab,'logits':pred,'top1':float((pred.argmax(1)==lab).double().mean()*100),
                        'ce':float(torch.nn.functional.cross_entropy(pred.double(),lab)), 'local_replays_verified':len(batches)*len(r.names)}
        r.e0.atomic_torch(r.output/'measurements.pt',records)
        r.progress('measured',state=ident,batches=len(batches),interfaces=len(r.names))
    torch.cuda.synchronize()
    result={'status':'completed','states':list(records),'unique_fit_images':len(expected),'batches':len(batches),'interfaces':len(r.names),
            'seconds':time.perf_counter()-started,'max_memory_reserved':torch.cuda.max_memory_reserved(),'max_memory_allocated':torch.cuda.max_memory_allocated(),
            'hardware':r.hardware,'data_status':'posthoc_fit_only_no_training_no_development','non_bias_frozen':True,'teacher_frozen':True,'all_strict_reloads':True}
    r.e0.atomic_json(r.output/'result.json',result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cpu-check',action='store_true');a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    try: print(json.dumps(cpu_check(c) if a.cpu_check else run(c)))
    except Exception:
        if not a.cpu_check:
            out=Path(c['output']);out.mkdir(exist_ok=True,parents=True);(out/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}))
        raise
