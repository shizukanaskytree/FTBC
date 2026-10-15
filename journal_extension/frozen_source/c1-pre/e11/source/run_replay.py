"""E11 fixed-state batch-shape replay. No fitting or development evaluation."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare
from trace_ops import local_input,counts,replay,trace_difference
import torch

def compare(a,b):
    d=(a-b).double().abs()
    return {'bitwise_equal':torch.equal(a,b),'max_abs':float(d.max()),'mean_abs':float(d.mean()),
            'prediction_disagreements':int((a.argmax(1)!=b.argmax(1)).sum())}

@torch.no_grad()
def run(c):
    start=time.perf_counter(); r=prepare(c);spiking=c['state']!='teacher'
    assert not r.rows['development'] and c['diagnostic_count']==8 and c['fit_count'] in (8,256)
    model=r.model if spiking else r.teacher
    if c['state']=='kl':
        saved=torch.load(c['kl_bias'],map_location='cpu',weights_only=False)
        assert saved['parent_sha256']==r.e0.sha256(c['parent_deployment'])
        assert saved['thresholds']==r.bn['interface_thresholds'];r.assign(saved['biases'])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    modules=dict(model.named_modules());assert all(n in modules for n in r.names)
    state_before=r.e0.state_digest(model)
    loaded=list(r.e0.loader(r.rows['fit'],r.parent,8))
    images=torch.cat([a for a,_,_ in loaded]).cuda(); labels=torch.cat([b for _,b,_ in loaded]); ids=torch.cat([d for _,_,d in loaded]);del loaded
    assert torch.equal(ids,torch.tensor([v['index'] for v in r.rows['fit']]))
    import hashlib
    input_hash=hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest()
    r.e0.atomic_torch(r.output/'input-identities.pt',{'ids':ids,'labels':labels,'image_tensor_sha256':input_hash,'sample_sha256':[v['sha256'] for v in r.rows['fit']]})
    predictions={}
    for key,batch in [('b4',4),('b8',8),('b4-repeat',4)]:
        out=torch.cat([r.e0.logits(model,x,r.parent,spiking).cpu() for x in images.split(batch)])
        assert torch.isfinite(out).all()
        predictions[key]={'ids':ids,'labels':labels,'logits':out,'predictions':out.argmax(1),
             'correct':int((out.argmax(1)==labels).sum()),'ce':float(torch.nn.functional.cross_entropy(out.double(),labels))}
        r.progress('replayed',state=c['state'],layout=key)
    r.e0.atomic_torch(r.output/'predictions.pt',predictions)
    anchors={};n=len(ids)
    if c['state']=='teacher':
        old=torch.load(c['teacher_scoring'],map_location='cpu',weights_only=False)['fit']
        eight=torch.load(c['teacher_fit'],map_location='cpu',weights_only=False)
        assert eight['teacher_digest']==r.teacher_digest
        for label,d in [('b4',old),('b8',eight)]:
            assert torch.equal(ids,d['ids'][:n]);anchors[label]=compare(predictions[label]['logits'],d['logits'][:n])
        assert torch.equal(labels,old['labels'][:n])
    else:
        old=torch.load(c['anchor_predictions'],map_location='cpu',weights_only=False)[{'bn':'bn-only','kl':'full-kl-lr0.001'}[c['state']]]['fit']
        assert torch.equal(ids,old['ids'][:n]) and torch.equal(labels,old['labels'][:n])
        anchors['b4']=compare(predictions['b4']['logits'],old['logits'][:n])
    strict_reload=None
    if spiking:
        actual=r.snapshot();r.e0.atomic_torch(r.output/'actual-bias.pt',{'biases':actual,'thresholds':r.bn['interface_thresholds'],'parent_sha256':r.e0.sha256(c['parent_deployment'])})
        saved=torch.load(r.output/'actual-bias.pt',map_location='cpu',weights_only=False)
        st=dict(r.bn['state_dict']);st.update({name+'.output_bias_correction':v for name,v in saved['biases'].items()})
        r.restored.load_state_dict(st,strict=True)
        for name in r.names:dict(r.restored.named_modules())[name].v_threshold=saved['thresholds'][name]
        restored=r.e0.logits(r.restored,images[:4],r.parent,True).cpu()
        strict_reload=torch.equal(restored,predictions['b4']['logits'][:4]);assert strict_reload
    traces={};trace_anchor={}
    for label,batch in [('b4',4),('b8',8)]:
        pieces={n:[] for n in r.names};handles=[]
        def hook(name):
            def record(module,args,out):
                pieces[name].append({'input':local_input(args[0],module,spiking).detach().cpu().clone(),'count':counts(out,module,spiking).cpu()})
            return record
        for name in r.names:handles.append(modules[name].register_forward_hook(hook(name)))
        try:out=torch.cat([r.e0.logits(model,x,r.parent,spiking).cpu() for x in images[:8].split(batch)])
        finally:
            for h in handles:h.remove()
        trace_anchor[label]=compare(out,predictions[label]['logits'][:8])
        traces[label]={name:{key:torch.cat([p[key] for p in parts]) for key in ['input','count']} for name,parts in pieces.items()}
        raw={'ids':ids[:8],'layers':traces[label],'logits':out,'thresholds':{name:float(modules[name].v_threshold if spiking else modules[name].up.item()) for name in r.names},'state':c['state'],'batch':batch}
        r.e0.atomic_torch(r.output/f'trace-{label}.pt',raw);del pieces
    diagnostics={};replays={}
    for name in r.names:
        a,b=traces['b4'][name],traces['b8'][name]
        aa,ok1=replay(modules[name],a['input'],spiking,4,'cuda')
        bb,ok2=replay(modules[name],b['input'],spiking,8,'cuda')
        ab,ok3=replay(modules[name],a['input'],spiking,8,'cuda')
        rec=trace_difference(a,b)
        rec.update(local_b4_reconstruct=torch.equal(aa,a['count']),local_b8_reconstruct=torch.equal(bb,b['count']),common_input_b4_b8_equal=torch.equal(aa,ab),repeated_mean_bitwise=ok1 and ok2 and ok3)
        diagnostics[name]=rec;replays[name]={'b4-input-b4':aa,'b8-input-b8':bb,'b4-input-b8':ab,'mean_identity':[ok1,ok2,ok3]}
    r.e0.atomic_torch(r.output/'local-replays.pt',replays)
    r.e0.atomic_json(r.output/'layer-diagnostics.json',diagnostics)
    assert r.e0.state_digest(model)==state_before and r.e0.state_digest(r.teacher)==r.teacher_digest
    assert r.e0.state_digest(r.model,r.immutable)==r.initial_digest
    torch.cuda.synchronize()
    result={'status':'completed','state':c['state'],'fit_count':n,'dev_count':0,'diagnostic_count':8,
        'repeat':compare(predictions['b4']['logits'],predictions['b4-repeat']['logits']),
        'batch_effect':compare(predictions['b4']['logits'],predictions['b8']['logits']),
        'anchors':anchors,'trace_anchors':trace_anchor,'strict_reload':strict_reload,'state_digest':state_before,
        'state_unchanged':True,'teacher_digest':r.teacher_digest,'teacher_unchanged':True,'non_bias_unchanged':True,
        'max_memory_reserved':torch.cuda.max_memory_reserved(),'max_memory_allocated':torch.cuda.max_memory_allocated(),
        'hardware':r.hardware,'seconds':time.perf_counter()-start,'image_tensor_sha256':input_hash,
        'source_module':str(Path(sys.modules['modules'].__file__).resolve()),
        'source_module_sha256':r.e0.sha256(sys.modules['modules'].__file__),
        'first_count_difference':next((name for name,v in diagnostics.items() if v['count_changed_elements']),None)}
    r.e0.atomic_json(r.output/'result.json',result);r.progress('completed');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);c=json.loads(Path(p.parse_args().config).read_text())
    try:print(json.dumps(run(c)))
    except Exception:
        Path(c['output']).mkdir(parents=True,exist_ok=True);(Path(c['output'])/'failure.json').write_text(json.dumps({'traceback':traceback.format_exc()}));raise
