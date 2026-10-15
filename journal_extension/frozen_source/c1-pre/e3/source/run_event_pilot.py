#!/usr/bin/env python3
"""E3 finite terminal pilot. GPU entry requires its Slurm allocation.

All actual model execution uses frozen E0 source. No official/confirmation data.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from event_solver import EventStream, LossOracle, full_scan, bounded_search, grid_coordinate, head_logits, Spike, sync


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def metrics(logits,labels):
    logits=logits.double(); labels=labels.cpu()
    pred=logits.argmax(1).cpu()
    return {'n':len(labels),'correct':int((pred==labels).sum()),
            'top1':float((pred==labels).double().mean()*100),
            'ce':float(F.cross_entropy(logits.cpu(),labels))}


def validate(config):
    manifest=json.loads(Path(config['freeze_manifest']).read_text())
    for path,sha in manifest['files_sha256'].items():
        if digest(path)!=sha:raise ValueError('Frozen identity mismatch: '+path)
    parent=json.loads(Path(config['parent_config']).read_text())
    e0=load_module('e3_frozen_e0',config['parent_runner'])
    roles=e0.validate_config(parent)
    selected=json.loads(Path(config['input_manifest']).read_text())
    assert selected['fit']==roles['fit'][:config['fit_count']]
    assert selected['development']==roles['development'][:config['dev_count']]
    for field in ('index','sha256'):
        a={r[field] for r in selected['fit']};b={r[field] for r in selected['development']}
        assert not a&b and len(a)==len(selected['fit']) and len(b)==len(selected['development'])
    train=Path(config['workspace'])/'datasets/imagenet/train'
    assert all(Path(r['path']).is_relative_to(train) for rows in selected.values() for r in rows)
    return parent,e0,selected,manifest


@torch.no_grad()
def capture(model,rows,parent,e0,channels,interface):
    module=dict(model.named_modules())[interface]
    values=[];outs=[];labels=[];ids=[];shape=[]
    def hook(m,inputs,out):
        x=inputs[0].reshape(parent['timesteps'],-1,*inputs[0].shape[1:])
        T,B,C,H,W=x.shape
        assert T==2 and C==model.fc.in_features
        p=m.TxT.view(T,1,1,1,1)*x.mean(0)+m.bias.view(T,1,1,1,1)
        sub=p[:,:,channels].permute(1,2,0,3,4).reshape(B,len(channels),T*H*W)
        response=out.reshape(T,B,C,H,W)[:,:,channels].permute(1,2,0,3,4).reshape_as(sub)
        expected=(((sub+torch.zeros_like(sub))-m.v_threshold)>=0)*m.v_threshold
        assert torch.equal(expected,response), 'Potential mapping differs from actual neuron'
        values.append(sub.cpu());shape.append((H,W))
    handle=module.register_forward_hook(hook)
    try:
        for x,y,i in e0.loader(rows,parent,4):
            o=e0.logits(model,x.cuda(),parent,True)
            outs.append(o.cpu());labels.append(y);ids.append(i)
    finally:handle.remove()
    assert len(set(shape))==1
    pot=torch.cat(values);logits=torch.cat(outs)
    return {'potential':pot,'logits':logits,'labels':torch.cat(labels),'ids':torch.cat(ids),
            'initial_counts':((pot-module.v_threshold)>=0).sum(-1),'spatial_shape':shape[0]}


def counts_logits(capture,delta,theta,weights,device='cuda'):
    with torch.no_grad():
        return head_logits(capture['potential'].to(device),torch.as_tensor(delta,device=device,dtype=torch.float32),
                           theta,capture['logits'].to(device),capture['initial_counts'].to(device),weights.to(device)).cpu()


@torch.no_grad()
def set_bias(model,interface,channels,delta):
    bias=dict(model.named_modules())[interface].output_bias_correction
    bias.zero_()
    bias[:,:,channels,0,0]=torch.as_tensor(delta,device=bias.device,dtype=bias.dtype).view(1,1,-1)


@torch.no_grad()
def evaluate_with_counts(model,rows,captured,parent,e0,channels,interface,delta):
    """Check integer responses at every replay point, not only small logit error."""
    module=dict(model.named_modules())[interface];offset=0
    expected=(((captured['potential']+torch.tensor(delta)[None,:,None])-module.v_threshold)>=0).sum(-1)
    def hook(m,inputs,out):
        nonlocal offset
        T=parent['timesteps'];B=out.shape[0]//T
        responses=out.reshape(T,B,*out.shape[1:])[:,:,channels].permute(1,2,0,3,4)
        assert torch.all((responses==0)|(responses==m.v_threshold))
        actual=(responses!=0).flatten(2).sum(-1).cpu()
        assert torch.equal(actual,expected[offset:offset+B]),'Actual integer pulse counts differ from event expression'
        offset+=B
    handle=module.register_forward_hook(hook)
    try:result=e0.evaluate(model,rows,parent,True)
    finally:handle.remove()
    assert offset==len(rows)
    return result


def run(config,cpu_only=False):
    started=time.perf_counter()
    torch.set_num_threads(8)
    parent,e0,rows,manifest=validate(config)
    source,classes=e0.source_model(parent)
    proxy=e0.make_proxy(source,parent,classes)
    proxy.load_state_dict(torch.load(config['parent_proxy'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
    weight=proxy.fc.weight.detach().cpu()
    order=sorted(range(weight.shape[1]),key=lambda c:(-float(weight[:,c].double().square().sum()),c))
    channels=config['channels'];assert channels==order[:len(channels)]
    if cpu_only:
        return {'passed':True,'cuda_initialized':torch.cuda.is_initialized(),'images_read':False,'channels':channels,
                'roles':{k:len(v) for k,v in rows.items()},'frozen_files':len(manifest['files_sha256'])}
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    if any((output/f).exists() for f in ('progress.json','failure.json','result.json')):
        raise ValueError('Attempt already exists; preserve it and use a new ID')
    def write(name,obj):e0.atomic_json(output/name,obj)
    def progress(stage,**kw):
        record={'stage':stage,'elapsed_seconds':time.perf_counter()-started,**kw}
        write('progress.json',record);print(json.dumps(record),flush=True)
    try:
        hardware=e0.gpu_guard({'allowed_uuid':config['allowed_uuid']});write('hardware.json',hardware)
        torch.manual_seed(config['seed']);np.random.seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        model=e0.make_snn(proxy,parent,classes)
        deployed=torch.load(config['parent_deployment'],map_location='cpu',weights_only=False)
        model.load_state_dict(deployed['state_dict'],strict=True)
        assert torch.equal(model.fc.weight.detach().cpu(),weight),'Proxy and deployed linear heads differ'
        assert parent.get('eval_batch_size',4)==4,'Capture/replay batch sizes must match'
        modules=dict(model.named_modules())
        for name,th in deployed['interface_thresholds'].items():modules[name].v_threshold=th
        interface=config['interface'];neuron=modules[interface];theta=float(neuron.v_threshold)
        bias_name=interface+'.output_bias_correction'
        frozen_names=[name for name in model.state_dict() if name!=bias_name]
        before_digest=e0.state_digest(model,frozen_names)
        assert not any(torch.count_nonzero(v) for v in e0.runtime_biases(model,parent,classes).values())
        setup_seconds=time.perf_counter()-started
        progress('capture');captures={};capture_seconds_by_role={}
        for role,subset in rows.items():
            sync('cuda');t=time.perf_counter()
            captures[role]=capture(model,subset,parent,e0,channels,interface)
            sync('cuda');capture_seconds_by_role[role]=time.perf_counter()-t
        capture_seconds=sum(capture_seconds_by_role.values())
        fit_capture_seconds=capture_seconds_by_role['fit']
        e0.atomic_torch(output/'captures.pt',captures)
        w=weight[:,channels].contiguous()
        K=len(channels);zero=np.zeros(K,dtype=np.float32)
        for c in captures.values():assert torch.equal(counts_logits(c,zero,theta,w),c['logits'].double())
        fit=captures['fit'];dev=captures['development'];N=len(fit['labels'])
        low,high=np.float32(-.5*theta),np.float32(.5*theta)
        records=[];predictions={};biases={};traces={}
        parent_anchor=torch.load(config['parent_predictions'],map_location='cpu',weights_only=True)['spatial_snn']
        assert torch.equal(dev['ids'],parent_anchor['ids'][:len(dev['ids'])])
        assert torch.equal(dev['labels'],parent_anchor['labels'][:len(dev['ids'])])
        assert torch.equal(dev['logits'],parent_anchor['logits'][:len(dev['ids'])]),'E0 anchor changed'
        def score(name,delta,solver_seconds,family,detail=None):
            fit_l=counts_logits(fit,delta,theta,w);dev_l=counts_logits(dev,delta,theta,w)
            counts=(((fit['potential']+torch.tensor(delta)[None,:,None])-theta)>=0).sum()
            r={'id':name,'family':family,'fit':metrics(fit_l,fit['labels']),
               'development':metrics(dev_l,dev['labels']),'solver_seconds':float(solver_seconds),
               'with_capture_seconds':float(solver_seconds+fit_capture_seconds),
               'selected_interface_spikes_fit':int(counts),'delta':[float(x) for x in delta]}
            if detail:r['detail']=detail
            records.append(r);predictions[name]={'fit_logits':fit_l,'development_logits':dev_l};biases[name]=delta.copy()
            write('metrics-progress.json',records)
            progress('score',candidate=name,development=r['development'])
        score('bn-only',zero,0.,'baseline')
        # Warm numeric kernels with an excluded 8-image stream before cost comparisons.
        sync('cuda');warm_start=time.perf_counter()
        warm_x=fit['potential'][:8,0].numpy();warm_y=fit['labels'][:8].numpy()
        warm_v=w[:,0].double().numpy()*(theta/fit['potential'].shape[-1])
        warm_base=fit['logits'][:8].double().numpy()-fit['initial_counts'][:8,0].numpy()[:,None]*warm_v
        warm_stream=EventStream(warm_x,theta,low,high)
        full_scan(warm_stream,LossOracle(warm_base,warm_v,warm_y,device='cuda'),0.)
        bounded_search(warm_stream,LossOracle(warm_base,warm_v,warm_y,device='cuda'),0.,.05)
        grid_coordinate(warm_x,warm_base,warm_v,warm_y,theta,low,high,0.,9,'cuda')
        warm_delta=torch.nn.Parameter(torch.zeros(K,device='cuda'))
        warm_optimizer=torch.optim.Adam([warm_delta],lr=theta*.001)
        warm_pot=fit['potential'][:8].cuda();warm_initial=fit['initial_counts'][:8].cuda()
        warm_counts=Spike.apply((warm_pot+warm_delta[None,:,None])-theta).sum(-1)
        warm_logits=fit['logits'][:8].cuda()+(warm_counts-warm_initial)@(w.cuda()*(theta/warm_pot.shape[-1])).T
        F.cross_entropy(warm_logits,fit['labels'][:8].cuda()).backward();warm_optimizer.step()
        del warm_delta,warm_optimizer,warm_pot,warm_initial,warm_counts,warm_logits
        sync('cuda');warmup_seconds=time.perf_counter()-warm_start
        # B1 gradient baseline: same cached input, original surrogate, all K biases jointly.
        for ratio in config['ce_lr_ratios']:
            progress('ce',lr_ratio=ratio);delta=torch.nn.Parameter(torch.zeros(K,device='cuda'))
            optimizer=torch.optim.Adam([delta],lr=float(theta)*ratio,weight_decay=0.)
            pot=fit['potential'].cuda();labels=fit['labels'].cuda();start_counts=fit['initial_counts'].cuda()
            baseline=fit['logits'].cuda();original_weights=w.cuda();v=original_weights*(theta/pot.shape[-1])
            best=zero.copy();best_loss=float(F.cross_entropy(baseline.double(),labels));trace=[];elapsed=0.
            for step in range(1,config['ce_steps']+1):
                sync('cuda');ts=time.perf_counter();optimizer.zero_grad()
                count=Spike.apply((pot+delta[None,:,None])-theta).sum(-1)
                logits=baseline+(count-start_counts)@v.T
                loss=F.cross_entropy(logits,labels);assert torch.isfinite(loss)
                loss.backward();optimizer.step()
                with torch.no_grad():delta.clamp_(float(low),float(high))
                current=delta.detach().cpu().numpy().copy()
                with torch.no_grad():
                    true_loss=float(F.cross_entropy(head_logits(pot,delta.detach(),theta,baseline,start_counts,original_weights),labels))
                if true_loss<best_loss-1e-10:best_loss=true_loss;best=current.copy()
                sync('cuda');elapsed+=time.perf_counter()-ts
                trace.append({'step':step,'current_fit_ce':true_loss,'best_fit_ce':best_loss,'seconds':elapsed})
                if step in config['ce_checkpoints']:
                    score(f'ce-{ratio:g}-s{step}',best,elapsed,'ce')
            traces[f'ce-{ratio:g}']=trace
            del pot,labels,start_counts,baseline,v,original_weights,optimizer,delta
        # B1 full scan and grid: identical coordinate order and sweep count.
        micro=None;bound_result=None
        for family,points in [('fullscan',None)]+[('grid'+str(p),p) for p in config['grid_points']]:
            delta=zero.copy();elapsed=0.;trace=[]
            for sweep in range(1,config['sweeps']+1):
                for k in range(K):
                    progress('coordinate',family=family,sweep=sweep,channel=channels[k])
                    sync('cuda');ts=time.perf_counter()
                    current=counts_logits(fit,delta,theta,w).numpy()
                    counts=(((fit['potential'][:,k]+delta[k])-theta)>=0).sum(-1).numpy()
                    vector=w[:,k].double().numpy()*(theta/fit['potential'].shape[-1])
                    base=current-counts[:,None]*vector
                    x=fit['potential'][:,k].numpy();labels=fit['labels'].numpy()
                    if points is None:
                        stream=EventStream(x,theta,low,high)
                        oracle=LossOracle(base,vector,labels,device='cuda')
                        answer=full_scan(stream,oracle,delta[k]);objectives=answer.pop('objectives')
                        assert answer['loss_queries']<=stream.support_count
                    else:
                        answer=grid_coordinate(x,base,vector,labels,theta,low,high,delta[k],points,'cuda')
                    delta[k]=np.float32(answer['bias'])
                    sync('cuda');elapsed+=time.perf_counter()-ts
                    item={'sweep':sweep,'channel':channels[k],**answer,'cumulative_seconds':elapsed}
                    trace.append(item)
                    if family=='fullscan' and sweep==1 and k==0:
                        # Same stream construction charge for both, separate strong caches.
                        limit=min(120.,max(5.,2*answer['seconds']))
                        bound_result=bounded_search(stream,LossOracle(base,vector,labels,device='cuda'),0.,limit)
                        bound_result['fullscan']=answer
                        bound_result['loss_sum_delta_vs_fullscan']=bound_result['loss_sum']-answer['loss_sum']
                        bound_result['loss_query_ratio']=bound_result['loss_queries']/answer['loss_queries']
                        bound_result['total_time_ratio']=(bound_result['seconds']+stream.seconds)/(answer['seconds']+stream.seconds)
                        assert bound_result['loss_queries']<=stream.support_count
                        assert bound_result['lower_bound_sum']<=answer['loss_sum']+1e-7
                        assert bound_result['loss_sum']>=answer['loss_sum']-1e-7
                        write('bound-comparison.json',bound_result)
                        # Every state on 8 images: recompute all image losses independently.
                        tiny=EventStream(x[:8],theta,low,high)
                        tiny_scan=full_scan(tiny,LossOracle(base[:8],vector,labels[:8],device='cuda'),0.)
                        brute=[];sync('cuda');micro_start=time.perf_counter()
                        xx=torch.as_tensor(x[:8],device='cuda');bb=torch.as_tensor(base[:8],device='cuda');vv=torch.as_tensor(vector,device='cuda')
                        yy=torch.as_tensor(labels[:8],device='cuda')
                        for first in range(0,len(tiny.params),32):
                            p=torch.as_tensor(tiny.params[first:first+32],device='cuda')
                            ct=(((xx[None]+p[:,None,None])-theta)>=0).sum(-1)
                            ll=bb[None]+ct[:,:,None]*vv
                            ce=F.cross_entropy(ll.reshape(-1,ll.shape[-1]),yy.repeat(len(p)),reduction='none').reshape(len(p),-1).sum(1)
                            brute.extend(ce.cpu().tolist())
                        sync('cuda');brute_seconds=time.perf_counter()-micro_start
                        error=float(np.max(np.abs(np.array(brute)-tiny_scan.pop('objectives'))))
                        assert error<=1e-7
                        micro={'images':8,'states':len(tiny.params),'max_loss_sum_error':error,
                               'full_recompute_seconds':brute_seconds,'cached_scan':tiny_scan,
                               'scope':'terminal head and spikes only, not repeated backbone'}
                        write('micro-ablation.json',micro)
                score(f'{family}-p{sweep}',delta,elapsed,family)
            traces[family]=trace
        progress('actual_replay_and_reload')
        # Check all preregistered points on the actual dev network, and save real full bias buffers.
        prediction_path=output/'predictions.pt'
        deployment_checks={}
        for r in records:
            sync('cuda');replay_start=time.perf_counter()
            name=r['id'];delta=biases[name]
            set_bias(model,interface,channels,delta)
            actual=evaluate_with_counts(model,rows['development'],dev,parent,e0,channels,interface,delta)
            expected=predictions[name]['development_logits']
            error=float((actual['logits'].double()-expected).abs().max())
            ce_error=abs(metrics(actual['logits'],actual['labels'])['ce']-r['development']['ce'])
            assert error<=2e-4 and ce_error<=1e-5
            assert torch.equal(actual['logits'].argmax(1),expected.argmax(1)), 'Cached top1 differs'
            assert torch.equal(actual['ids'],dev['ids']) and torch.equal(actual['labels'],dev['labels'])
            actual_fit=evaluate_with_counts(model,rows['fit'],fit,parent,e0,channels,interface,delta)
            expected_fit=predictions[name]['fit_logits']
            fit_error=float((actual_fit['logits'].double()-expected_fit).abs().max())
            fit_ce_error=abs(metrics(actual_fit['logits'],actual_fit['labels'])['ce']-r['fit']['ce'])
            assert fit_error<=2e-4 and fit_ce_error<=1e-5
            assert torch.equal(actual_fit['logits'].argmax(1),expected_fit.argmax(1))
            assert torch.equal(actual_fit['ids'],fit['ids']) and torch.equal(actual_fit['labels'],fit['labels'])
            actual_bias=neuron.output_bias_correction.detach().cpu().clone()
            saved={'parent_deployment':config['parent_deployment'],'parent_sha256':digest(config['parent_deployment']),
                   'interface':interface,'output_bias_correction':actual_bias,'channels':channels,
                   'threshold':theta,'T':2,'config_sha256':config['_config_sha256']}
            path=output/f'bias-{name}.pt';e0.atomic_torch(path,saved)
            first=next(iter(e0.loader(rows['development'][:4],parent,4)))[0].cuda()
            before=e0.logits(model,first,parent,True).detach().cpu()
            # Reset to the complete parent and load the actually serialized buffer.
            model.load_state_dict(deployed['state_dict'],strict=True)
            loaded=torch.load(path,map_location='cpu',weights_only=True)
            reloaded_state=dict(deployed['state_dict']);reloaded_state[bias_name]=loaded['output_bias_correction']
            model.load_state_dict(reloaded_state,strict=True)
            after=e0.logits(model,first,parent,True).detach().cpu()
            assert torch.equal(before,after) and torch.equal(neuron.output_bias_correction.cpu(),actual_bias)
            assert e0.state_digest(model,frozen_names)==before_digest
            deployment_checks[name]={'max_logit_error':error,'mean_ce_error':ce_error,'top1_equal':True,
                                     'fit_max_logit_error':fit_error,'fit_mean_ce_error':fit_ce_error,
                                     'actual_integer_counts_equal_fit_and_dev':True,'strict_model_state_reload':True,
                                     'exact_reload':True,'non_bias_state_unchanged':True,
                                     'nonzero_bias_entries':int(torch.count_nonzero(actual_bias)),
                                     'bias_sha256':digest(path)}
            predictions[name]['actual_development']=actual
            predictions[name]['actual_fit']=actual_fit
            r['actual_development']=metrics(actual['logits'],actual['labels'])
            r['actual_fit']=metrics(actual_fit['logits'],actual_fit['labels'])
            sync('cuda');r['replay_verification_seconds']=time.perf_counter()-replay_start
            r['with_capture_and_replay_seconds']=r['with_capture_seconds']+r['replay_verification_seconds']
        e0.atomic_torch(prediction_path,{'candidates':predictions,'fit_labels':fit['labels'],'fit_ids':fit['ids'],
                                      'development_labels':dev['labels'],'development_ids':dev['ids']})
        write('solver-traces.json',traces);write('deployment-checks.json',deployment_checks)
        for r in records:
            r['calibration_cost_seconds']=setup_seconds+fit_capture_seconds+warmup_seconds+r['solver_seconds']
        selected=sorted(records,key=lambda r:(-r['development']['top1'],r['development']['ce'],r['solver_seconds'],r['id']))[0]['id']
        for path,sha in manifest['files_sha256'].items():assert digest(path)==sha
        sync('cuda')
        result={'status':'completed','experiment':'C1-pre-E3','phase':config['phase'],
                'scope':'one-setting one-sample exploratory terminal pilot; protected evaluation closed',
                'hardware':hardware,'channels':channels,'theta':theta,'capture_seconds':capture_seconds,
                'capture_seconds_by_role':capture_seconds_by_role,'warmup_seconds':warmup_seconds,
                'shared_setup_seconds':setup_seconds,
                'records':records,'selected_development_candidate':selected,'all_deployment_checks_passed':True,
                'micro':micro,'bound':bound_result,'input_counts':{k:len(v) for k,v in rows.items()},
                'spatial_shape':fit['spatial_shape'],'elapsed_seconds_including_setup_preprocessing':time.perf_counter()-started,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'config_sha256':config['_config_sha256'],'freeze_sha256':digest(config['freeze_manifest']),
                'inference_latency_measured':False,'parent_state_unchanged':True}
        write('result.json',result)
        write('artifacts.json',{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in output.iterdir()
                                if p.is_file() and p.name not in ('artifacts.json','progress.json')})
        progress('completed',selected=selected)
        return result
    except BaseException as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
                              'elapsed_seconds':time.perf_counter()-started,'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());config['_config_sha256']=digest(args.config)
    value=run(config,args.cpu_check)
    if args.cpu_check:print(json.dumps(value))
