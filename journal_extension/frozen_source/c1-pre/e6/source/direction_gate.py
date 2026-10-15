"""Fit-only hard-network probe of exact first Adam steps, before any training sweep."""
import torch
from torch.nn import functional as F


def run_gate(forward,beta,wrappers,set_mode,y,c,r):
    modes=('comp','stop','center');vectors={};origin={};rows=[];traces={}
    for mode in modes:
        set_mode(mode)
        for p in beta:p.grad=None
        out=forward();loss=F.cross_entropy(out,y);loss.backward()
        vectors[mode]=[p.grad.detach().clone() for p in beta]
        traces[mode]=[{k:v.clone() for k,v in w.gradient_trace.items()} for w in wrappers]
        origin[mode]={'ce':float(F.cross_entropy(out.detach().double(),y)),
                      'correct':int((out.detach().argmax(1)==y).sum())}
    assert len({z['ce'] for z in origin.values()})==1,'Different initial hard forwards'
    # Each probe uses a new optimizer, reproducing the actual epsilon/bias corrections.
    for mode in modes:
        set_mode(mode)
        for scope in ('both','interface-0','interface-1'):
            for dose in c['direction_doses']:
                opt=torch.optim.Adam(list(beta),lr=dose,betas=(.9,.999),eps=1e-8,weight_decay=0)
                for j,p in enumerate(beta):
                    p.grad=vectors[mode][j].clone() if scope in ('both','interface-'+str(j)) else torch.zeros_like(p)
                opt.step()
                with torch.no_grad():
                    out=forward()
                    rows.append({'mode':mode,'scope':scope,'learning_rate':dose,
                                 'ce':float(F.cross_entropy(out.double(),y)),
                                 'correct':int((out.argmax(1)==y).sum()),
                                 'layers':[dict(w.detail) for w in wrappers],
                                 'beta_abs_max_normalized':max(float(p.abs().max()) for p in beta)})
                    for p in beta:p.zero_()
    info=[]
    for j in range(len(beta)):
        a,b,k=vectors['comp'][j],vectors['stop'][j],vectors['center'][j]
        info.append({'interface':j,'channels':a.numel(),'full_graph_comp_stop_sign_reversals':int((a*b<0).sum()),
                     'comp_norm':float(a.norm()),'stop_norm':float(b.norm()),'center_norm':float(k.norm()),
                     'comp_stop_cosine':float(F.cosine_similarity(a[None],b[None])),
                     'gradient_below_1e-6':{m:int((vectors[m][j].abs()<1e-6).sum()) for m in modes}})
    primary={z['mode']:z['ce'] for z in rows if z['scope']=='both' and z['learning_rate']==c['learning_rate']}
    tol=c['gate_ce_tolerance']
    difference=primary['stop']-primary['comp']
    material=max(tol,.1*abs(primary['stop']-origin['stop']['ce']))
    passed=primary['comp']<min(origin['comp']['ce'],primary['stop'],primary['center'])-tol and difference>material
    out={'passed':passed,'initial':origin,'gradient_summary':info,'replays':rows,
         'material_comp_stop_ce_requirement':material,
         'criterion':'primary joint Adam: comp below initial/stop/center by > 1e-6; comp-stop benefit also >10% of absolute stop CE change',
         'scope':'one fit population, state and dose; not a general impossibility theorem or confirmation'}
    decomposition=[]
    for j,w in enumerate(wrappers):
        downstream=w.theta*(traces['comp'][j]['local_raw_beta']-traces['stop'][j]['local_raw_beta'])
        local=w.theta*traces['comp'][j]['local_correction_beta']
        actual=(vectors['comp'][j]-vectors['stop'][j]).cpu()
        decomposition.append({'interface':j,'downstream':downstream,'local':local,'actual':actual,
                              'max_reconstruction_error':float((downstream+local-actual).abs().max())})
    r.e0.atomic_torch(r.output/'gradient-decomposition.pt',{'traces':traces,'decomposition':decomposition})
    out['gradient_decomposition_max_errors']=[z['max_reconstruction_error'] for z in decomposition]
    r.e0.atomic_torch(r.output/'direction-vectors.pt',{m:[z.cpu() for z in v] for m,v in vectors.items()})
    r.e0.atomic_json(r.output/'direction-gate.json',out)
    for p in beta:p.grad=None
    return out
