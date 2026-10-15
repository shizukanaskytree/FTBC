import hashlib,json,sys,ast
from pathlib import Path
import torch
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'e4/source'))
from common import prepare
from directions import construct,bind,clamp

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    c=json.loads((E/'config/main-root.json').read_text());report=prepare(c,cpu=True)
    saved=torch.load(c['references']['finite-solve']['bias'],map_location='cpu',weights_only=False)
    root=saved['biases'];theta=saved['thresholds'];allbasis={}
    for mode in ['root','permuted','constant']:
        b,lim,meta=construct(root,theta,mode,c['permutation_seed']);allbasis[mode]=b
        assert list(b)==report['interfaces']
        for n,v in b.items():
            assert abs(float(v.square().mean())-1)<2e-7 or not torch.count_nonzero(v)
            assert float((lim[n]*v).abs().max())<=.5+1e-7
        assert all(torch.equal(v,construct(root,theta,mode,c['permutation_seed'])[0][n]) for n,v in b.items())
    assert all(torch.equal(torch.sort(allbasis['root'][n]).values,torch.sort(allbasis['permuted'][n]).values) for n in root)
    toy={'n':torch.tensor([-.5,.1,.3]).view(1,1,3,1,1).expand(2,1,3,1,1).clone(),'z':torch.zeros(2,1,2,1,1)}
    from types import SimpleNamespace
    for mode in ['root','permuted','constant']:
        b,lim,_=construct(toy,{'n':1.,'z':1.},mode,7)
        params={n:torch.nn.Parameter(torch.ones(1)*100) for n in toy};mods={n:SimpleNamespace() for n in toy}
        clamp(params,lim);bind(mods,params,b,{'n':1.,'z':1.},2)
        assert params['z']==0 and not torch.count_nonzero(mods['z'].output_bias_correction)
        for n,m in mods.items():assert m.output_bias_correction.abs().max()<=.5000001 and torch.equal(m.output_bias_correction[0],m.output_bias_correction[1])
        sum((m.output_bias_correction*torch.arange(m.output_bias_correction.numel()).reshape_as(m.output_bias_correction)).sum() for m in mods.values()).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    from modules import ParaInfNeuron
    mods={'n':ParaInfNeuron(2,th=1.,init_mem=.5,dim=5)}
    mods['n'].output_bias_correction=torch.zeros(2,1,3,1,1)
    q={'n':torch.nn.Parameter(torch.zeros(1))};b={'n':torch.ones(3)}
    bind(mods,q,b,{'n':1.},2);mods['n'](torch.full((4,3,2,2),.51)).sum().backward()
    assert q['n'].grad is not None and q['n'].grad.abs()>0
    for p in (E/'source').glob('*.py'):ast.parse(p.read_text())
    for p in (E/'config').glob('*.json'):
        d=json.loads(p.read_text());assert d['fit_count']%d['train_batch']==0 and d['epochs']==(1 if d['sanity'] else 4)
    report.update(directions_matched=True,zero_and_boundary_checked=True,actual_neuron_gradient=True,active_dimensions=sum(x['active'] for x in meta.values()),cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized']
    files=list((E/'source').glob('*.py'))+list((E/'config').glob('*.json'))+[E/'EXPERIMENT_PLAN.md']
    report['files_sha256']={str(p):sha(p) for p in files}
    (E/'cache/cpu-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['files_sha256','interfaces']}))
if __name__=='__main__':check()
