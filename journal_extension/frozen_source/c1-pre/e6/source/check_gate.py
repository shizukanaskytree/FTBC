"""Exercise the exact 18-replay gate and gradient decomposition on a CPU inherited block."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'source-snapshot/parallel'))
from modules import ParaInfNeuron
from models.ResNet_QCFS import BasicBlock
from run_method import TrainNeuron,tail_forward,BNS,SPIKES
from direction_gate import run_gate
torch.set_num_threads(2);torch.manual_seed(10)
block=BasicBlock(2,3,3,False).eval()
block.residual_function[2]=ParaInfNeuron(2,th=.8,dim=5)
block.relu=ParaInfNeuron(2,th=1.1,dim=5)
for n in SPIKES:dict(block.named_modules())[n].output_bias_correction=torch.zeros(2,1,3,1,1)
tail=nn.Sequential(block,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,1000)).eval()
for p in tail.parameters():p.requires_grad_(False)
x=.5+.4*torch.randn(14,3,5,5);labels=torch.arange(7)
ws=[];base=[];beta=[]
for bn,sp in zip(BNS,SPIKES):
 old=dict(tail[0].named_modules())[sp];w=TrainNeuron(old,torch.full((3,),.3*float(old.v_threshold)),'comp',.1)
 parent,_,leaf=sp.rpartition('.');setattr(dict(tail[0].named_modules())[parent] if parent else tail[0],leaf,w)
 ws.append(w);base.append(dict(tail[0].named_modules())[bn].bias.clone());beta.append(nn.Parameter(torch.zeros(3)))
def forward():return tail_forward(tail,x,{'0.'+n+'.bias':b+w.theta*p for n,b,w,p in zip(BNS,base,ws,beta)})
def mode(m):
 for w in ws:w.mode=m
dest=E/'cache/gate-toy';dest.mkdir(exist_ok=True)
r=SimpleNamespace(output=dest,e0=SimpleNamespace(atomic_torch=lambda p,v:torch.save(v,p),atomic_json=lambda p,v:p.write_text(json.dumps(v,indent=2))))
out=run_gate(forward,beta,ws,mode,labels,{'direction_doses':[.01,.04],'learning_rate':.01,'gate_ce_tolerance':1e-6},r)
assert len(out['replays'])==18 and all(p.count_nonzero()==0 for p in beta)
assert max(out['gradient_decomposition_max_errors'])<1e-6
assert not torch.cuda.is_initialized()
result={'passed':True,'gate_pass_is_not_asserted_on_random_toy':out['passed'],
        'full_replays':18,'gradient_decomposition_max_errors':out['gradient_decomposition_max_errors'],
        'beta_reset':True,'cuda_initialized':False}
(E/'cache/cpu-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
