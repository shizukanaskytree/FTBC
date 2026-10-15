"""CPU real inherited BasicBlock: functional BN beta, original neuron reload, all arms."""
import copy,json,sys
from pathlib import Path
import torch
from torch import nn
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'source-snapshot/parallel'))
from modules import ParaInfNeuron
from models.ResNet_QCFS import BasicBlock
from run_method import TrainNeuron,tail_forward,BNS,SPIKES
torch.set_num_threads(2);torch.manual_seed(10)
block=BasicBlock(2,3,3,False)
block.residual_function[2]=ParaInfNeuron(2,th=.8,dim=5)
block.relu=ParaInfNeuron(2,th=1.1,dim=5)
for n in SPIKES:dict(block.named_modules())[n].output_bias_correction=torch.zeros(2,1,3,1,1)
base=nn.Sequential(block,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,1000)).eval()
for p in base.parameters():p.requires_grad_(False)
x=.5+.4*torch.randn(14,3,5,5);labels=torch.arange(7)
result=[]
for mode in ['comp','stop','free-ce','free-kl','beta-ce']:
 tail=copy.deepcopy(base);ws=[];initial=[];ps=[];scales=[]
 for bn,sp in zip(BNS,SPIKES):
  old=dict(tail[0].named_modules())[sp];s=float(old.v_threshold)
  w=TrainNeuron(old,torch.full((3,),.3*s),mode,.1)
  parent,_,leaf=sp.rpartition('.');setattr(dict(tail[0].named_modules())[parent] if parent else tail[0],leaf,w)
  ws.append(w);scales.append(s);initial.append(dict(tail[0].named_modules())[bn].bias.clone())
  ps.append(nn.Parameter(torch.zeros(3)))
 def f():return tail_forward(tail,x,{'0.'+n+'.bias':b+s*p for n,b,s,p in zip(BNS,initial,scales,ps)})
 with torch.no_grad():
  for w in ws:w.mode='stop'
  calibrated_start=f()
  for w in ws:w.delta0.copy_(w.delta);w.free.zero_();w.mode=mode
  if mode!='beta-ce':assert torch.equal(f(),calibrated_start),'Calibration/free initial hard state differs'
 train=ps+([w.free for w in ws] if mode.startswith('free') else [])
 opt=torch.optim.Adam(train,lr=.01)
 for step in range(2):
  opt.zero_grad();z=f();loss=nn.functional.cross_entropy(z,labels);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in train)
  opt.step()
 with torch.no_grad():
  expected=f();deployment=copy.deepcopy(base)
  for bn,sp,b,s,p,w in zip(BNS,SPIKES,initial,scales,ps,ws):
   dict(deployment[0].named_modules())[bn].bias.copy_(b+s*p)
   dict(deployment[0].named_modules())[sp].output_bias_correction=w.delta.reshape(1,1,3,1,1).repeat(2,1,1,1,1)
  actual=tail_forward(deployment,x,{})
  assert torch.equal(expected,actual),(mode,(expected-actual).abs().max())
 result.append({'arm':mode,'original_block_deployment_bitwise_equal':True,'two_adam_steps_finite':True,
                'calibrated_start_exact':True if mode!='beta-ce' else 'separate zero-delta arm'})
assert not torch.cuda.is_initialized()
out={'passed':True,'checks':result,'cuda_initialized':False}
(E/'cache/cpu-real-block.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
