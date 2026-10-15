"""Finite FP32 hard mean solver. No global-discrete-optimality claim."""
import torch
REDUCE=(0,1,3,4)

def native_base(module,x):
 T=module.T;x=x.reshape(T,-1,*x.shape[1:]);z=x.mean(0)
 assert getattr(module,'learned_calibration',None) is None
 assert getattr(module,'runtime_output_bias_correction',None) is None
 return module.TxT.view(T,1,1,1,1)*z+module.bias.view(T,1,1,1,1)

def hard(base,delta,theta):
 return (((base+delta.reshape(1,1,-1,1,1))-theta)>=0).to(base.dtype)*theta

def counts(base,delta,theta):
 return torch.count_nonzero((base+delta.reshape(1,1,-1,1,1))-theta>=0,dim=REDUCE)

@torch.no_grad()
def solve(base,target,theta,steps=32,bound=.5):
 assert base.dtype==torch.float32 and torch.isfinite(base).all()
 C=base.shape[2];target=target.to(device=base.device,dtype=torch.float64)
 assert target.shape==(C,) and torch.isfinite(target).all() and theta>0
 n=base.numel()//C
 lo=torch.full((C,),-bound*theta,device=base.device);hi=-lo
 lower=counts(base,lo,theta);upper=counts(base,hi,theta)
 def mean(c):return c.double()*theta/n
 for _ in range(steps):
  mid=(lo+hi)/2;less=mean(counts(base,mid,theta))<target
  lo=torch.where(less,mid,lo);hi=torch.where(less,hi,mid)
 zero=torch.zeros_like(lo);boundlo=torch.full_like(lo,-bound*theta);boundhi=-boundlo
 best=lo.clone();bestcount=counts(base,best,theta);err=(mean(bestcount)-target).abs()
 for d in [hi,zero,boundlo,boundhi]:
  c=counts(base,d,theta);e=(mean(c)-target).abs()
  take=(e<err)|((e==err)&((d.abs()<best.abs())|((d.abs()==best.abs())&(d<best))))
  best=torch.where(take,d,best);bestcount=torch.where(take,c,bestcount);err=torch.where(take,e,err)
 zcount=counts(base,zero,theta)
 assert torch.all(err<=(mean(zcount)-target).abs())
 assert torch.all(best.abs()<=bound*theta)
 return best,{'target_mean':target.cpu(),'before_count':zcount.cpu(),'after_count':bestcount.cpu(),
  'lower_count':lower.cpu(),'upper_count':upper.cpu(),'lo_count':counts(base,lo,theta).cpu(),
  'hi_count':counts(base,hi,theta).cpu(),'lo':lo.cpu(),'hi':hi.cpu(),'delta':best.cpu(),
  'normalized_residual':((mean(bestcount)-target)/theta).cpu(),'observations_per_channel':n,'theta':theta,
  'below_range':(target<mean(lower)).cpu(),'above_range':(target>mean(upper)).cpu(),
  'at_lower_mean':(target==mean(lower)).cpu(),'at_upper_mean':(target==mean(upper)).cpu(),
  'bracketed':((target>mean(lower))&(target<mean(upper))).cpu(),
  'float_adjacent':(torch.nextafter(lo,torch.full_like(lo,float('inf')))>=hi).cpu()}
