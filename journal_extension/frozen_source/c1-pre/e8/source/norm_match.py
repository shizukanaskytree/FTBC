"""Deterministic saved-bias interventions. No data or prediction access."""
import torch

def construct(root, old, thresholds, mode):
 assert mode in ('global','layer') and list(root)==list(old)==list(thresholds)
 names=list(root)
 def vec(d,n):
  b=d[n];assert torch.isfinite(b).all() and torch.equal(b,b[:1].expand_as(b))
  assert thresholds[n]>0
  return b[0].flatten().double()/thresholds[n]
 rv={n:vec(root,n) for n in names};gv={n:vec(old,n) for n in names}
 def ratio(g,r):
  rn=float(r.norm());gn=float(g.norm())
  if rn==0:
   assert gn==0,'nonzero target norm with zero direction'
   return 0.
  a=gn/rn;assert 0<=a<=1,'expansion violates frozen shrink-only design'
  return a
 a=ratio(torch.cat(list(gv.values())),torch.cat(list(rv.values()))) if mode=='global' else None
 scale={n:a if a is not None else ratio(gv[n],rv[n]) for n in names}
 out={n:root[n]*scale[n] for n in names}
 ov={n:vec(out,n) for n in names}
 for n in names:assert out[n].dtype==torch.float32 and float(out[n].abs().max())<=.500001*thresholds[n]
 def relative_error(x,y):
  target=float(y.norm());actual=float(x.norm())
  return abs(actual-target)/target if target else actual
 errors={n:relative_error(ov[n],gv[n]) for n in names}
 global_error=relative_error(torch.cat(list(ov.values())),torch.cat(list(gv.values())))
 assert (max(errors.values()) if mode=='layer' else global_error)<=1e-6
 return out,{'mode':mode,'scale':scale,'per_layer_norm_relative_error':errors,'global_norm_relative_error':global_error}
