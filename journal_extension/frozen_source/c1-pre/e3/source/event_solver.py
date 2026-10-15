"""Finite-event reference and optional reachable-support pruning.

No model, image, CUDA initialization or implicit execution on import.
The objective is an FP64 affine terminal readout; deployment is checked separately.
"""
import bisect
import heapq
import time
import numpy as np
import torch
import torch.nn.functional as F


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def critical_bias(potential, theta):
    """First FP32 delta whose actual rounded add/subtract fires.

    Build near the rounding midpoint below theta, then check both neighboring
    representable values. No real-root claim is substituted for this check.
    """
    x = np.asarray(potential, dtype=np.float32)
    th = np.float32(theta)
    below = np.nextafter(th, np.float32(-np.inf))
    midpoint = (float(th) + float(below))/2
    b = (midpoint - x.astype(np.float64)).astype(np.float32)
    assert np.isfinite(b).all() and np.isfinite(x).all() and th > 0
    for _ in range(8):
        fires = ((x+b)-th) >= 0
        prev = np.nextafter(b, np.float32(-np.inf))
        prev_fires = ((x+prev)-th) >= 0
        if fires.all() and not prev_fires.any():
            return b
        b = np.where(~fires, np.nextafter(b, np.float32(np.inf)),
                     np.where(prev_fires, prev, b)).astype(np.float32)
    raise AssertionError('FP32 transition adjustment did not converge')


class EventStream:
    def __init__(self, potential, theta, low, high):
        start=time.perf_counter()
        x=np.asarray(potential,dtype=np.float32)
        self.n, self.width=x.shape
        self.low,self.high=np.float32(low),np.float32(high)
        roots=critical_bias(x,theta)
        initial=(((x+self.low)-np.float32(theta))>=0).sum(1)
        self.local=[]
        root_lists=[]
        for n in range(self.n):
            r,m=np.unique(roots[n][(roots[n]>self.low)&(roots[n]<=self.high)],return_counts=True)
            counts=np.r_[initial[n],initial[n]+m.cumsum()].astype(np.int64)
            self.local.append({'roots':r,'counts':counts})
            root_lists.append(r)
        self.roots=np.unique(np.concatenate(root_lists))
        self.params=np.r_[self.low,self.roots].astype(np.float32)
        for row in self.local:
            row['states']=np.r_[0,np.searchsorted(self.roots,row['roots'])+1].astype(np.int64)
        self.seconds=time.perf_counter()-start
        self.event_count=int(sum(len(row['roots']) for row in self.local))
        self.support_count=int(sum(len(row['counts']) for row in self.local))

    def counts(self,p):
        return np.array([r['counts'][np.searchsorted(r['roots'],p,side='right')] for r in self.local])

    def ranks(self,state):
        return np.array([np.searchsorted(r['states'],state,side='right')-1 for r in self.local])


class LossOracle:
    def __init__(self, base, vector, labels, device='cpu', chunk=4096):
        self.device=torch.device(device)
        self.base=torch.as_tensor(base,dtype=torch.float64,device=device)
        self.vector=torch.as_tensor(vector,dtype=torch.float64,device=device)
        self.labels=torch.as_tensor(labels,dtype=torch.long,device=device)
        self.chunk=chunk
        self.cache={}

    def values(self,pairs):
        pairs=[(int(n),int(m)) for n,m in pairs]
        missing=list(dict.fromkeys(p for p in pairs if p not in self.cache))
        for start in range(0,len(missing),self.chunk):
            block=missing[start:start+self.chunk]
            n=torch.tensor([p[0] for p in block],device=self.device)
            m=torch.tensor([p[1] for p in block],dtype=torch.float64,device=self.device)
            logits=self.base[n]+m[:,None]*self.vector
            losses=F.cross_entropy(logits,self.labels[n],reduction='none').detach().cpu().numpy()
            for key,val in zip(block,losses):self.cache[key]=float(val)
        return np.array([self.cache[p] for p in pairs])


def full_scan(stream,oracle,current):
    sync(oracle.device);start=time.perf_counter()
    pairs=[(n,m) for n,row in enumerate(stream.local) for m in row['counts']]
    values=oracle.values(pairs)
    deltas=np.zeros(len(stream.params),dtype=np.float64)
    offset=0; initial=0.
    for row in stream.local:
        vals=values[offset:offset+len(row['counts'])];offset+=len(vals)
        initial+=vals[0]
        np.add.at(deltas,row['states'][1:],np.diff(vals))
    objectives=initial+np.cumsum(deltas)
    best=int(np.argmin(objectives))
    current_loss=float(oracle.values(enumerate(stream.counts(current))).sum())
    chosen=float(stream.params[best]) if objectives[best]<current_loss-1e-10 else float(current)
    sync(oracle.device)
    return {'bias':chosen,'loss_sum':min(current_loss,float(objectives[best])),
            'seconds':time.perf_counter()-start,'loss_queries':len(oracle.cache),
            'states':len(stream.params),'stream_seconds':stream.seconds,
            'event_count':stream.event_count,'objectives':objectives}


def bounded_search(stream,oracle,current,seconds):
    sync(oracle.device);start=time.perf_counter()
    n=stream.n
    left=np.zeros(n,dtype=int)
    right=np.array([len(r['counts'])-1 for r in stream.local])
    # Convex loss restricted to sorted support is unimodal; search support ranks.
    while np.any(left<right):
        ids=np.flatnonzero(left<right);mid=(left[ids]+right[ids])//2
        a=oracle.values((i,stream.local[i]['counts'][m]) for i,m in zip(ids,mid))
        b=oracle.values((i,stream.local[i]['counts'][m+1]) for i,m in zip(ids,mid))
        for i,m,up in zip(ids,mid,b>=a):
            if up:right[i]=m
            else:left[i]=m+1
    minima=left.copy()
    def values_at(ranks):
        return float(oracle.values((i,stream.local[i]['counts'][r]) for i,r in enumerate(ranks)).sum())
    def lower(a,b):return values_at(np.maximum(stream.ranks(a),np.minimum(minima,stream.ranks(b))))
    cache={}
    def evaluate(i):
        if i not in cache:cache[i]=values_at(stream.ranks(i))
        return cache[i]
    best=float(oracle.values(enumerate(stream.counts(current))).sum());bias=float(current)
    last=len(stream.params)-1
    for i in sorted({0,last,last//2}):
        value=evaluate(i)
        if value<best-1e-10:best=value;bias=float(stream.params[i])
    queue=[(lower(0,last),0,last)];terminal=[];count=0
    while queue and time.perf_counter()-start<seconds:
        bound,a,b=heapq.heappop(queue);count+=1
        if bound>=best-1e-9:
            terminal.append(bound);continue
        if a==b:
            value=evaluate(a)
            if value<best-1e-10:best=value;bias=float(stream.params[a])
            terminal.append(bound);continue
        mid=(a+b)//2
        for i in (mid,mid+1):
            value=evaluate(i)
            if value<best-1e-10:best=value;bias=float(stream.params[i])
        for lo,hi in ((a,mid),(mid+1,b)):heapq.heappush(queue,(lower(lo,hi),lo,hi))
    lower_bound=min([best]+terminal+[r[0] for r in queue])
    sync(oracle.device)
    return {'bias':bias,'loss_sum':best,'lower_bound_sum':lower_bound,'gap_sum':best-lower_bound,
            'completed':not queue,'seconds':time.perf_counter()-start,
            'loss_queries':len(oracle.cache),'intervals':count,'candidate_evaluations':len(cache),
            'budget_seconds':seconds,'stream_seconds':stream.seconds}


def head_logits(potential,delta,theta,base_logits,initial_counts,weights):
    # potential [N,K,TM], weights [J,K], both float32; preserve original add/subtract.
    counts=(((potential+delta[None,:,None])-theta)>=0).sum(-1)
    v=weights.double()*(float(theta)/potential.shape[-1])
    return base_logits.double()+(counts-initial_counts).double()@v.T


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        return grad*(1-x.abs()).clamp(min=0)


def grid_coordinate(potential,base,vector,labels,theta,low,high,current,points,device):
    sync(device);start=time.perf_counter()
    params=np.unique(np.r_[np.linspace(low,high,points,dtype=np.float32),np.float32(current)])
    x=torch.as_tensor(potential,device=device)
    b=torch.as_tensor(base,dtype=torch.float64,device=device)
    v=torch.as_tensor(vector,dtype=torch.float64,device=device)
    y=torch.as_tensor(labels,device=device)
    values=[]
    for first in range(0,len(params),8):
        p=torch.as_tensor(params[first:first+8],device=device)
        counts=(((x[None]+p[:,None,None])-theta)>=0).sum(-1)
        logits=b[None]+counts[:,:,None]*v
        losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.repeat(len(p)),reduction='none')
        values.extend(losses.reshape(len(p),-1).sum(1).cpu().tolist())
    best=int(np.argmin(values));i=int(np.flatnonzero(params==np.float32(current))[0])
    chosen=float(params[best]) if values[best]<values[i]-1e-10 else float(current)
    sync(device)
    return {'bias':chosen,'loss_sum':min(values[best],values[i]),'seconds':time.perf_counter()-start,
            'candidates':len(params),'image_loss_evaluations':len(params)*len(labels)}
