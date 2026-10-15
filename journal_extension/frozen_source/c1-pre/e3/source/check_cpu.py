"""Meaningful E3 checks: brute force, ties, original neuron, gradients, reload."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from event_solver import *


def main():
    torch.set_num_threads(4)
    rng=np.random.default_rng(20261009);cases=[]
    for seed in range(24):
        n,m,j=4,32,7
        theta=np.float32([.3,1.,3.7][seed%3])
        x=(rng.normal(1,.7,(n,m))*theta).astype(np.float32)
        if seed%2==0:x[:,::3]=theta
        x[:,0]=0;x[:,1]=-theta
        stream=EventStream(x,theta,-.5*theta,.5*theta)
        base=rng.normal(size=(n,j));v=rng.normal(size=j)/m;labels=rng.integers(0,j,n)
        ref=full_scan(stream,LossOracle(base,v,labels),0.)
        exact=[]
        for p in stream.params:
            counts=(((x+p)-theta)>=0).sum(1)
            assert np.array_equal(counts,stream.counts(p))
            logits=torch.tensor(base+counts[:,None]*v)
            exact.append(float(F.cross_entropy(logits,torch.tensor(labels),reduction='sum')))
        assert np.max(np.abs(np.array(exact)-ref['objectives']))<1e-8
        bound=bounded_search(stream,LossOracle(base,v,labels),0.,10)
        assert abs(bound['loss_sum']-min(exact))<1e-8
        assert bound['lower_bound_sum']<=min(exact)+1e-8
        assert bound['loss_queries']<=stream.support_count
        cases.append({'seed':seed,'states':len(stream.params),'full_queries':ref['loss_queries'],
                      'bound_queries':bound['loss_queries'],'gap':bound['gap_sum']})
    for x in (np.zeros((3,16),np.float32),np.full((3,16),1,np.float32),np.full((3,16),-1,np.float32)):
        s=EventStream(x,1.,-.5,.5);base=np.zeros((3,2));v=np.zeros(2);labels=np.zeros(3,dtype=int)
        a=full_scan(s,LossOracle(base,v,labels),0.)
        b=bounded_search(s,LossOracle(base,v,labels),0.,10)
        assert a['bias']==0. and b['bias']==0. and abs(a['loss_sum']-b['loss_sum'])<1e-8
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'source-snapshot/parallel'))
    from modules import ParaInfNeuron,Heaviside
    neuron=ParaInfNeuron(2,th=1.,init_mem=.5)
    torch.manual_seed(13);x=torch.randn(8,3,2,2)
    neuron.output_bias_correction=torch.zeros(2,1,3,1,1)
    xx=x.reshape(2,4,3,2,2)
    pot=neuron.TxT.reshape(2,1,1,1,1)*xx.mean(0)+neuron.bias.reshape(2,1,1,1,1)
    for value in (-.5,0.,.1,.5):
        neuron.output_bias_correction.fill_(value)
        expected=(((pot+value)-1.)>=0).float().reshape_as(x)
        assert torch.equal(neuron(x),expected)
    edge_checks=0
    for channel in range(3):
        for t in range(2):
            root=critical_bias(pot[t,0,channel,0,0].numpy(),1.)
            for bias,should_fire in ((root,True),(np.nextafter(root,np.float32(-np.inf)),False)):
                neuron.output_bias_correction.zero_()
                neuron.output_bias_correction[:,:,channel,0,0]=float(bias)
                actual=neuron(x).reshape(2,4,3,2,2)[t,0,channel,0,0]
                assert bool(actual!=0)==should_fire
                edge_checks+=1
    neuron.output_bias_correction.fill_(.5)
    a=torch.linspace(-2,2,57,requires_grad=True);b=a.detach().clone().requires_grad_()
    Heaviside.apply(a,1.,'triangle').sum().backward();Spike.apply(b).sum().backward()
    assert torch.equal(a.grad,b.grad)
    saved={k:v.clone() for k,v in neuron.state_dict().items()}
    neuron.output_bias_correction.zero_();neuron.load_state_dict(saved,strict=True)
    assert torch.count_nonzero(neuron.output_bias_correction)==6
    report={'passed':True,'random_cases':cases,'degenerate_cases':3,'original_neuron_cases':4,
            'original_surrogate_gradient_equal':True,'nonzero_bias_strict_reload':True,
            'original_neuron_exact_boundary_checks':edge_checks,
            'cuda_initialized':torch.cuda.is_initialized(),'scope':'CPU math and inherited neuron, no images'}
    path=Path(__file__).resolve().parents[1]/'cache/cpu-checks.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='random_cases'}))


if __name__=='__main__':main()
