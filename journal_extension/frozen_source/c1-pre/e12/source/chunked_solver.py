"""Memory-only modification of frozen E7 solver: exact integer count accumulation."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'e7/source'))
import solver
full_counts=solver.counts

def chunked_counts(base,delta,theta):
    total=torch.zeros(base.shape[2],dtype=torch.int64,device=base.device)
    for part in base.split(8,dim=1):total+=full_counts(part,delta,theta)
    return total

def install():
    assert solver.counts in (full_counts,chunked_counts)
    solver.counts=chunked_counts
