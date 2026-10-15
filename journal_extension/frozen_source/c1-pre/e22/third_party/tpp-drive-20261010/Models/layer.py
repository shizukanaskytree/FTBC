# from cv2 import mean
from sympy import print_rcode
import random
import torch
import torch.nn as nn

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(nn.Module):
    def __init__(
            self,
            T=0,
            L=8,
            thresh=8.0,
            tau=1.,
            gama=1.0,
            scaling_factor=1.0):
        """
        T: the number of timesteps, controlled by the function set_T in each model file
        L: controlled by the function set_L in each model file
        scaling_factor: controlled by the function set_scaling_factor in each model file
        """
        super(IF, self).__init__()
        self.act = ZIF.apply
        self.thresh = nn.Parameter(torch.tensor([thresh]), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        ### when init, T==0 since IF() with default T; ONLY after calling `model.set_T(args.time)`
        self.T = T
        self.loss = 0
        ### control how much amount of x0 added to mem at t=1 as the warmup
        self.scaling_factor = scaling_factor

        ### mode can be 'normal', 'prob', 'warmup'
        self.mode = 'normal'


    def forward(self, x):
        if self.T > 0:
            if self.mode == 'normal':
                thre = self.thresh.data
                x = self.expand(x)
                mem = 0.5 * thre
                spike_pot = []
                for t in range(self.T):
                    mem = mem + x[t, ...]
                    spike = self.act(mem - thre, self.gama) * thre
                    mem = mem - spike
                    spike_pot.append(spike)

                x = torch.stack(spike_pot, dim=0)
                x = self.merge(x)

            elif self.mode == 'shuffle':
                thre = self.thresh.data
                x = self.expand(x)
                mem = 0.5 * thre
                spike_pot = []
                for t in range(self.T):
                    mem = mem + x[t, ...]
                    spike = self.act(mem - thre, self.gama) * thre
                    mem = mem - spike
                    spike_pot.append(spike)

                random.shuffle(spike_pot)
                x = torch.stack(spike_pot, dim=0)
                x = self.merge(x)

            elif self.mode == 'prob':
                thre = self.thresh.data
                x = self.expand(x)
                # mem = 0.5 * thre
                z = x.sum(0)
                self.mem_pot = z

                spike_pot = []
                for t in range(self.T):
                    mem_temp = torch.clamp(self.mem_pot / (self.T - t) / thre, 0, 1)
                    spike = torch.bernoulli(mem_temp) * thre
                    self.mem_pot = self.mem_pot - spike
                    spike_pot.append(spike)

                x = torch.stack(spike_pot, dim=0)
                x = self.merge(x)

        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x*self.L+0.5)/self.L
            x = x * self.thresh

        return x



def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
