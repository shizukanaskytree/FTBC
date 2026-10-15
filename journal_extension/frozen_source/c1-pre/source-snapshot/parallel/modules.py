import torch
from torch import nn
from torch.autograd import Function
import random


def _consume_runtime_output_bias_correction(module, potential, mean_input=None):
    learned = getattr(module, "learned_calibration", None)
    if learned is not None:
        potential = potential + learned(mean_input).unsqueeze(0)
    correction = getattr(module, "runtime_output_bias_correction", None)
    if correction is None:
        return potential
    module.runtime_output_bias_correction = None
    if not isinstance(correction, torch.Tensor):
        return potential
    correction = correction.to(device=potential.device, dtype=potential.dtype)
    if tuple(correction.shape) != tuple(potential.shape):
        try:
            correction = correction.expand_as(potential)
        except RuntimeError:
            return potential
    return potential + correction


class IFNeuron(nn.Module):
    def __init__(self, T, th=1., init_mem=0.5):
        super(IFNeuron, self).__init__()
        self.T = T
        self.t = 0
        if isinstance(th, torch.Tensor):
            self.register_buffer('v_threshold', th)
            self.register_buffer('v', init_mem * th)
        else:
            self.v_threshold = th
            self.v = init_mem * th
        self.init_mem = init_mem

    def forward(self, x):
        self.t += 1
        self.v = self.v + x
        spike = (self.v >= self.v_threshold).float() * self.v_threshold
        self.v = self.v - spike
        if self.t == self.T:
            self.reset()
        return spike

    def reset(self):
        self.v = self.init_mem * self.v_threshold
        self.t = 0


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ParaInfNeuron(nn.Module):
    def __init__(self, T, th=1., init_mem=0.5, dim=5):
        super(ParaInfNeuron, self).__init__()
        self.T = T
        self.v_threshold = th
        self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1))
        self.register_buffer('bias', (init_mem * th) / torch.arange(1, T+1).unsqueeze(-1))
        self.dim = dim
        # FTBC uses a lazily-shaped temporal bias map with shape [T, 1, ...].
        self.ftbc_correction_granularity = 'tc'
        self.register_buffer('output_bias_correction', torch.zeros(0))

    def forward(self, x):
        # x.shape: [TxB, C, H, W] or [TxB, N]
        x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])
        if self.dim == 5:
            T, B, C, H, W = x.size()
            # 计算电位：与 ParaInfNeuron_CW_ND 保持一致的广播风格
            mean_input = x.mean(dim=0)
            potential = self.TxT.view(T, 1, 1, 1, 1) * mean_input + self.bias.view(T, 1, 1, 1, 1)  # [T,B,C,H,W]

            # #-------------------------------------------------------------------
            # # 若仅微调 output_bias_correction（其余参数均冻结），切断对上游激活的梯度，
            # # 避免在整个网络和 T 维度上构建巨大的计算图，显著降低显存与加速反向。
            # bias_trainable = isinstance(self.output_bias_correction, torch.nn.Parameter) and self.output_bias_correction.requires_grad
            # other_trainable = any(p.requires_grad for n, p in self.named_parameters() if n != 'output_bias_correction')
            # if bias_trainable and not other_trainable:
            #     potential = potential.detach()
            # #-------------------------------------------------------------------

            if getattr(self, 'ftbc_correction_granularity', 'tc') == 'thwc':
                target_shape = (T, 1, C, H, W)
            else:
                target_shape = (T, 1, C, 1, 1)
            if (self.output_bias_correction.numel() == 0) or (tuple(self.output_bias_correction.shape) != target_shape):
                new_corr = torch.zeros(target_shape, device=potential.device, dtype=potential.dtype)
                if isinstance(self.output_bias_correction, torch.nn.Parameter):
                    with torch.no_grad():
                        self.output_bias_correction.set_(new_corr)
                else:
                    self.register_buffer('output_bias_correction', new_corr)
            else:
                # 设备/精度对齐
                if self.output_bias_correction.device != potential.device or self.output_bias_correction.dtype != potential.dtype:
                    if isinstance(self.output_bias_correction, torch.nn.Parameter):
                        with torch.no_grad():
                            self.output_bias_correction.set_(self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))
                    else:
                        self.register_buffer('output_bias_correction', self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))

            potential = potential + self.output_bias_correction  # FTBC temporal bias map
            potential = _consume_runtime_output_bias_correction(self, potential, mean_input)

            # 使用 Heaviside 近似，保证反向传播可微
            out = Heaviside.apply(potential - self.v_threshold, 1.0, "triangle") * self.v_threshold
            return out.view(-1, C, H, W)
        else:
            # 全连接/向量情形
            T, B, N = x.size()
            mean_input = x.mean(dim=0)
            potential = self.TxT.view(T, 1, 1) * mean_input + self.bias.view(T, 1, 1)  # [T,B,N]

            # #-------------------------------------------------------------------
            # # 仅 bias 微调场景下切断对上游的梯度
            # bias_trainable = isinstance(self.output_bias_correction, torch.nn.Parameter) and self.output_bias_correction.requires_grad
            # other_trainable = any(p.requires_grad for n, p in self.named_parameters() if n != 'output_bias_correction')
            # if bias_trainable and not other_trainable:
            #     potential = potential.detach()
            # #-------------------------------------------------------------------

            target_shape = (T, 1, N)
            if (self.output_bias_correction.numel() == 0) or (tuple(self.output_bias_correction.shape) != target_shape):
                new_corr = torch.zeros(target_shape, device=potential.device, dtype=potential.dtype)
                if isinstance(self.output_bias_correction, torch.nn.Parameter):
                    with torch.no_grad():
                        self.output_bias_correction.set_(new_corr)
                else:
                    self.register_buffer('output_bias_correction', new_corr)
            else:
                if self.output_bias_correction.device != potential.device or self.output_bias_correction.dtype != potential.dtype:
                    if isinstance(self.output_bias_correction, torch.nn.Parameter):
                        with torch.no_grad():
                            self.output_bias_correction.set_(self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))
                    else:
                        self.register_buffer('output_bias_correction', self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))

            potential = potential + self.output_bias_correction  # FTBC temporal bias map
            potential = _consume_runtime_output_bias_correction(self, potential, mean_input)
            out = Heaviside.apply(potential - self.v_threshold, 1.0, "triangle") * self.v_threshold
            return out.view(-1, N)

    # 使 output_bias_correction 可转为可训练参数，以便微调
    def enable_output_bias_finetune(self):
        if isinstance(self.output_bias_correction, torch.nn.Parameter):
            self.output_bias_correction.requires_grad = True
            return

        buf = self.output_bias_correction
        if hasattr(self, '_buffers') and 'output_bias_correction' in self._buffers:
            self._buffers.pop('output_bias_correction')

        if isinstance(buf, torch.Tensor) and buf.numel() > 0:
            param = nn.Parameter(buf.detach().clone(), requires_grad=True)
        else:
            device = buf.device if isinstance(buf, torch.Tensor) else 'cpu'
            dtype = buf.dtype if isinstance(buf, torch.Tensor) else torch.float32
            dummy = torch.zeros(1, device=device, dtype=dtype)
            param = nn.Parameter(dummy, requires_grad=True)

        self.register_parameter('output_bias_correction', param)

# class ParaInfNeuron_CW_ND(nn.Module):
#     def __init__(self, T, pre_th, post_bias, bias):
#         super(ParaInfNeuron_CW_ND, self).__init__()
#         self.T = T
#         self.register_buffer('pre_th', pre_th) # shape: [C, 1, 1]
#         self.register_buffer('post_th', pre_th + post_bias) # shape: [C, 1, 1]
#         self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, 1, 1, 1]
#         self.register_buffer('bias', bias / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, C, 1, 1]

#     def forward(self, x):
#         # x.shape: [TxB, C, H, W]

#         x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])
#         T, B, C, H, W = x.size()
#         return (((self.TxT * x.mean(dim=0) + self.bias) >= self.pre_th).float() * self.post_th).view(-1, C, H, W)


class ParaInfNeuron_CW_ND(nn.Module):
    def __init__(self, T, pre_th, post_bias, bias):
        super(ParaInfNeuron_CW_ND, self).__init__()
        self.T = T
        self.register_buffer('pre_th', pre_th) # shape: [C, 1, 1]
        self.register_buffer('post_th', pre_th + post_bias) # shape: [C, 1, 1]
        self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, 1, 1, 1]
        self.register_buffer('bias', bias / torch.arange(1, T+1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # shape: [T, 1, C, 1, 1]
        # FTBC uses a lazily-shaped temporal bias map with shape [T, 1, C, 1, 1].
        self.ftbc_correction_granularity = 'tc'
        self.register_buffer('output_bias_correction', torch.zeros(0))

    def forward(self, x):
        # x.shape: [TxB, C, H, W]
        x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:]) # [T, B, C, H, W]
        T, B, C, H, W = x.size()

        # Potential domain: include input-side correction to shift thresholding boundary
        mean_input = x.mean(dim=0)
        potential = self.TxT * mean_input + self.bias  # [T,1,C,1,1]

        # #-----------------------------------------------------------------------
        # # 若当前层仅训练 output_bias_correction，将 potential 与上游图断开，
        # # 避免跨层/跨时刻的梯度传播导致显存爆炸与训练极慢。
        # bias_trainable = isinstance(self.output_bias_correction, torch.nn.Parameter) and self.output_bias_correction.requires_grad
        # other_trainable = any(p.requires_grad for n, p in self.named_parameters() if n != 'output_bias_correction')
        # if bias_trainable and not other_trainable:
        #     potential = potential.detach()
        # #-----------------------------------------------------------------------

        if getattr(self, 'ftbc_correction_granularity', 'tc') == 'thwc':
            target_shape = (T, 1, C, H, W)
        else:
            target_shape = (T, 1, C, 1, 1)
        if (self.output_bias_correction.numel() == 0) or (tuple(self.output_bias_correction.shape) != target_shape):
            new_corr = torch.zeros(target_shape, device=potential.device, dtype=potential.dtype)
            if isinstance(self.output_bias_correction, torch.nn.Parameter):
                with torch.no_grad():
                    self.output_bias_correction.set_(new_corr)
            else:
                self.register_buffer('output_bias_correction', new_corr)
        else:
            # keep device/dtype aligned
            if self.output_bias_correction.device != potential.device or self.output_bias_correction.dtype != potential.dtype:
                if isinstance(self.output_bias_correction, torch.nn.Parameter):
                    with torch.no_grad():
                        self.output_bias_correction.set_(self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))
                else:
                    self.register_buffer('output_bias_correction', self.output_bias_correction.to(device=potential.device, dtype=potential.dtype))

        potential = potential + self.output_bias_correction  # FTBC temporal bias map
        potential = _consume_runtime_output_bias_correction(self, potential, mean_input)

        # Generate spikes at calibrated threshold
        out = Heaviside.apply(potential - self.pre_th, 1.0, "triangle") * self.post_th
        # out = Heaviside.apply(potential - self.pre_th, 8.0, "superspike") * self.post_th  # 使用 superspike surrogate，beta=8
        # out = (Heaviside.apply(potential - self.pre_th) * self.post_th)  # [T, B, C, H, W]
        # out = (potential >= self.pre_th).float() * self.post_th

        return out.view(-1, C, H, W)

    # Enable converting output_bias_correction buffer to a trainable Parameter
    def enable_output_bias_finetune(self):
        # If it's already a Parameter, just ensure it's trainable
        if isinstance(self.output_bias_correction, torch.nn.Parameter):
            self.output_bias_correction.requires_grad = True
            return

        # Fetch current buffer (may be empty for lazy init)
        buf = self.output_bias_correction

        # Remove from buffers dict to avoid name conflict
        if hasattr(self, '_buffers') and 'output_bias_correction' in self._buffers:
            self._buffers.pop('output_bias_correction')

        # Create Parameter - if buffer is empty, trigger forward pass to initialize it properly
        if isinstance(buf, torch.Tensor) and buf.numel() > 0:
            # Buffer already has proper shape, convert to Parameter
            param = nn.Parameter(buf.detach().clone(), requires_grad=True)
        else:
            # Buffer is empty - create a dummy one that will be replaced on first forward
            # Use a small non-zero tensor to ensure numel() > 0 for parameter counting
            device = buf.device if isinstance(buf, torch.Tensor) else 'cpu'
            dtype = buf.dtype if isinstance(buf, torch.Tensor) else torch.float32
            dummy = torch.zeros(1, device=device, dtype=dtype)  # Temporary placeholder
            param = nn.Parameter(dummy, requires_grad=True)

        # Register as parameter
        self.register_parameter('output_bias_correction', param)

# ------------------------------------------------







# ++++++++++++++++++++++++++++++++++++++++++++++++

### T-shuffle

# class ParaInfNeuron(nn.Module):
#     def __init__(self, T, th=1., init_mem=0.5, dim=5):
#         super(ParaInfNeuron, self).__init__()
#         self.T = T
#         self.v_threshold = th
#         self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1))
#         self.register_buffer('bias', (init_mem * th) / torch.arange(1, T+1).unsqueeze(-1))
#         self.dim = dim

#     def forward(self, x):
#         # x.shape: [TxB, C, H, W] or [TxB, N]

#         x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])

#         ### shuffle x with repect to the T dim
#         # perm = torch.randperm(x.size(0), device='cpu')
#         # x = x[perm.to(x.device)]

#         if self.dim == 5:
#             T, B, C, H, W = x.size()
#             out = (((self.TxT * x.view(T, -1).mean(dim=0) + self.bias) >= self.v_threshold).float() * self.v_threshold).view(T, B, C, H, W)
#             perm = torch.randperm(T, device=out.device)
#             # out = out[perm]
#             out = out.contiguous()
#             out = torch.index_select(out, 0, perm)
#             out = out.view(-1, C, H, W)
#             return out
#         else:
#             T, B, N = x.size()
#             out = (((self.TxT * x.view(T, -1).mean(dim=0) + self.bias) >= self.v_threshold).float() * self.v_threshold).view(T, B, N)
#             perm = torch.randperm(T, device=out.device)
#             # out = out[perm]
#             out = out.contiguous()
#             out = torch.index_select(out, 0, perm)
#             out = out.view(-1, N)
#             return out


# class ParaInfNeuronWithShuffle(ParaInfNeuron):
#     pass


# class ParaInfNeuronNoShuffle(nn.Module):
#     def __init__(self, T, th=1., init_mem=0.5, dim=5):
#         super(ParaInfNeuronNoShuffle, self).__init__()
#         self.T = T
#         self.v_threshold = th
#         self.register_buffer('TxT', T / torch.arange(1, T+1).unsqueeze(-1))
#         self.register_buffer('bias', (init_mem * th) / torch.arange(1, T+1).unsqueeze(-1))
#         self.dim = dim

#     def forward(self, x):
#         # x.shape: [TxB, C, H, W] or [TxB, N]
#         x = x.view((self.T, int(x.shape[0] / self.T)) + x.shape[1:])

#         if self.dim == 5:

######################################################


class FloorLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

qcfs = FloorLayer.apply

class QCFS(nn.Module):
    ### ResNet-34, t=8
    ###resnet_qcfs 8, vgg16_qcfs 4 or 16.
    def __init__(self, in_ch=1, up=8., t=4, dim=5, is_cab=False, is_relu=False):
        super().__init__()
        if isinstance(up, torch.Tensor):
            self.register_buffer('up', up)
        else:
            self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.in_ch = in_ch
        self.dim = dim
        self.is_cab = is_cab and (dim>3)
        self.is_relu = is_relu
        self.cab_inf = False
        if self.is_cab is True:
            self.register_buffer('rec_in_mean', torch.zeros(self.in_ch, 1, 1))
            self.register_buffer('rec_th_mean', torch.zeros(self.in_ch, 1, 1))
            self.is_cab = False

    def forward(self, x): # inf: inference mode, cab: calibration mode
        # This is **stage 2** during execution: Inference mode with calibrated bounds
        # Uses pre-calculated self.rec_in_mean and self.rec_th_mean
        # Single forward pass with calibrated scaling;
        if self.cab_inf is True:
            return torch.clamp(qcfs((x+self.rec_in_mean)*self.t/self.up+0.5)/self.t,0,1)*(self.up+self.rec_th_mean)
        # This is **stage1** during execution: Calibration mode
        # Splits input into 2 parts for error calculation
        # Updates self.rec_in_mean and self.rec_th_mean during forward pass
        # Uses different scaling factors (8 for ResNet, 4/16 for VGG16)
        elif self.is_cab is True:
            x = x.view((2, x.shape[0]//2) + x.shape[1:])
            err = (x[1] - x[0]).transpose(0, 1).flatten(1)
            alpha = 0.99 # 0.99: original
            self.rec_in_mean = alpha * self.rec_in_mean + (1-alpha) * err.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

            x[0] = torch.clamp(qcfs((x[0]+self.rec_in_mean)*self.t/self.up+0.5)/self.t,0,1)*(self.up+self.rec_th_mean)

            ##########################################
            ### resnet_qcfs 8, vgg16_qcfs 4 or 16.
            ### VGG16 on CIFAR100: L=4
            L = 8
            ##########################################
            x[1] = torch.clamp(qcfs(x[1]*L/self.up+0.5)/L,0,1)*self.up if self.is_relu is False else torch.clamp(x[1], torch.zeros_like(self.up), self.up)
            ### the above is the relu ANN or qcfs ANN.

            err = (x[1] - x[0]).transpose(0, 1).flatten(1)
            beta = 0.99 # 0.99: original
            self.rec_th_mean = beta * self.rec_th_mean + (1-beta) * err.mean(dim=1).unsqueeze(-1).unsqueeze(-1)

            return x.flatten(0, 1)
        # Branch 3: Standard QCFS mode
        # Simple quantized clipping function
        # No calibration or error calculation
        # Basic scaling and clamping operations
        else:
            x = x / self.up
            x = qcfs(x*self.t+0.5) / self.t
            x = torch.clamp(x, 0, 1)
            return x * self.up

    def extra_repr(self) -> str:
        return f"in_ch={self.in_ch}, t={self.t}, dim={self.dim}, is_cab={self.is_cab}, is_relu={self.is_relu}"


class RecReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_up = False

    def forward(self, x):
        max_th = x.transpose(0, 1).flatten(1).max(1)[0].unsqueeze(-1).unsqueeze(-1)
        if self.init_up is False:
            self.init_up = True
            self.register_buffer('up', max_th)
        else:
            self.up = torch.max(self.up, max_th)

        return torch.clamp(x, torch.zeros_like(self.up), self.up)



# class Heaviside(Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return (x >= 0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         # Surrogate gradient
#         temp = (1 - torch.abs(x)).clamp(min=0)
#         grad_input = grad_input * temp
#         return grad_input


class Heaviside(Function):
    @staticmethod
    def forward(ctx, x, beta=10.0, kind: str = "superspike"):
        """
        x: pre-activation (e.g., u - theta)
        beta: slope parameter (float or tensor)
        kind: surrogate type: "sigmoid", "superspike", "arctan", "triangle"
        """
        ctx.save_for_backward(x, torch.tensor(beta))
        ctx.kind = kind
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, beta = ctx.saved_tensors
        b = beta.clamp(min=1e-6)

        if ctx.kind == "sigmoid":
            s = torch.sigmoid(b * x)
            sg = b * s * (1 - s)
        elif ctx.kind == "superspike":
            sg = 1.0 / (1.0 + b * x.abs())**2
        elif ctx.kind == "arctan":
            sg = b / (1.0 + (b * x)**2)
        elif ctx.kind == "triangle":
            sg = (1 - x.abs()).clamp(min=0.)
        else:
            raise ValueError(f"Unknown surrogate kind: {ctx.kind}")

        grad_input = grad_output * sg
        return grad_input, None, None
