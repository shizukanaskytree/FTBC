"""Native forward with exactly two surrogate calls replaced; original frozen code untouched."""
import torch
from types import MethodType
from modules import _consume_runtime_output_bias_correction

class Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, mode):
        assert theta > 0 and mode in ('native','relative-area','relative-peak')
        ctx.save_for_backward(x)
        ctx.theta, ctx.mode = theta, mode
        return (x >= 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        if ctx.mode == 'native':
            slope = (1-x.abs()).clamp(min=0.)
        else:
            slope = (1-x.abs()/ctx.theta).clamp(min=0.)
            if ctx.mode == 'relative-area':
                slope = slope/ctx.theta
        return grad*slope, None, None

def install(modules, names, mode):
    for n in names:
        m=modules[n]
        assert type(m).__name__=='ParaInfNeuron'
        m._e10_surrogate_mode=mode
        m.forward=MethodType(forward,m)

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
        out = Surrogate.apply(potential - self.v_threshold, self.v_threshold, self._e10_surrogate_mode) * self.v_threshold
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
        out = Surrogate.apply(potential - self.v_threshold, self.v_threshold, self._e10_surrogate_mode) * self.v_threshold
        return out.view(-1, N)
