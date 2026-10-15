"""Unmodified calibration definitions exported for Route C E1.
No training or evaluation entrypoint is executed.
"""
from Models.layer import IF

import sys

import torch

import torch.nn.parallel

import torch.optim

class ActivationSaverHookEachTimeStep:
    def __init__(self):
        self.stored_output = None
        self.stored_input = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch.detach().clone()

        if self.stored_input is None:
            self.stored_input = input_batch[0].detach().clone()

    def reset(self):
        self.stored_output = None
        self.stored_input = None

class GetLayerInputOutput:
    def __init__(self, model, target_module):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHookEachTimeStep()

    @torch.no_grad()
    def __call__(self, input):
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), self.data_saver.stored_output.detach()

def bias_corr_step_by_step(
        ann, module_ann,
        snn, module_snn,
        T,
        train_data: torch.Tensor,
        curr_t_alpha: float = 0.5,
    ):
    ann_get_out = GetLayerInputOutput(ann, module_ann)
    ann_out = ann_get_out(train_data)[1]
    ann_get_out.data_saver.reset()

    snn_get_out = GetLayerInputOutput(snn, module_snn)
    snn_in = snn_get_out(train_data)[0]
    snn_get_out.data_saver.reset()

    snn_in = snn_in.view(T, -1, *snn_in.shape[1:])

    thre = module_snn.thresh.data
    v_t_1 = 0.5 * thre
    spike_pot = []

    for t in range(module_snn.T):
        x_l_1_t = snn_in[t, ...]
        mem_t = v_t_1 + x_l_1_t
        spike = module_snn.act(mem_t - thre, module_snn.gama) * thre

        bias = spike - ann_out
        mem_t_updated = mem_t - bias

        spike_updated = module_snn.act(mem_t_updated - thre, module_snn.gama) * thre
        v_t_updated = mem_t_updated - spike_updated

        if len(bias.shape) == 4:
            bias_mean = bias.mean(dim=[0, 2, 3])
        elif len(bias.shape) == 2:
            bias_mean = bias.mean(dim=0)

        module_snn.time_based_bias[t] = curr_t_alpha * bias_mean + module_snn.time_based_bias[t]
        v_t_1 = v_t_updated

def bias_corr_model(ann, snn, T, train_loader: torch.utils.data.DataLoader, curr_t_alpha=0.5, num_cali_sample_batches=2):
    print(f"==> Start Bias Correction...")
    device = next(ann.parameters()).device

    ann.eval()
    snn.eval()

    for i, (inputs, target) in enumerate(train_loader):
        print(f"calibration using batch_idx: {i}")

        if i >= num_cali_sample_batches:
            break

        sys.stdout.flush()
        inputs = inputs.to(device=device)
        for (name_ann, module_ann), (name_snn, module_snn) in zip(ann.named_modules(), snn.named_modules()):
            assert name_ann == name_snn
            if isinstance(module_snn, IF):
                #--------------------------------------------------------------------------------------------#
                ### our method: time-based bias cali: go to class IF in qcfs_models/layer.py
                bias_corr_step_by_step(
                    ann, module_ann, snn, module_snn, T, inputs, curr_t_alpha=curr_t_alpha)
