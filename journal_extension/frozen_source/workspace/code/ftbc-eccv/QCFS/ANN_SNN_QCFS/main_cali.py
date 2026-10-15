import argparse
import os
import copy
import random
import numpy as np
import sys

import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer import *


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
                #--------------------------------------------------------------------------------------------#


def main():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # just use default setting
    parser.add_argument('-j','--workers',default=16, type=int,metavar='N',help='number of data loading workers')
    parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
    parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
    parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

    # model configuration
    parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset: cifa10, cifar100, imagenet')
    parser.add_argument('-arch', '--model',default='vgg16',type=str,help='model: resnet34')
    parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

    # test configuration
    parser.add_argument('-dev','--device',default='0',type=str,help='device')
    parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
    parser.add_argument('-L', '--L', default=8, type=int, help='Step L')
    args = parser.parse_args()

    print(args)

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size, dist_sample=False)

    # preparing model
    model = modelpool(args.model, args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))

    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2]+'thresh'] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)

    ann = copy.deepcopy(model)

    # acc = val(model=ann, test_loader=test_loader, T=0, device=device) # , sample_iter=10
    # print(f"ANN accuracy: {acc}")

    #--------------------------------------------------------------------------#
    ### SNN version in QCFS

    model.set_T(args.time)
    model.set_L(args.L)

    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)

    ### get QCFS SNN model accuracy:
    # snn_acc = val(model, test_loader, T=args.time, device=device)
    # print(f"SNN accuracy: {snn_acc}, T: {args.time}, L: {args.L}")

    #--------------------------------------------------------------------------#

    ### Our calibration method
    bias_corr_model(
        ann=ann, # ann
        snn=model, # snn
        T=args.time,
        train_loader=train_loader,
        curr_t_alpha=0.4,
        num_cali_sample_batches=3,
    )

    cali_snn_acc = val(model, test_loader, T=args.time, device=device)
    print(f"Calibrated SNN accuracy: {cali_snn_acc}, T: {args.time}, L: {args.L}")


if __name__ == "__main__":
    main()
