import torch
from torch import Tensor
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # enable_internal_sync_shuffle: bool = False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.enable_internal_sync_shuffle = True
        # print(f"BasicBlock: enable_internal_sync_shuffle={self.enable_internal_sync_shuffle}")

    # def forward(self, x: Tensor) -> Tensor:
    #     identity = x

    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu1(out)

    #     out = self.conv2(out)
    #     out = self.bn2(out)

    #     if self.downsample is not None:
    #         identity = self.downsample(x)

    #     out += identity
    #     out = self.relu2(out)

    #     return out

    def forward(self, x: Tensor) -> Tensor:
        # x 的输入形状: [T*B, C_in, H, W]
        # perm = None
        # # --- 1. 在输入处，对 x 进行 Shuffle ---
        # if self.enable_internal_sync_shuffle and hasattr(self.relu1, 'T'):
        #     T = self.relu1.T
        #     B = x.shape[0] // T
        #     C_in, H_in, W_in = x.shape[1:]
        #     x_T = x.view(T, B, C_in, H_in, W_in)
        #     perm = torch.randperm(T, device=x.device)
        #     x = x_T.index_select(0, perm).view(-1, C_in, H_in, W_in)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # # --- 2. 在 relu1 后进行 Shuffle ---
        # if self.enable_internal_sync_shuffle and hasattr(self.relu1, 'T'):
        #     T = self.relu1.T
        #     B = x.shape[0] // T
        #     C_mid, H_mid, W_mid = out.shape[1:]
        #     out_T = out.view(T, B, C_mid, H_mid, W_mid)
        #     # perm = torch.randperm(T, device=x.device)
        #     out = out_T.index_select(0, perm).view(-1, C_mid, H_mid, W_mid)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        # --- 3. 在输出处，进行 Un-shuffle ---
        # if self.enable_internal_sync_shuffle and hasattr(self.relu2, 'T'):
        #     T = self.relu2.T
        #     B = x.shape[0] // T
        #     C_out, H_final, W_final = out.shape[1:]
        #     out_T = out.view(T, B, C_out, H_final, W_final)
        #     perm = torch.randperm(T, device=x.device)
        #     out = out_T.index_select(0, perm).view(-1, C_out, H_final, W_final)

        return out



# class Bottleneck(nn.Module):
#     expansion: int = 4

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#         # 标志位，用于启用此特定shuffle策略
#         enable_crude_shuffle: bool = False
#     ) -> None:
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def _shuffle_tensor(self, tensor_in, T, B):
#         # tensor_in has shape (T*B, C, H, W)

#         # Split into a list of T tensors, each of shape (B, C, H, W)
#         tensor_list = torch.split(tensor_in, B, dim=0)

#         # Generate permutation for T dimension
#         perm = torch.randperm(T, device=tensor_in.device)

#         # Create a new list of tensors in shuffled order
#         shuffled_list = [tensor_list[i] for i in perm]

#         # Concatenate back into a single tensor
#         shuffled_tensor = torch.cat(shuffled_list, dim=0)

#         return shuffled_tensor

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         # --- 残差路径计算 ---
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)

#         # 1. 在 relu1 之后 shuffle
#         if self.enable_crude_shuffle and hasattr(self.relu1, 'T') and self.relu1.T > 1:
#             # print("SHUFFLE")
#             T = self.relu1.T
#             B = x.shape[0] // T
#             out = self._shuffle_tensor(out, T, B)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu2(out)

#         # 2. 在 relu2 之后 shuffle
#         if self.enable_crude_shuffle and hasattr(self.relu2, 'T') and self.relu2.T > 1:
#             T = self.relu2.T
#             B = x.shape[0] // T
#             out = self._shuffle_tensor(out, T, B)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         # --- 快捷路径计算 ---
#         if self.downsample is not None:
#             identity = self.downsample(x)

#         # 3. 在相加前 shuffle 快捷路径
#         if self.enable_crude_shuffle and hasattr(self.relu1, 'T') and self.relu1.T > 1:
#             T = self.relu1.T # 使用与输入一致的T
#             B = x.shape[0] // T
#             identity = self._shuffle_tensor(identity, T, B)

#         # --- 残差连接 ---
#         # 注意: 此时 'out' 和 'identity' 的时序被独立打乱，是不对齐的。
#         out += identity

#         # 最终激活
#         out = self.relu3(out)

#         # 2. 在 relu3 之后 shuffle
#         if self.enable_crude_shuffle and hasattr(self.relu3, 'T') and self.relu3.T > 1:
#             T = self.relu3.T
#             B = x.shape[0] // T
#             out = self._shuffle_tensor(out, T, B)

#         return out




class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out



class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x) # 代码里面其实是不会替换第一个 nn.functional.relu 作为 SNN parallel IF 的, 对吗?  yes. replace_relu_by_func will not replace nn.functional.relu but only on nn.ReLU
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
