# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, num_channel, apply_lens_factor=False):
        super(ChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channel // 2, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        self.apply1 = ApplyVectorWeights() if apply_lens_factor else IdentityMod()

    def forward(self, x: torch.Tensor, lens_factor: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.apply1(x, weights=lens_factor)


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_channel, apply_lens_factor=False):
        super(SimplifiedChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.apply1 = ApplyVectorWeights() if apply_lens_factor else IdentityMod()

    def forward(self, x: torch.Tensor, lens_factor) -> torch.Tensor:
        x = self.model(x)
        return self.apply1(x=x, weights=lens_factor)


class ApplyVectorWeights(nn.Module):
    def __init__(self):
        super(ApplyVectorWeights, self).__init__()

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return x * weights


class SkipConnection(nn.Module):
    def __init__(self,):
        super(SkipConnection, self).__init__()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return x + skip


class IdentityMod(nn.Module):
    def __init__(self):
        super(IdentityMod, self).__init__()

    def forward(self, x=None, *args, **kwargs) -> torch.Tensor:
        return x


def get_activation(activation_type=None):
    if activation_type == "GELU":
        return nn.GELU()
    elif activation_type == 'Simple_Gate':
        return SimpleGate()
    else:
        raise NotImplementedError(f'Activation type {activation_type} is not implemented.')


def get_attention(attention_type=None):
    if attention_type == 'CA':
        return ChannelAttention
    elif attention_type == 'SCA':
        return SimplifiedChannelAttention
    else:
        raise NotImplementedError(f'Attention type {attention_type} is not implemented.')
