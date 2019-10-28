import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, noise_std=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.noise_std = noise_std
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.bias_std = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_std', None)

        self.register_buffer('weight_noise', torch.zeros(self.weight.size()))
        self.register_buffer('bias_noise', torch.zeros(self.bias.size()) if bias else None)

        self.reset_parameters()
        self.randomize_noise()

    def reset_parameters(self):
        with torch.no_grad():
            std = math.sqrt(3 / self.weight.size(1))
            self.weight.uniform_(-std, std)
            self.weight_std.fill_(self.noise_std)
            if self.bias is not None:
                self.bias.zero_()
                self.bias_std.fill_(self.noise_std)

    def randomize_noise(self):
        with torch.no_grad():
            self.weight_noise.normal_()
            if self.bias is not None:
                self.bias_noise.normal_()

    def forward(self, input):
        weight = self.weight + self.weight_std * self.weight_noise
        bias = self.bias + self.bias_std * self.bias_noise if self.bias is not None else None
        return F.linear(input, weight, bias)

    def __repr__(self):
        return f'{self.__class__.__name__}(' \
            f'in_features={self.in_features}, ' \
            f'out_features={self.out_features}, ' \
            f'bias={self.bias is not None})'