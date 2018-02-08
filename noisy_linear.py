import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, noise_std=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.register_buffer('noise_w', torch.zeros(self.weight.size()))
        self.noise_w_std = nn.Parameter(noise_std * torch.ones(self.weight.size()))
        if bias:
            self.register_buffer('noise_b', torch.zeros(self.bias.size()))
            self.noise_b_std = nn.Parameter(noise_std * torch.ones(self.bias.size()))
        else:
            self.register_parameter('noise_b_std', None)
            self.register_buffer('noise_b', None)

        self.reset_noise()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_noise(self):
        self.noise_w.normal_()
        if self.bias is not None:
            self.noise_b.normal_()

    def forward(self, input):
        weight = self.weight + self.noise_w_std * Variable(self.noise_w)
        bias = self.bias + self.noise_b_std * Variable(self.noise_b) if self.bias is not None else None
        return F.linear(input, weight, bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'