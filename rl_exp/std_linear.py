import torch
from numpy import random as rng
from torch import nn as nn


class StdLinear(nn.Module):
    def __init__(self, in_features, out_features, std_frac=1/8, bias=True):
        super(StdLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        std_features = int(out_features * std_frac)
        self.linear = nn.Linear(in_features, out_features - std_features, bias=bias)
        self.std_weight = nn.Parameter(torch.Tensor(std_features, in_features))

        if bias:
            self.std_bias = nn.Parameter(torch.Tensor(std_features, in_features))
        else:
            self.register_parameter('std_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

        rand = rng.binomial(1, 0.3, self.std_weight.size())
        rand[rand.sum(-1) == 0] = 1
        rand = rand / rand.mean(-1, keepdims=True)
        self.std_weight.data.copy_(torch.from_numpy(rand))

        if self.std_bias is not None:
            self.std_bias.data.fill_(0)

    def forward(self, input):
        lin = self.linear(input)
        std_w = self.std_weight.detach()
        std_input = (input.view(input.size(0), 1, input.size(1)) + self.std_bias) * std_w
        std = std_input.std(-1)
        return torch.cat([lin, std], -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'