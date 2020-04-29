import math

import torch
import torch.nn as nn


def exp_scale(x):
    assert x.dim() == 2 and x.size(1) % 2 == 0
    scale, log_actions = x.chunk(2, dim=1)
    return scale * torch.exp(log_actions)


def log_scale(x, a=0.1):
    return x.sign() * torch.log((x.abs() + a) / a) / -math.log(a)


class LogScale(nn.Module):
    def forward(self, x):
        return log_scale(x)


class ExpScale(nn.Module):
    def forward(self, x):
        return exp_scale(x)
