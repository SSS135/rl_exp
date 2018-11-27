# https://github.com/salesforce/pytorch-qrnn
from typing import Optional, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchqrnn.forget_mult import ForgetMult


class LinearTempNormLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: Optional[int]=None,
                 use_cuda: bool=True, activation: Callable=torch.tanh, log_forget_range: Tuple[float, float]=(-3, -6)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.use_cuda = use_cuda
        self.activation = activation
        self.register_buffer('forget_gate', torch.exp(torch.rand(hidden_size) * (log_forget_range[1] - log_forget_range[0]) + log_forget_range[0]))
        self.forget_gate.requires_grad = False

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.input_size,
                                self.hidden_size,
                                bias=True)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor]=None, reset_flags: Optional[torch.Tensor]=None):
        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size

        assert input_size == self.input_size
        assert hidden is None or hidden.shape == (batch_size, hidden_size * 2)
        assert reset_flags is None or reset_flags.shape == (seq_len, batch_size)

        if hidden is None:
            mu, var = x.new_zeros((batch_size, hidden_size)), x.new_ones((batch_size, hidden_size))
        else:
            mu, var = hidden.chunk(2, -1)

        y = self.linear(x)

        assert y.shape == (seq_len, batch_size, hidden_size)

        forget = self.forget_gate.repeat((seq_len, batch_size, 1))
        if reset_flags is not None:
            reset_flags = reset_flags.unsqueeze(-1)
            retain_flags = 1 - reset_flags
            forget = 1 - (1 - forget) * retain_flags

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * forget may be useful
        mu_new = ForgetMult()(forget, y, mu, use_cuda=self.use_cuda)
        mu_calc = torch.cat([mu.unsqueeze(0), mu_new[:-1]], 0)
        diff_sq = (y - mu_calc) ** 2
        if reset_flags is not None:
            diff_sq = diff_sq * retain_flags + reset_flags
        var_new = ForgetMult()(forget, diff_sq, var, use_cuda=self.use_cuda)

        # print('mu', mu)
        # print('std', var ** 0.5)
        # print('mu_new', mu_new)
        # print('diff', diff_sq ** 0.5)
        # print('std_new', var_new ** 0.5)

        assert mu_new.shape == y.shape
        assert var_new.shape == y.shape

        std_calc = (torch.cat([var.unsqueeze(0), var_new[:-1]], 0) + 1e-4) ** 0.5
        y = (y - mu_calc) / std_calc
        y = self.activation(y)

        new_hidden = torch.cat([mu_new[-1], var_new[-1]], -1)

        assert y.shape == (seq_len, batch_size, hidden_size)
        assert new_hidden.shape == (batch_size, hidden_size * 2)

        return y, new_hidden


class LinearTempNorm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int=1, bias: bool=True, batch_first: bool=False, **kwargs):
        assert not batch_first, 'Batch first mode is not yet supported'
        assert bias, 'Removing underlying bias is not yet supported'

        super().__init__()

        self.layers = nn.ModuleList([LinearTempNormLayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.num_layers = num_layers

    def forward(self, input: torch.Tensor, hidden: Optional[torch.Tensor]=None, reset_flags: Optional[torch.Tensor]=None):
        seq_len, batch_size, input_size = input.shape
        assert hidden is None or hidden.shape == (self.num_layers, batch_size, self.hidden_size * 2)

        next_hidden = []

        for i, layer in enumerate(self.layers):
            input, hn = layer(input, None if hidden is None else hidden[i], reset_flags)
            next_hidden.append(hn)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])

        return input, next_hidden


# class DenseLinearTempNorm(nn.Module):
#     def __init__(self, input_size, hidden_size,
#                  num_layers=1, bias=True, batch_first=False,
#                  dropout=0, bidirectional=False, layers=None, dense_output=False, **kwargs):
#         assert bidirectional == False, 'Bidirectional QRNN is not yet supported'
#         assert batch_first == False, 'Batch first mode is not yet supported'
#         assert bias == True, 'Removing underlying bias is not yet supported'
#
#         super().__init__()
#
#         self.layers = nn.ModuleList(layers if layers else [
#             LinearTempNormLayer(input_size if l == 0 else input_size + l * hidden_size,
#                       hidden_size, **kwargs)
#             for l in range(num_layers)])
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = len(layers) if layers else num_layers
#         self.bias = bias
#         self.batch_first = batch_first
#         self.dropout = dropout
#         self.bidirectional = bidirectional
#         self.dense_output = dense_output
#
#     def reset(self):
#         r'''If your convolutional window is greater than 1, you must reset at the beginning of each new sequence'''
#         [layer.reset() for layer in self.layers]
#
#     def forward(self, input, hidden=None, reset_flags=None):
#         next_hidden = []
#
#         for i, layer in enumerate(self.layers):
#             new_input, hn = layer(input, None if hidden is None else hidden[i], reset_flags)
#             input = torch.cat([input, new_input], 2)
#             next_hidden.append(hn)
#
#             if self.dropout != 0 and i < len(self.layers) - 1:
#                 input = F.dropout(input, p=self.dropout, training=self.training, inplace=False)
#
#         next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size()[-2:])
#
#         return input if self.dense_output else new_input, next_hidden