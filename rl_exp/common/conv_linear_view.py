from torch import nn as nn


class ConvView(nn.Module):
    def __init__(self, conv_shape=None):
        super().__init__()
        self.conv_shape = conv_shape
        assert conv_shape is None or len(conv_shape) == 3 or len(conv_shape) == 4

    def forward(self, input):
        # assert input.dim() == 2
        if self.conv_shape is None:
            shape = *input.shape, 1, 1
        elif len(self.conv_shape) == 3:
            shape = input.shape[0], *self.conv_shape
        else:
            shape = self.conv_shape
        return input.view(shape)


class LinearView(nn.Module):
    def forward(self, input):
        # assert input.dim() == 4
        return input.view(input.shape[0], -1)