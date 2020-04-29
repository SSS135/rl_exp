import math

import torch


def mse_loss(pred, target):
    return (pred - target) ** 2


def huber_loss(pred, target):
    x = (pred - target).abs()
    return 0.5 * x.clamp(max=1) ** 2 + x.clamp(min=1) - 1


def phuber_loss(pred, target, pow=0.5):
    x = (pred - target).abs()
    return 0.5 * x.clamp(max=1) ** 2 + (1 / pow) * x.clamp(min=1) ** pow - (1 / pow)


def loghuber_loss(pred, target):
    x = (pred - target).abs()
    return 0.5 * x.clamp(max=1) ** 2 + torch.log(x.clamp(min=1))


def plog_loss(pred, target, a=4):
    diff = pred - target
    return 0.5 * (a + 1) * torch.log(diff * diff + a) - 0.5 * (a + 1) * math.log(a)


LOSSES = dict(huber=huber_loss, mse=mse_loss, phuber=phuber_loss, loghuber=loghuber_loss, plog=plog_loss)