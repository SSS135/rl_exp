import torch
import torch.autograd


class GradBoundsLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, x_low: float, x_high: float):
        ctx.save_for_backward(x)
        ctx.x_low = x_low
        ctx.x_high = x_high
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        x_low = ctx.x_low
        x_high = ctx.x_high
        grad_output = grad_output.clone()
        grad_output[(x > x_high) & (grad_output < 0)] = 0
        grad_output[(x < x_low) & (grad_output > 0)] = 0
        return grad_output, None, None


def limit_bounds_by_grad(x: torch.Tensor, x_low: float, x_high: float, inplace=False):
    return GradBoundsLimit.apply(x if inplace else x.clone(), x_low, x_high)
