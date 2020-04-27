import torch.autograd


class InvertingGradients(torch.autograd.Function):
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
        diff = 0.5 * (x_high - x_low)
        dir = grad_output.clone().gt_(0)
        asc = (x_high - x).div_(diff)
        desc = (x - x_low).div_(diff)
        grad_input = asc.lerp_(desc, dir).mul_(grad_output)
        return grad_input, None, None


def inverting_gradients(x: torch.Tensor, x_low: float, x_high: float, inplace=False):
    return InvertingGradients.apply(x if inplace else x.clone(), x_low, x_high)


def test_inverting_gradients():
    torch.manual_seed(123)
    x = torch.tensor([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], requires_grad=True)
    for lim in (1, 2, 10):
        gpos = -(lim - x.detach()) / lim
        gneg = (x.detach() + lim) / lim

        inverting_gradients(x, -lim, lim).sum().backward()
        assert ((x.grad - gneg).abs() > 1e-3).sum().item() == 0, (x.grad, gneg)
        x.grad = None

        inverting_gradients(x, -lim, lim).sum().neg().backward()
        assert ((x.grad - gpos).abs() > 1e-3).sum().item() == 0, (x.grad, gpos)
        x.grad = None