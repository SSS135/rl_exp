from typing import Tuple

import torch
import torch.jit
import torch.nn.functional as F


@torch.jit.script
def scale_mat_2d(scale: torch.Tensor) -> torch.Tensor:
    assert scale.dim() == 2 and scale.shape[1] == 2
    mat = torch.eye(3, device=scale.device, dtype=scale.dtype).unsqueeze(0).repeat(scale.shape[0], 1, 1)
    mat[:, 0, 0] = scale[:, 0]
    mat[:, 1, 1] = scale[:, 1]
    return mat


@torch.jit.script
def rotate_mat_2d(angle: torch.Tensor) -> torch.Tensor:
    assert angle.dim() == 1
    mat = torch.eye(3, device=angle.device, dtype=angle.dtype).unsqueeze(0).repeat(angle.shape[0], 1, 1)
    mat[:, 0, 0] = angle.cos()
    mat[:, 0, 1] = -angle.sin()
    mat[:, 1, 0] = angle.sin()
    mat[:, 1, 1] = angle.cos()
    return mat


@torch.jit.script
def translate_mat_2d(trans: torch.Tensor) -> torch.Tensor:
    assert trans.dim() == 2 and trans.shape[1] == 2
    mat = torch.eye(3, device=trans.device, dtype=trans.dtype).unsqueeze(0).repeat(trans.shape[0], 1, 1)
    mat[:, 0, 2] = trans[:, 0]
    mat[:, 1, 2] = trans[:, 1]
    return mat


@torch.jit.script
def trs_matrix_2d(trans: torch.Tensor, angle: torch.Tensor, scale: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
    """
    :param trans: [b, (x, y)]
    :param angle: [b]
    :param scale: [b, (x, y)]
    :param input_size: tuple(w, h)
    :return: [b, 3, 3]
    """
    B = trans.shape[0]
    assert trans.shape == (B, 2)
    assert angle.shape == (B,)
    assert scale.shape == (B, 2)
    assert len(input_size) == 2
    input_size = torch.tensor([[input_size[1] / input_size[0], 1.0]], dtype=scale.dtype, device=scale.device)
    mat = translate_mat_2d(trans) @ scale_mat_2d(input_size) @ rotate_mat_2d(angle) @ scale_mat_2d(scale / input_size)
    return mat


@torch.jit.script
def crop_images(images, affine, output_size: Tuple[int, int]):
    grid = F.affine_grid(affine[:, :2], (images.shape[0], images.shape[1], output_size[1], output_size[0]), align_corners=False)
    crop = F.grid_sample(images, grid, align_corners=False)
    return crop


@torch.jit.script
def random_crop(images, crop_fraction: float = 0.9):
    im_size = images.shape[-2], images.shape[-1]
    tr = (torch.rand((images.shape[0], 2), device=images.device) * 2 - 1) * (1 - crop_fraction)
    angle = torch.zeros(images.shape[0], device=images.device)
    scale = torch.full((images.shape[0], 2), crop_fraction, device=images.device)
    trs = trs_matrix_2d(tr, angle, scale, im_size)
    return crop_images(images, trs, im_size)

