
import torch
import cv2
import numpy as np


def batch_argmax(tensor):
    '''
    Locate uv of max value in a tensor in dim 2 and dim 3.
    The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).

    Args:
        tensor: torch.tensor

    Returns: uv of max value with batch.

    '''
    # if tensor.shape = (16, 3, 64, 64)
    assert tensor.ndim == 4, 'The tensor.ndim should be 4 (typically tensor.shape = (16, 3, 64, 64)).'
    b, c, h, w = tensor.shape
    tensor = tensor.reshape(b, c, -1)  # flatten the tensor to (16, 3, 4096)
    idx = tensor.argmax(dim=2)
    v = idx // w    # shape (16, 3)
    u = idx - v * w  # shape (16, 3)
    u, v = u.unsqueeze(-1), v.unsqueeze(-1)  # reshape u, v to (16, 3, 1) for cat
    uv = torch.cat((u, v), dim=-1)  # shape (16, 3, 2)
    return uv


def heatmap_to_uv(hm, mode='max'):
    '''
    Locate single keypoint pixel coordinate uv in heatmap.

    Args:
        hm:  shape (bz, c, w, h), dim=4
        mode: -

    Returns: keypoint pixel coordinate uv, shape  (1, 2)

    '''

    assert mode in ('max', 'average')
    if mode == 'max':
        uv = batch_argmax(hm)
    elif mode == 'average':
        b, c, h, w = hm.shape
        hm = hm.reshape(b, c, -1)
        hm = hm / torch.sum(hm, dim=-1, keepdim=True)
        v_map, u_map = torch.meshgrid(torch.arange(h), torch.arange(w))
        u_map = u_map.reshape(1, 1, -1).float().to(hm.device)
        v_map = v_map.reshape(1, 1, -1).float().to(hm.device)
        u = torch.sum(u_map * hm, -1, keepdim=True)
        v = torch.sum(v_map * hm, -1, keepdim=True)
        uv = torch.cat((u, v), dim=-1)
    return uv


def generate_heatmap_2d(uv, heatmap_shape, sigma=7):
    '''
    Generate one heatmap in range (0, 1).
    Args:
        uv: single pixel coordinate, shape (1, 2),
        heatmap_shape: output shape of heatmap, tuple or list, typically: (256, 256)
        sigma:Gaussian sigma

    Returns:heatmap

    '''
    hm = np.zeros(heatmap_shape)
    hm[uv[1], uv[0]] = 1
    hm = cv2.GaussianBlur(hm, (sigma, sigma), 0)
    hm /= hm.max()  # normalize hm to [0, 1]
    return hm # outshape