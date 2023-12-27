import torch 
import torch.nn as nn
import torch.nn.functional as F 

from typing import Optional
import numpy as np

import pdb 

def weighted_loss(loss_fn, pred_y, true_y, weights):
    if not isinstance(weights, (tuple, list)):
        weights = [weights]

    err = loss_fn(pred_y, true_y, reduction='none')

    y_ndim = err.ndim
    prod_weight = torch.ones_like(err)
    for j, weight in enumerate(weights):
        m_ndim = weight.ndim
        assert m_ndim <= y_ndim
        for i in range(m_ndim, y_ndim):
            weight = weight.unsqueeze(i)
        prod_weight *= weight

    loss = (prod_weight * err).sum() / (prod_weight.sum() + 1.e-12)
    return loss

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    modified from sigmoid_focal_loss: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

    inputs and targets: same as F.cross_entropy
    weight: class weight as in F.cross_entropy
    """
    def cross_entropy(x, y):
        """
        pytorch 1.7.1 requires the 'targets' in F.cross_entropy() to be class indices.
        This custom implementation supports 'targets' as probability distribution.

        x: (B, C, ...)
        y: (B, C, ...)
        """
        assert y.dtype == torch.float32
        assert x.shape == y.shape
        return -y * F.log_softmax(x, dim=1)   # (B, C, ...)

    if targets.dtype in [torch.int32, torch.long]:
        target = F.one_hot(targets).permute(0, 3, 1, 2).float()    # categorical to one-hot probability
    p = torch.softmax(inputs, dim=1)    # (B, C, ...)
    ce_loss = cross_entropy(inputs, targets)    # (B, C, ...)
    p_t = p * targets + (1 - p) * (1 - targets)     # (B, C, ...)
    loss = ce_loss * ((1 - p_t) ** gamma)   # (B, C, ...)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    num_classes = targets.shape[1]
    if weight is None:
        weight = torch.ones_like(loss)
    else:
        assert weight.shape == torch.Size([num_classes])
        for _ in range(loss.ndim - 2):
            weight = weight.unsqueeze(-1)
        weight = weight.unsqueeze(0).expand_as(loss)

    loss = torch.sum(loss * weight, dim=1)   # (B, ...)

    if reduction == "mean":
        loss = loss.sum() / (weight.sum() + 1.e-8)
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def appx_l1_loss(predict, alpha, reduction='mean', eps=1.e-12):
    loss = torch.sqrt(torch.square(predict - alpha) + 1.e-12)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

## Laplacian loss is refer to 
## https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
def build_gaussian_kernel(size=5, sigma=1.0, n_channels=1):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    kernel = np.tile(kernel, (n_channels, 1, 1))
    return kernel[:, np.newaxis]

def conv_gaussian(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gaussian(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyr.append(current)
    return pyr

def laplacian_loss(predict, alpha, weight):
    gauss_kernel = build_gaussian_kernel(size=5, sigma=1.0, n_channels=1)
    gauss_kernel = torch.as_tensor(gauss_kernel, dtype=torch.float32, device=predict.device)
    pyr_alpha  = laplacian_pyramid(alpha, gauss_kernel, 5)
    pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
    pyr_weight = [weight] + [F.interpolate(weight, x.shape[2:], mode='bilinear', align_corners=False) for x in pyr_alpha[1:]]
    laplacian_loss_weighted = sum([weighted_loss(F.l1_loss, b, a, w)  for a, b, w in zip(pyr_alpha, pyr_predict, pyr_weight)])
    return laplacian_loss_weighted