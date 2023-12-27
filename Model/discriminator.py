import torch
from torch import nn
import torch.nn.functional as F

from .layers import SameBlock2d, DownBlock2d

import pdb 


def get_padding_for_same(kernel_size):
    pad1 = kernel_size // 2
    pad0 = pad1 - 1 if kernel_size % 2 == 0 else pad1
    return pad0, pad1

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=None, sigma=None):
        super(GaussianBlur, self).__init__()
        if kernel_size is None and sigma is None:
            raise ValueError("At least one of kernel_size and sigma must be provided")
        elif kernel_size is not None and sigma is not None:
            pass
        elif kernel_size is None and sigma is not None:
            kernel_size = 2 * int(round(sigma * 3)) + 1
        elif kernel_size is not None and sigma is None:
            sigma = kernel_size / 6

        pad0, pad1 = get_padding_for_same(kernel_size)
        self.pad = max(pad0, pad1)

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel.requires_grad_(False)

        self.register_buffer('weight', kernel)
    
    def forward(self, x):
        in_chns = x.shape[1]
        #out = torch.cat(
        #    [
        #        F.conv2d(x[:, i:i+1], self.weight, padding=(self.pad, self.pad)) for i in range(in_chns)      # list of (b, 1, h, w) 
        #    ],
        #    dim=1,
        #)       # (b, in_chns, h, w)
        out = F.conv2d(x, self.weight.repeat(in_chns, 1, 1, 1), padding=(self.pad, self.pad), groups=in_chns)
        return out

class AntiAliasDownsample2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, scale):
        super(AntiAliasDownsample2d, self).__init__()
        assert scale <= 1
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.blur = GaussianBlur(kernel_size, sigma)
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = self.blur(input)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

class Discriminator(nn.Module):
    base_chns = 32
    max_chns = 512

    def __init__(self, logits_layers, dropout, num_blocks=4, shrink_ratio=1):
        super(Discriminator, self).__init__()
        self.dropout = dropout
        self.shrink_ratio = shrink_ratio

        NF = max(int(self.base_chns * self.shrink_ratio), 16)
        chns = [1 + 3 + 1] + [min(NF * (2**i), self.max_chns) for i in range(num_blocks)] 
        layers = [DownBlock2d(chns[i], chns[i+1], ksize=4, padding=2) for i in range(num_blocks)]    # 'num_blocks' downsampling blocks
        self.layers = nn.ModuleList(layers)
        if self.dropout:
            self.drop_rates = list(np.linspace(0, 0.5, num_blocks))

        self.logits_layers = logits_layers
        out_convs = []
        for i in logits_layers:
            assert i < 0, "logits_layer index should be negative, counting backwards."
            out_convs.append(nn.Conv2d(chns[i], 1, kernel_size=4, padding=2))
        self.out_convs = nn.ModuleList(out_convs)

    def forward(self, x, condition):
        """
        x: img
        condition: additional condition inputs; kpt heatmaps are used
        """
        out = torch.cat([x, condition.detach()], dim=1)
        feats = []
        for i, layer in enumerate(self.layers):
            out = layer(out)
            feats.append(out)
            if self.dropout:
                out = F.dropout(out, self.drop_rates[i])
        
        logits = []
        for i, j in enumerate(self.logits_layers):
            out = self.out_convs[i](feats[j])
            logits.append(out)
        return feats, logits

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales, logits_layers, dropout, num_blocks=4, shrink_ratio=1):
        super(MultiScaleDiscriminator, self).__init__()
        if not isinstance(scales, (tuple, list)):
            scales = [scales]
        discs = dict()
        downs = dict()
        for scale in scales:
            assert scale in ['x16', 'x8', 'x4', 'x2', 'x1']
            discs[scale] = Discriminator(logits_layers, dropout, num_blocks, shrink_ratio)
            downs[scale] = AntiAliasDownsample2d(1./float(scale.replace('x', '')))
        self.discs = nn.ModuleDict(discs)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x, condition):
        all_feats, all_logits =[], []
        for scale, disc in self.discs.items():
            downsample = self.downs[scale]
            x_s = downsample(x)
            condition_s = downsample(condition)
            feats_s, logits_s = disc(x_s, condition_s)
            all_feats += feats_s 
            all_logits += logits_s 
        return all_feats, all_logits
    
    def __getitem__(self, k):
        return self.discs[k]

    def __contains__(self, k):
        return k in self.discs.keys()