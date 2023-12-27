import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import einops
from functools import partial
import pdb

SN = False   # spectral norm
BS = True   # enforce bias regardless of normalization layer
NORM = 'BN'  # batch norm (BN) or instance norm (IN)

ACT_FN = lambda x: F.leaky_relu_(x, 0.1)

ZIG = True

def _make_spectral_norm(layer_class):
    def wrapper(*args, **kwargs):
        layer = layer_class(*args, **kwargs)
        return nn.utils.spectral_norm(layer)
    return wrapper

class LinearBlock(nn.Module):
    def __init__(self, in_chns, out_chns, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(LinearBlock, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Linear = _make_spectral_norm(nn.Conv2d)
        else:
            Linear = nn.Conv2d

        self.linear = Linear(in_chns, out_chns, kernel_size=1, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)
        self.act_fn = act_fn

    def forward(self, x):
        # Since we use conv2d for linear layer, we need to reshape input x to 4D tensor
        lead_shape = x.shape[:-1]
        feat_dim = x.shape[-1]
        x = x.view([-1, feat_dim, 1, 1])

        out = self.linear(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)

        # reshape back to input dimensions
        out = out.view(list(lead_shape) + [-1]) 

        return out

class UpBlock2d(nn.Module):
    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(UpBlock2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        Conv = nn.Conv2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out

class DownBlock2d(nn.Module):
    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN, downsample_method='conv'):
        super(DownBlock2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        Conv = nn.Conv2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)
        
        if downsample_method == 'avg_pool':
            Pool = nn.AvgPool2d
            self.pool = Pool(kernel_size=2)
            conv_stride = 1
        elif downsample_method == 'max_pool':
            Pool = nn.MaxPool2d
            self.pool = Pool(kernel_size=2)
            conv_stride = 1
        elif downsample_method == 'conv':
            conv_stride = 2
        else:
            raise ValueError("Unrecognized downsample_method={}".format(downsample_method))

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, stride=conv_stride, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        if hasattr(self, 'pool'):
            out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(SameBlock2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        Conv = nn.Conv2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        self.conv = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = self.conv(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out

class SameBlockTranspose2d(nn.Module):
    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(SameBlockTranspose2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        ConvTranspose = nn.ConvTranspose2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            ConvTranspose = _make_spectral_norm(ConvTranspose)

        self.deconv = ConvTranspose(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = self.deconv(x)
        if hasattr(self, 'norm'):
            out = self.norm(out)
        out = self.act_fn(out)
        return out


class ResBlock2d(nn.Module):
    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM, spectral_norm=SN):
        super(ResBlock2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        Conv = nn.Conv2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        stride = 2 if downsample else 1
        if stride != 1 or in_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None

        self.conv1 = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=ksize, stride=stride, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        self.conv2 = Conv(in_channels=out_chns, out_channels=out_chns, kernel_size=ksize, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm1 = BatchNorm(in_chns, affine=True, momentum=0.99, eps=0.001)
            self.norm2 = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm1 = InstNorm(in_chns, affine=True)
            self.norm2 = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn

    def forward(self, x):
        out = x
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv1(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.act_fn(out)
        out = self.conv2(out)

        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x)
        out += x

        return out


class ResBottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, in_chns, out_chns, ksize=3, padding=1, act_fn=ACT_FN, downsample=False, norm_fn=NORM, spectral_norm=SN):
        super(ResBottleneck2d, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        Conv = nn.Conv2d 
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(Conv)

        stride = 2 if downsample else 1
        if stride != 1 or in_chns != out_chns:
            self.non_identity_shortcut = Conv(in_channels=in_chns, out_channels=out_chns, kernel_size=1, stride=stride, padding=0)
        else:
            self.non_identity_shortcut = None


        width = out_chns // self.expansion
        self.conv1 = Conv(in_channels=in_chns, out_channels=width, kernel_size=1, padding=0, bias=norm_fn is not None or not norm_fn != 'None' or BS)
        self.conv2 = Conv(in_channels=width, out_channels=width, kernel_size=ksize, stride=stride, padding=padding, bias=norm_fn is None or norm_fn == 'None' or BS)
        self.conv3 = Conv(in_channels=width, out_channels=out_chns, kernel_size=1, padding=0, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm1 = BatchNorm(width, affine=True, momentum=0.99, eps=0.001)
            self.norm2 = BatchNorm(width, affine=True, momentum=0.99, eps=0.001)
            self.norm3 = BatchNorm(out_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm1 = InstNorm(width, affine=True)
            self.norm2 = InstNorm(width, affine=True)
            self.norm3 = InstNorm(out_chns, affine=True)

        self.act_fn = act_fn
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        if hasattr(self, 'norm1'):
            out = self.norm1(out)
        out = self.act_fn(out)
        out = self.conv2(out)
        if hasattr(self, 'norm2'):
            out = self.norm2(out)
        out = self.act_fn(out)
        out = self.conv3(out)
        if hasattr(self, 'norm3'):
            out = self.norm3(out)

        if self.non_identity_shortcut:
            x = self.non_identity_shortcut(x)
        out += x

        return out

class AdaIn(nn.Module):
    def __init__(self, in_chns, out_chns, act_fn=ACT_FN, norm_fn=NORM, spectral_norm=SN):
        super(AdaIn, self).__init__()

        distributed = dist.is_available() and dist.is_initialized()
        BatchNorm = nn.SyncBatchNorm if distributed else nn.BatchNorm2d
        InstNorm = nn.InstanceNorm2d

        if spectral_norm:
            Conv = _make_spectral_norm(nn.Conv2d)
        else:
            Conv = nn.Conv2d

        self.conv = Conv(in_chns, out_chns, kernel_size=1, padding=0, bias=norm_fn is None or norm_fn == 'None' or BS)
        if norm_fn == 'BN':
            self.norm = BatchNorm(in_chns, affine=True, momentum=0.99, eps=0.001)
        elif norm_fn == 'IN':
            self.norm = InstNorm(in_chns, affine=True)
        else:
            raise ValueError("AdaIn must have a normalization layer. Unrecognized normalization layer: {}".format(norm_fn))
        self.act_fn = act_fn

    def forward(self, x, gamma, beta):
        """
        x: (b, c, h, w)
        gamma: (b, c, 1, 1)
        beta: (b, c, 1, 1)
        """
        out = self.norm(x) * (1 + gamma) + beta
        out = self.conv(out)
        out = self.act_fn(out)
        return out


############################### transformer layers
# code from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0., qkv_bias = False):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if ZIG:
            self.qk_head_dim = head_dim
            self.v_head_dim = head_dim

    def forward(self, x, bias=None):
        """
        bias: positional bias (batch, heads, num_tokens, num_tokens)
        """
        if ZIG:
            qkv = self.to_qkv(x).split([self.heads * self.qk_head_dim, self.heads * self.qk_head_dim, self.heads * self.v_head_dim], dim = -1)
        else:
            qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # (b, h, n, n)

        if bias is not None:
            dots = dots + bias

        attn = self.attend(dots)    # (b, h, n, n)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)     # (b, h, n, d)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout = 0., qkv_bias = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, head_dim = head_dim, dropout = dropout, qkv_bias = qkv_bias)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ReductionAttention(nn.Module):
    """
    Only query embeddings at specified locations
    """
    def __init__(self, dim, query_inds, heads = 8, head_dim = 64, dropout = 0., qkv_bias = False):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == dim)

        if isinstance(query_inds, int):
            query_inds = [query_inds]
        self.query_inds = query_inds

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if ZIG:
            self.qk_head_dim = head_dim
            self.v_head_dim = head_dim

    def forward(self, x, bias=None):
        """
        x: (batch, num_tokens, dim)
        bias: positional bias (batch, heads, num_tokens, num_tokens)
        """
        q = self.to_q(x[:, self.query_inds])
        if ZIG:
            kv = self.to_kv(x).split([self.heads * self.qk_head_dim, self.heads * self.v_head_dim], dim = -1)
        else:
            kv = self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q,) + kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if bias is not None:
            dots = dots + bias[:, :, self.query_inds]

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerV2(nn.Module):
    """
    Transformer with relative positional bias.
    Last output reduces to first token's embedding.
    """
    def __init__(self, query_inds, dim, depth, heads, head_dim, mlp_dim, dropout = 0., qkv_bias = False):
        super().__init__()
        if isinstance(query_inds, int):
            query_inds = [query_inds]
        self.query_inds = query_inds
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads = heads, head_dim = head_dim, dropout = dropout, qkv_bias = qkv_bias) if i < depth - 1 else ReductionAttention(dim, query_inds, heads = heads, head_dim = head_dim, dropout = dropout, qkv_bias = qkv_bias),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout = dropout),
            ]))
    def forward(self, x, bias):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x), bias) + (x[:, self.query_inds] if isinstance(attn, ReductionAttention) else x)
            x = ff(norm2(x)) + x
        return x

###################################### Swin Transformer

class SwinTransformer(nn.Module):
    def __init__(self, arch, img_size=224, patch_size=4):
        super(SwinTransformer, self).__init__()

        import sys
        sys.path.append('..')
        from SwinTransformer.models import build_model
        from SwinTransformer.config import _update_config_from_file, _C
        sys.path.pop()

        import argparse

        assert arch in ['Swin-T', 'Swin-S', 'Swin-B', 'Swin-L']

        arch2name = {
            'Swin-T': 'swin_tiny_patch4_window7_224_22k',
            'Swin-S': 'swin_small_patch4_window7_224_22k',
            'Swin-B': 'swin_base_patch4_window7_224_22k',
            'Swin-L': 'swin_large_patch4_window7_224_22k',
        }
        model_name = arch2name[arch]
        cfg_dir = '../SwinTransformer/configs/swin'
        ckpt_dir = '../SwinTransformer/checkpoints/swin'

        config = _C.clone()
        _update_config_from_file(config, f'{cfg_dir}/{model_name}.yaml')
        if '22k' in model_name and '22kto1k' not in model_name:
            config.defrost()
            config.MODEL.NUM_CLASSES = 21841
            config.freeze()

        self.swin = build_model(config)
        
        # init from pre-trained checkpoint
        ckpt_file = f'{ckpt_dir}/{model_name}.pth'
        ckpt_state_dict = torch.load(ckpt_file)['model']

        # ignore buffer 'attn_mask', which is not trainable and input shape independent
        new_state_dict = self.swin.state_dict()
        for k, v in new_state_dict.items():
            if 'attn_mask' not in k:
                new_state_dict[k] = ckpt_state_dict[k]
        self.swin.load_state_dict(new_state_dict)

        del self.swin.norm
        del self.swin.avgpool
        del self.swin.head

        self.update_input_resolution(img_size)
        self.update_patch_size(patch_size)
    
    def forward(self, x):
        x = self.swin.patch_embed(x)
        if self.swin.ape:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)

        outs = []
        for layer in self.swin.layers:
            e, x = layer(x)     # e is the output before downsampling (patch merging); x the output of downsampling (patch merging) if any
            outs.append(e)
        return outs

    def update_patch_size(self, new_patch_size):
        """
        Update patch embedding layer to accommandate new patch size
        """
        import sys
        sys.path.append('..')
        from SwinTransformer.models.swin_transformer import BasicLayer, SwinTransformerBlock, PatchEmbed, PatchMerging, to_2tuple, window_partition
        sys.path.pop()

        new_patch_size = to_2tuple(new_patch_size)
        old_patch_size = self.swin.patch_embed.patch_size
        if new_patch_size == old_patch_size:
            return

        for m in self.swin.modules():
            if isinstance(m, PatchEmbed):
                m.patch_size = new_patch_size
                m.patches_resolution = (m.img_size[0] // m.patch_size[0], m.img_size[1] // m.patch_size[1])
                m.num_patches = m.patches_resolution[0] * m.patches_resolution[1]

                old_proj = m.proj
                old_weight = old_proj.weight    # (embed_dim, in_chns, old_patch_size, old_patch_size)
                device = old_weight.device
                new_proj = nn.Conv2d(m.in_chans, m.embed_dim, kernel_size=m.patch_size, stride=m.patch_size).to(device)
                new_weight = new_proj.weight    # (embed_dim, in_chns, new_patch_size, new_patch_size)

                with torch.no_grad():
                    new_weight.copy_(
                        F.interpolate(old_weight, new_patch_size, mode='nearest') / (new_patch_size[0] * new_patch_size[1]) * (old_patch_size[0] * old_patch_size[1])
                    )
                del old_proj
                m.proj = new_proj

            elif isinstance(m, (BasicLayer, PatchMerging)):
                m.input_resolution = tuple([int(m.input_resolution[i] / new_patch_size[i] * old_patch_size[i]) for i in range(2)])
            elif isinstance(m, SwinTransformerBlock):
                m.input_resolution = tuple([int(m.input_resolution[i] / new_patch_size[i] * old_patch_size[i]) for i in range(2)])

                if min(m.input_resolution) <= m.window_size:
                    # if window size is larger than input resolution, we don't partition windows
                    m.shift_size = 0
                    #m.window_size = min(m.input_resolution)    # commented out by Yatao because we use padding to address this issue
                
                # added by Yatao to handle img size not divisible by window_size
                if m.input_resolution[0] % m.window_size != 0:
                    m.pad_h = m.window_size - m.input_resolution[0] % m.window_size
                else:
                    m.pad_h = 0
                if m.input_resolution[1] % m.window_size != 0:
                    m.pad_w = m.window_size - m.input_resolution[1] % m.window_size
                else:
                    m.pad_w = 0
                
                if m.shift_size > 0:
                    # calculate attention mask for SW-MSA
                    H, W = m.input_resolution
                    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                    h_slices = (slice(0, -m.window_size),
                                slice(-m.window_size, -m.shift_size),
                                slice(-m.shift_size, None))
                    w_slices = (slice(0, -m.window_size),
                                slice(-m.window_size, -m.shift_size),
                                slice(-m.shift_size, None))
                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1

                    img_mask = F.pad(img_mask, [0, 0, m.pad_w//2, m.pad_w-m.pad_w//2, m.pad_h//2, m.pad_h - m.pad_h//2])

                    mask_windows = window_partition(img_mask, m.window_size)  # nW, window_size, window_size, 1
                    mask_windows = mask_windows.view(-1, m.window_size * m.window_size)
                    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                    device = m.attn_mask.device
                    attn_mask = attn_mask.to(device)
                else:
                    attn_mask = None
                del m.attn_mask
                m.register_buffer("attn_mask", attn_mask) 

    def update_input_resolution(self, new_resolution):
        """
        Update all params associated with input resolution during class initialization
        """
        import sys
        sys.path.append('..')
        from SwinTransformer.models.swin_transformer import BasicLayer, SwinTransformerBlock, PatchEmbed, PatchMerging, to_2tuple, window_partition
        sys.path.pop()

        old_resolution = self.swin.patch_embed.img_size
        new_resolution = to_2tuple(new_resolution)

        if new_resolution == old_resolution:
            return

        for m in self.swin.modules():
            if isinstance(m, PatchEmbed):
                m.img_size = new_resolution
                m.patches_resolution = (m.img_size[0] // m.patch_size[0], m.img_size[1] // m.patch_size[1])
                m.num_patches = m.patches_resolution[0] * m.patches_resolution[1]
            elif isinstance(m, (BasicLayer, PatchMerging)):
                m.input_resolution = tuple([int(m.input_resolution[i] / old_resolution[i] * new_resolution[i]) for i in range(2)])
            elif isinstance(m, SwinTransformerBlock):
                m.input_resolution = tuple([int(m.input_resolution[i] / old_resolution[i] * new_resolution[i]) for i in range(2)])

                if min(m.input_resolution) <= m.window_size:
                    # if window size is larger than input resolution, we don't partition windows
                    m.shift_size = 0
                    # m.window_size = min(m.input_resolution)   # commented out by Yatao because we use padding to address this issue
                
                # added by Yatao to handle img size not divisible by window_size
                if m.input_resolution[0] % m.window_size != 0:
                    m.pad_h = m.window_size - m.input_resolution[0] % m.window_size
                else:
                    m.pad_h = 0
                if m.input_resolution[1] % m.window_size != 0:
                    m.pad_w = m.window_size - m.input_resolution[1] % m.window_size
                else:
                    m.pad_w = 0

                if m.shift_size > 0:
                    # calculate attention mask for SW-MSA
                    H, W = m.input_resolution
                    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                    h_slices = (slice(0, -m.window_size),
                                slice(-m.window_size, -m.shift_size),
                                slice(-m.shift_size, None))
                    w_slices = (slice(0, -m.window_size),
                                slice(-m.window_size, -m.shift_size),
                                slice(-m.shift_size, None))
                    cnt = 0
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                    
                    img_mask = F.pad(img_mask, [0, 0, m.pad_w//2, m.pad_w-m.pad_w//2, m.pad_h//2, m.pad_h - m.pad_h//2])

                    mask_windows = window_partition(img_mask, m.window_size)  # nW, window_size, window_size, 1
                    mask_windows = mask_windows.view(-1, m.window_size * m.window_size)
                    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                    device = m.attn_mask.device
                    attn_mask = attn_mask.to(device)
                else:
                    attn_mask = None
                del m.attn_mask
                m.register_buffer("attn_mask", attn_mask) 
    
###################################### Hourglass
# code from https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/posenet.py

class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = ResBottleneck2d(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResBottleneck2d(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf)
        else:
            self.low2 = ResBottleneck2d(nf, nf)
        self.low3 = ResBottleneck2d(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class StackedHourglass(nn.Module):
    def __init__(self, num_level, num_stack, inp_dim, out_dim, increase):
        super(StackedHourglass, self).__init__()

        self.num_level = num_level
        self.num_stack = num_stack

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(num_level, inp_dim, increase),
            ) for i in range(num_stack)
        ])
        
        self.features = nn.ModuleList([
            nn.Sequential(
                ResBottleneck2d(inp_dim, inp_dim),
                SameBlock2d(inp_dim, inp_dim, ksize=3, padding=1),
            ) for i in range(num_stack)
        ])
        
        self.outs = nn.ModuleList( [SameBlock2d(inp_dim, out_dim, ksize=3, padding=1) for i in range(num_stack)] )
        self.merge_features = nn.ModuleList( [nn.Conv2d(inp_dim, inp_dim, kernel_size=1, padding=0) for i in range(num_stack-1)] )
        self.merge_preds = nn.ModuleList( [nn.Conv2d(out_dim, inp_dim, kernel_size=1, padding=0) for i in range(num_stack-1)] )

    def forward(self, x):
        outs = []
        for i in range(self.num_stack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            pred = self.outs[i](feature)
            outs.append(pred)
            if i < self.num_stack - 1:
                x = x + self.merge_preds[i](pred) + self.merge_features[i](feature)
        return outs
