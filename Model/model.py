import torch
from torch import nn
import torch.nn.functional as F

import einops
import numpy as np
from scipy.spatial import KDTree
import functools
from collections import OrderedDict
try:
    from pytorch3d.ops import knn_points
except:
    pass

from .layers import SameBlock2d, DownBlock2d, TransformerV2, SameBlockTranspose2d, SwinTransformer

import pdb 

class KNNSearch(torch.autograd.Function):
    @staticmethod
    def forward(cxt, data, x, k, workers):
        """
        Find the knn in data for each point in x.
        data: (n, m)
        x: (n', m)
        """
        tree = KDTree(data.detach().cpu().numpy()) 
        dist, idx = tree.query(x.detach().cpu().numpy(), k=k, workers=workers)
        return torch.tensor(dist, dtype=x.dtype, device=x.device), torch.tensor(idx, dtype=torch.long, device=x.device)
    
    @staticmethod
    def backward(cxt, grad_output):
        raise NotImplementedError("To Do.")

def knn_search(data, x, k, workers):
    return KNNSearch.apply(data, x, k, workers)

class PatchEncoder(nn.Module):
    """
    only uses valid conv with stride=1 and padding=0
    """
    def __init__(self, input_size, input_channel, channels, min_num_convs_per_block=2):
        """
        channels should be for decreasing resolutions
        """
        super(PatchEncoder, self).__init__()

        num_blocks = len(channels)
        max_stride = 2**num_blocks
        assert (input_size - max_stride) % 2 == 0, f"input_size - max_stride must be multiples of 2, but got input_size={input_size}, max_stride={max_stride}"    # ensures the input size is muptiples of 2 to order to apply 3x3 valid conv
        assert input_size >= max_stride

        channels = [input_channel] + channels
        down_blocks = []
        for i in range(num_blocks):
            if i == 0:
                size_reduction = input_size - max_stride
            else:
                size_reduction = 2**(num_blocks - i)  # e.g., 16->8, 8->4 or 4->2, etc.
            num_convs = size_reduction // 2

            block = []
            num_extra_convs = max(0, min_num_convs_per_block - num_convs)
            # convs with ksize=3 and valid padding
            for j in range(num_convs): 
                block.append(SameBlock2d(channels[i] if j == 0 else channels[i+1], channels[i+1], ksize=3, padding=0))
            # extra convs with ksize=1 and valid padding
            for j in range(num_convs, num_convs + num_extra_convs):
                block.append(SameBlock2d(channels[i] if j == 0 else channels[i+1], channels[i+1], ksize=1, padding=0))
            down_blocks.append( nn.Sequential(*block) )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, img):
        res = []
        out = img
        for block in self.down_blocks:
            out = block(out)
            res.append(out)
        return res
        
class PatchDecoder(nn.Module):
    """
    only uses valid deconv with stride=1 and padding=0
    """
    def __init__(self, output_size, input_channels, channels, min_num_convs_per_block=2):
        """
        input_channels should be for increasing resolutions
        input_channels: encoder channels
        """
        super(PatchDecoder, self).__init__()

        num_blocks = len(input_channels)
        max_stride = 2**num_blocks
        assert (output_size - max_stride) % 2 == 0, f"output_size - max_stride must be multiples of 2, but got output_size={output_size}, max_stride={max_stride}"    # ensures the output size is muptiples of 2 to order to apply 3x3 valid conv
        assert output_size >= max_stride

        channels = [0] + channels
        up_blocks = []
        for i in range(num_blocks):
            if i == num_blocks - 1:
                size_increase = output_size - max_stride
            else:
                size_increase = 2**(i+1)  # e.g., 2->4, 4->8 or 8->16, etc.
            num_convs = size_increase // 2

            block = []
            num_extra_convs = max(0, min_num_convs_per_block - num_convs)
            # extra convs with ksize=1 and valid padding
            for j in range(num_extra_convs):
                prev_chn = channels[i]
                block.append(SameBlockTranspose2d((input_channels[i] + prev_chn) if j == 0 else channels[i+1], channels[i+1], ksize=1, padding=0))
            # convs with ksize=3 and valid padding
            for j in range(num_extra_convs, num_extra_convs + num_convs): 
                prev_chn = channels[i]
                block.append(SameBlockTranspose2d((input_channels[i] + prev_chn) if j == 0 else channels[i+1], channels[i+1], ksize=3, padding=0))
            up_blocks.append( nn.Sequential(*block) )
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, inputs):
        assert len(inputs) == len(self.up_blocks)
        out = None
        outputs = []
        for block in self.up_blocks:
            skip = inputs.pop()
            if out is None:
                out = skip
            else:
                out = torch.cat([out, skip], dim=1)
            out = block(out)
            outputs.append(out)
        return outputs

class ROIHead(nn.Module):
    def __init__(self, 
        # encoder/decoder params
        patch_size, input_dim, encoder_channels, decoder_channels,
        # transformer params
        use_transformer, rel_pos_embed, embed_dim, depth, heads, head_dim, mlp_dim, dropout,
        ):
        super(ROIHead, self).__init__()

        self.num_levels = len(encoder_channels)
        self.encoder = PatchEncoder(patch_size, input_dim, encoder_channels)
        self.decoder = PatchDecoder(patch_size, encoder_channels[::-1], decoder_channels)

        # transformer
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.rel_pos_embed = rel_pos_embed
            enc_dim = encoder_channels[-1]
            self.enc_head = nn.Linear(enc_dim, embed_dim)
            self.transformer = TransformerV2([0], embed_dim, depth, heads, head_dim, mlp_dim, dropout)
            self.dec_head = nn.Linear(embed_dim, enc_dim)

    def forward(self, rgbm_rois, rel_pos_bias=None, nearest_idx=None):
        """
        rgbm_rois: (num_rois, 4, h, w)
        rel_pos_bias: (num_rois, num_heads, nearest_k+1, nearest_k+1)
        nearest_idx: (num_rois, nearest_k)
        """
        enc_feats = self.encoder(rgbm_rois)

        if self.use_transformer:
            # compute input embedding for transformer
            num_rois, enc_dim, enc_out_h, enc_out_w = enc_feats[-1].shape
            if enc_out_h * enc_out_w == 1:
                embed_feat = enc_feats[-1].view(num_rois, enc_dim)     # (num_rois, enc_dim)
            else:
                embed_feat = F.adaptive_avg_pool2d(enc_feats[-1], output_size=1).view(num_rois, enc_dim)

            embed_feat = self.enc_head(embed_feat)          # (num_rois, embed_dim)
            knn_embed_feat = embed_feat[nearest_idx, :]       # (num_rois, nearest_k, embed_dim)
            embed_feat = torch.cat([embed_feat.unsqueeze(1), knn_embed_feat], dim=1)      # (num_rois, nearest_k+1, embed_dim)
            embed_feat = self.transformer(embed_feat, rel_pos_bias)
            embed_feat = self.dec_head(embed_feat).view(num_rois, enc_dim, 1, 1)      # (num_rois, enc_dim, 1, 1)

            if enc_out_h * enc_out_w == 1:
                enc_feats[-1] = embed_feat
            else:
                enc_feats[-1] = enc_feats[-1] + embed_feat
        
        # decoder
        dec_feats = self.decoder(enc_feats)
        return enc_feats, dec_feats
    
class Decoder(nn.Module):
    def __init__(self, channels, in_channels, num_convs_per_block):
        super(Decoder, self).__init__()

        assert len(channels) == len(in_channels) + 1
        self.num_blocks = len(channels)

        up_blocks = []
        for i in range(self.num_blocks): 
            block = [] 
            if i != 0:
                for j in range(num_convs_per_block):
                    if j == 0:
                        block.append( SameBlock2d(channels[i-1] + in_channels[i-1], channels[i], ksize=3, padding=1, norm_fn=None) )
                    else:
                        block.append( SameBlock2d(channels[i], channels[i], ksize=3, padding=1, norm_fn=None) )
            #if i != self.num_blocks - 1:
            #    block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            block = nn.Sequential(*block)
            up_blocks.append(block)
        self.up_blocks = nn.ModuleList(up_blocks)
    
    def forward(self, inputs):
        assert len(inputs) == len(self.up_blocks)
        out = None
        outputs = []
        for block in self.up_blocks:
            skip = inputs.pop()
            if out is not None:
                out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                out = torch.cat([out, skip], dim=1)
            else:
                out = skip
            out = block(out)
            outputs.append(out)
        return outputs

class QKVLinear(nn.Module):
    """
    A layer that splits the original combined qkv weights
    """
    def __init__(self, qkv):
        super(QKVLinear, self).__init__()

        qkv_weight = qkv.weight
        qkv_bias = qkv.bias
        dim = qkv_weight.shape[1]
        qkv_weight = qkv_weight.view(3, -1, dim)
        qkv_bias = qkv_bias.view(3, -1)
        self.q_weight = nn.Parameter(qkv_weight[0])
        self.q_bias = nn.Parameter(qkv_bias[0])
        self.k_weight = nn.Parameter(qkv_weight[1])
        self.k_bias = nn.Parameter(qkv_bias[1])
        self.v_weight = nn.Parameter(qkv_weight[2])
        self.v_bias = nn.Parameter(qkv_bias[2])

    def forward(self, x):
        qkv_weight = torch.cat([self.q_weight, self.k_weight, self.v_weight], dim=0)
        qkv_bias = torch.cat([self.q_bias, self.k_bias, self.v_bias], dim=0)
        return F.linear(x, qkv_weight, qkv_bias)

class PatchMergingNorm(nn.Module):
    """
    A layer that splits the original combined norm layer in patch merging
    """
    def __init__(self, norm):
        super(PatchMergingNorm, self).__init__()

        norm_weight = norm.weight        
        norm_bias = norm.bias        
        dim = norm_weight.shape[0] // 4
        norm_weight = norm_weight.view(4, dim)
        norm_bias = norm_bias.view(4, dim)
        weights = []
        biases = []
        for i in range(4):
            weights.append(nn.Parameter(norm_weight[i]))
            biases.append(nn.Parameter(norm_bias[i]))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.eps = norm.eps
        self.normalized_shape = norm.normalized_shape

    def forward(self, x):
        norm_weight = torch.cat(list(self.weights), dim=0)
        norm_bias = torch.cat(list(self.biases), dim=0)
        return F.layer_norm(x, self.normalized_shape, norm_weight, norm_bias, self.eps)

class Encoder(nn.Module):

    def __init__(self, arch, img_size, patch_size, shrink_ratio):
        super(Encoder, self).__init__()

        assert isinstance(patch_size, int)

        self.img_size = img_size
        self.patch_size = patch_size

        self.model = SwinTransformer(arch, img_size, patch_size)

        # convert qkv layers and patch merging layer to custom equivalent for ZIG partition
        for i, layer in enumerate(self.model.swin.layers):
            for j, block in enumerate(layer.blocks):
                # split weight/bias in qkv for zig partition
                qkv = QKVLinear(block.attn.qkv)
                del block.attn.qkv
                block.attn.qkv = qkv
            # split weight/bias in layernorm for zig partition
            if layer.downsample is not None:
                norm = PatchMergingNorm(layer.downsample.norm)
                del layer.downsample.norm
                layer.downsample.norm = norm

        # shrink model
        if shrink_ratio < 1:
            import sys
            sys.path.append('..')
            from SwinTransformer.models.swin_transformer import BasicLayer, SwinTransformerBlock, PatchEmbed, PatchMerging, WindowAttention
            sys.path.pop()

            _shrink = lambda c: int(round(c * shrink_ratio))

            self.model.swin.embed_dim = _shrink(self.model.swin.embed_dim)

            for name, m in self.model.swin.named_modules():
                if isinstance(m, (BasicLayer, SwinTransformerBlock, PatchMerging, WindowAttention)):
                    if hasattr(m, 'dim'):
                        m.dim = _shrink(m.dim)
                    if isinstance(m, WindowAttention):
                        if hasattr(m, 'qk_head_dim'):
                            m.qk_head_dim = _shrink(m.qk_head_dim)
                        if hasattr(m, 'v_head_dim'):
                            m.v_head_dim = _shrink(m.v_head_dim)
                        
                elif isinstance(m, nn.Conv2d):
                    if 'patch_embed' in name:
                        # shrink output channels only
                        old_weight = m.weight
                        old_bias = m.bias
                        new_out_channels = _shrink(m.out_channels)
                        new_weight = interp_out_channels(old_weight, new_out_channels)
                        new_bias = interp_out_channels(old_bias, new_out_channels)
                        del m.weight
                        del m.bias
                        m.weight = nn.Parameter(new_weight)
                        m.bias = nn.Parameter(new_bias)
                        m.out_channels = new_out_channels
                    else: 
                        old_weight = m.weight
                        old_bias = m.bias
                        new_in_channels = _shrink(m.in_channels)
                        new_out_channels = _shrink(m.out_channels)
                        new_weight = interp_in_out_channels(old_weight, new_in_channels, new_out_channels)
                        new_bias = interp_out_channels(old_bias, new_out_channels)
                        del m.weight
                        del m.bias
                        m.weight = nn.Parameter(new_weight)
                        m.bias = nn.Parameter(new_bias)
                        m.in_channels = new_in_channels
                        m.out_channels = new_out_channels
                elif isinstance(m, nn.Linear):
                    old_weight = m.weight
                    old_bias = m.bias
                    new_in_features = _shrink(m.in_features)
                    new_out_features = _shrink(m.out_features)
                    new_weight = interp_in_out_channels(old_weight, new_in_features, new_out_features)
                    new_bias = interp_out_channels(old_bias, new_out_features) if old_bias is not None else None
                    del m.weight
                    del m.bias
                    m.weight = nn.Parameter(new_weight)
                    m.bias = nn.Parameter(new_bias) if new_bias is not None else None
                    m.in_features = new_in_features
                    m.out_features = new_out_features
                elif isinstance(m, QKVLinear):
                    for pre in ['q', 'k', 'v']:
                        old_weight = getattr(m, f'{pre}_weight') 
                        old_bias = getattr(m, f'{pre}_bias') 
                        new_in_features = _shrink(old_weight.shape[1])
                        new_out_features = _shrink(old_weight.shape[0])
                        new_weight = interp_in_out_channels(old_weight, new_in_features, new_out_features)
                        new_bias = interp_out_channels(old_bias, new_out_features)
                        delattr(m, f'{pre}_weight')
                        delattr(m, f'{pre}_bias')
                        setattr(m, f'{pre}_weight', nn.Parameter(new_weight))
                        setattr(m, f'{pre}_bias', nn.Parameter(new_bias))
                elif isinstance(m, nn.LayerNorm):
                    old_weight = m.weight
                    old_bias = m.bias
                    new_out_features = _shrink(m.normalized_shape[0])
                    new_weight = interp_out_channels(old_weight, new_out_features)
                    new_bias = interp_out_channels(old_bias, new_out_features)
                    del m.weight
                    del m.bias
                    m.weight = nn.Parameter(new_weight)
                    m.bias = nn.Parameter(new_bias)
                    m.normalized_shape = (new_out_features,)
                elif isinstance(m, PatchMergingNorm):
                    for i, (old_weight, old_bias) in enumerate(zip(m.weights, m.biases)):
                        new_out_features = _shrink(old_weight.shape[0])
                        new_weight = interp_out_channels(old_weight, new_out_features)
                        new_bias = interp_out_channels(old_bias, new_out_features)
                        del old_weight
                        del old_bias
                        m.weights[i] = nn.Parameter(new_weight)
                        m.biases[i] = nn.Parameter(new_bias)
                    m.normalized_shape = (_shrink(m.normalized_shape[0]), )


        self.output_channels = OrderedDict() 
        for i in range(self.model.swin.num_layers):
            self.output_channels[f'x{patch_size * 2**i}'] = self.model.swin.embed_dim * 2**i     # the resolution scale here is relative to the 'input_rate' downsampled input, not the full original input

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, img):
        img = (img - self.mean) / self.std  # normalize image according to pretrained models
        outs = self.model(img)
        
        batch_size = img.shape[0]
        for i in range(self.model.swin.num_layers):
            stride = self.model.swin.patch_embed.patch_size[0] * 2**i
            h, w = self.model.swin.patch_embed.img_size
            outs[i] = outs[i].transpose(1, 2).view(batch_size, -1, h // stride, w // stride)
        return outs

class LowresModel(nn.Module):
    def __init__(self, arch, input_resolution, input_rate, downsample_rate, patch_size, shrink_ratio, d2s):
        super(LowresModel, self).__init__()

        d2s_factor = input_rate * patch_size // downsample_rate
        d2s = d2s_factor > 1 and d2s    # depth to space to compensate insufficient resolution
        if d2s:
            n2 = int(round(np.log(d2s_factor) / np.log(2)))
        else:
            assert d2s_factor <= 1
            n2 = 0
        self.d2s_factor = d2s_factor

        self.encoder = Encoder(arch, input_resolution // input_rate, patch_size, shrink_ratio)
        sorted_output_channels = [c for c in self.encoder.output_channels.values()]    # #channels sorted by decreasing resolution
        n0 = int(round(np.log(input_rate) / np.log(2)))
        n1 = int(round(np.log(self.encoder.patch_size / 4) / np.log(2)))
        n = n0 + n1
        if downsample_rate == 4:
            dec_chns = [sorted_output_channels[-1]] + [256] * (n - n2) + [128, 64, 32]        
            enc_chns = sorted_output_channels[-(4+n):-1][::-1]
        elif downsample_rate == 8:
            dec_chns = [sorted_output_channels[-1]] + [256] * (n - n2) + [128, 64]        
            enc_chns = sorted_output_channels[-(3+n):-1][::-1]
        elif downsample_rate == 16:
            dec_chns = [sorted_output_channels[-1]] + [256] * (n - n2) + [128]        
            enc_chns = sorted_output_channels[-(2+n):-1][::-1]
        else:
            raise NotImplementedError("To Do")

        self.decoder = Decoder(dec_chns, enc_chns, 1)
        self.enc_chns = enc_chns
        self.dec_chns = dec_chns

        self.num_lowres_pyramid = self.decoder.num_blocks

    def forward(self, lowres_img):
        enc_outs = self.encoder(lowres_img)
        dec_outs = self.decoder(enc_outs[-self.num_lowres_pyramid:])
        return enc_outs, dec_outs

class MattingModel(nn.Module):
    _embed_dim = 128
    _depth = 2
    _heads = 8
    _head_dim = 16
    _mlp_dim = 128
    _dropout = 0

    def __init__(self, model_cfg, **kwargs):
        super(MattingModel, self).__init__()

        self.enable_highres_head = model_cfg.ROI_HEAD.ENABLE
        self.input_rate = model_cfg.INPUT_RATE
        self.downsample_rate = model_cfg.DOWNSAMPLE_RATE

        self.lowres_model = LowresModel(
            model_cfg.ENCODER_ARCH,
            model_cfg.INPUT_RESOLUTION,
            model_cfg.INPUT_RATE,
            model_cfg.DOWNSAMPLE_RATE,
            model_cfg.PATCH_SIZE,
            model_cfg.SHRINK_RATIO,
            model_cfg.D2S,
        )

        self.num_lowres_heads = int(np.log(self.downsample_rate) / np.log(2))   # 3 for downsample_rate=8; 2 for downsample_rate=4
        self.trimap_heads = nn.ModuleList([nn.Conv2d(c, 3 * self.lowres_model.d2s_factor**2, kernel_size=1, padding=0) for c in self.lowres_model.dec_chns[-self.num_lowres_heads:]])
        self.seg_heads = nn.ModuleList([nn.Conv2d(c, 1 * self.lowres_model.d2s_factor**2, kernel_size=1, padding=0) for c in self.lowres_model.dec_chns[-self.num_lowres_heads:]])

        if self.enable_highres_head:
            self.adaptive_sampling = model_cfg.ROI_HEAD.ADAPTIVE_SAMPLING
            self.use_transformer = model_cfg.ROI_HEAD.TRANSFORMER.ENABLE
            self.rel_pos_embed = model_cfg.ROI_HEAD.TRANSFORMER.REL_POS_EMBED
            self.roi_unfold_size = model_cfg.ROI_HEAD.UNFOLD_SIZE

            # for transformer
            if self.use_transformer:
                self.nearest_k = model_cfg.ROI_HEAD.TRANSFORMER.NEAREST_K
                self.rel_pos_range = model_cfg.ROI_HEAD.TRANSFORMER.REL_POS_RANGE
                self.num_valid_rel_pos = (2 * self.rel_pos_range - 1) * (2 * self.rel_pos_range - 1)
                self.rel_pos_bias_table = nn.Parameter(
                    torch.zeros(self.num_valid_rel_pos + 1, self._heads))  # (num_valid_rel_pos + 1, heads); the extra 1 entry is for rel pos outside range

            if self.downsample_rate == 4:
                enc_chns = [32, 64]     
                dec_chns = [32, 32]    
            elif self.downsample_rate == 8:
                enc_chns = [32, 64, 64]      
                dec_chns = [32, 32, 16]    
            elif self.downsample_rate == 16:
                enc_chns = [32, 64, 64, 128]      
                dec_chns = [32, 32, 16, 8]    
            else:
                raise NotImplementedError("To Do")
            roi_size = self.downsample_rate
            self.roi_head = ROIHead(
                roi_size, 4*self.roi_unfold_size**2, enc_chns, dec_chns,
                self.use_transformer, self.rel_pos_embed, self._embed_dim, self._depth, self._heads, self._head_dim, self._mlp_dim, self._dropout,
                )
            self.roi_matte_head = nn.Conv2d(dec_chns[-1], 1, kernel_size=1, padding=0)

    def normalize_coords(self, coords, size):
        step = 1. / float(size)
        return step / 2. + coords.float() * step

    def create_grid_coords(self, h, w, device):
        my, mx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
        )   # (h, w)
        # normalize points to [0, 1]
        mx = self.normalize_coords(mx, w)
        my = self.normalize_coords(my, h)
        mxy = torch.stack([mx, my], dim=-1)      # (h, w, 2)
        return mxy

    def find_region_idx(self, trimap_score, adaptive):
        b, c, h, w = trimap_score.shape
        unknown_id = 1      # 0 is bg, 1 is unknown region, 2 is fg
        trimap_cls = torch.argmax(trimap_score, dim=1)
        unknown_mask = (trimap_cls == unknown_id)
        if adaptive:
            idx = torch.nonzero(unknown_mask)    # (total_rois, 3)
        else:
            uncertainty_map = trimap_score[:, unknown_id] * unknown_mask.float()
            topk_rois = 2*(h+w)
            idx = uncertainty_map.view(b, -1).topk(topk_rois, dim=1, sorted=False).indices      # (b, topk_rois)
            h_idx = idx // w    # (b, topk_rois)
            w_idx = idx % w     # (b, topk_rois)
            b_idx = torch.arange(b, dtype=torch.long, device=uncertainty_map.device).unsqueeze(1).expand(b, topk_rois)    # (b, topk_rois)
            idx = torch.stack([b_idx, h_idx, w_idx], dim=-1).view(-1, 3)     # (b*topk_rois, 3)
        return idx

    def get_roi_pixel_indices(self, lowres_idx_b, lowres_idx_h, lowres_idx_w, size, extension):
        if isinstance(extension, int):
            extension = np.array([extension, extension])
        highres_idx_h = lowres_idx_h * size     # (num_rois,)
        highres_idx_w = lowres_idx_w * size     # (num_rois,)
        roi_size = size + extension[0] + extension[1]
        my, mx = torch.meshgrid(
            torch.arange(roi_size, dtype=lowres_idx_b.dtype, device=lowres_idx_b.device),
            torch.arange(roi_size, dtype=lowres_idx_b.dtype, device=lowres_idx_b.device),
        )   # (roi_size, roi_size)
        highres_roi_idx_h = highres_idx_h.unsqueeze(1) + my.reshape(1, -1)     # (num_rois, roi_size*roi_size)
        highres_roi_idx_w = highres_idx_w.unsqueeze(1) + mx.reshape(1, -1)     # (num_rois, roi_size*roi_size)
        highres_roi_idx_b = lowres_idx_b.unsqueeze(1).expand(-1, roi_size*roi_size)   # (num_rois, roi_size*roi_size)
        
        return highres_roi_idx_b, highres_roi_idx_h, highres_roi_idx_w
    
    def crop_regions(self, highres_src, lowres_idx, size, extension):
        """
        highres_src: (b, c, h, w)
        lowres_idx: (num_rois, 3). Indices into batch, height, width
        size: region size
        """
        if extension == 0 or extension == tuple([0, 0]) or extension == [0, 0] or extension is False or extension is None:
            b, c, h, w = highres_src.shape
            return highres_src.view(b, c, h//size, size, w//size, size).permute(0, 1, 2, 4, 3, 5)[lowres_idx[:, 0], :, lowres_idx[:, 1], lowres_idx[:, 2]]     # (num_rois, c, size, size)
        else:
            if isinstance(extension, int):
                extension = np.array([extension, extension])

            b, c, h, w = highres_src.shape
            num_rois = lowres_idx.shape[0]
            roi_size = size + extension[0] + extension[1]

            # compute indices for all entries in ROI
            highres_roi_idx_b, highres_roi_idx_h, highres_roi_idx_w = self.get_roi_pixel_indices(*[lowres_idx[:, i] for i in range(3)], size, extension)

            # crop
            padded_highres_src = F.pad(highres_src, (extension[0], extension[1], extension[0], extension[1]))
            regions = padded_highres_src[highres_roi_idx_b, :, highres_roi_idx_h, highres_roi_idx_w]    # (num_rois, c, roi_size*roi_size)
            return regions.view(num_rois, c, roi_size, roi_size)

    def replace_region(self, highres_src, rois, lowres_idx, extension):
        """
        highres_src: (b, c, h, w)
        rois: (num_rois, c, roi_size, roi_size)
        lowres_idx: (num_rois, 3). Indices into batch, height, width
        """
        if extension == 0 or extension == tuple([0, 0]) or extension == [0, 0] or extension is False or extension is None:
            size = rois.shape[2]
            b, c, h, w = highres_src.shape
            unfold_src = highres_src.view(b, c, h//size, size, w//size, size).permute(0, 1, 2, 4, 3, 5)    # (b, c, h//size, w//size, size, size)
            unfold_src[lowres_idx[:, 0], :, lowres_idx[:, 1], lowres_idx[:, 2]] = rois
            return unfold_src.permute(0, 1, 2, 4, 3, 5).view(b, c, h, w)
        else:
            if isinstance(extension, int):
                extension = np.array([extension, extension])

            num_rois = rois.shape[0]
            roi_size = rois.shape[2]
            size = roi_size - extension[0] - extension[1]
            b, c, h, w = highres_src.shape

            # compute indices for all entries in ROI
            highres_roi_idx_b, highres_roi_idx_h, highres_roi_idx_w = self.get_roi_pixel_indices(*[lowres_idx[:, i] for i in range(3)], size, extension)
            idx = highres_roi_idx_b * (h + extension[0] + extension[1]) * (w + extension[0] + extension[1]) + highres_roi_idx_h * (w + extension[0] + extension[1]) + highres_roi_idx_w   # (num_rois, roi_size*roi_size)

            # first, set all indexed entries to zero
            padded_highres_src = F.pad(highres_src, (extension[0], extension[1], extension[0], extension[1]))       # (b, c, h+2*p, w+2*p)
            padded_highres_src = einops.rearrange(padded_highres_src, 'b c h w -> (b h w) c')   # (b*(h+2*p)*(w+2*p), c)
            padded_highres_src_with_cnt = torch.cat(
                [
                    padded_highres_src,
                    torch.ones_like(padded_highres_src[:, :1])
                ],
                dim=-1
            )       # (b*(h+2*p)*(w+2*p), c+1)
            padded_highres_src_with_cnt[idx] = 0
            
            # second, overwrite and accumulate all indexed entries
            rois = einops.rearrange(rois, 'n c kh kw -> n (kh kw) c')              # (num_rois, roi_size*roi_size, c)
            rois_with_cnt = torch.cat(
                [
                    rois, 
                    torch.ones_like(rois[..., :1])
                ],
                dim=-1,
            ) 
            padded_highres_src_with_cnt.index_put_(
                indices=(idx,),      # (num_rois, roi_size*roi_size)
                values=rois_with_cnt,
                accumulate=True,
            )       # returned is (b*(h+2*p)*(w+2*p), c+1)

            # compute average after accumulation
            assert padded_highres_src_with_cnt[..., -1].min() > 0   # no zeros in cnt
            padded_highres_src = padded_highres_src_with_cnt[..., :-1] / padded_highres_src_with_cnt[..., -1:]
            padded_highres_src = einops.rearrange(padded_highres_src, '(b h w) c -> b c h w', h=h+extension[0]+extension[1], w=w+extension[0]+extension[1])

            return padded_highres_src[:, :, extension[0]:-extension[1], extension[0]:-extension[1]]
    
    def find_nearest_neighbors(self, batch_size, idx):
        """
        idx: (num_rois, 3). Indices into batch, height and width of lowres images
        """
        num_rois = idx.shape[0]
        flat_idx = torch.arange(num_rois, dtype=idx.dtype, device=idx.device)
        nearest_flat_idx = torch.zeros([num_rois, self.nearest_k], dtype=idx.dtype, device=idx.device)
        for i in range(batch_size):
            select_mask = (idx[:, 0]==i)
            yx = idx[select_mask, 1:].float()     # (num_rois_of_batch, 2)
            if yx.shape[0] == 0:
                continue
            # compute pariwise distance matrix
            dist = torch.norm(yx.unsqueeze(1) - yx.unsqueeze(0), dim=-1)    # (num_rois_of_batch, num_rois_of_batch)
            dist += torch.eye(dist.shape[0], dtype=torch.float32, device=idx.device) * 1.e6       # set diagnal to large number to exclude self from k nearest neighbors
            # find k nearest neighbors
            k = min(self.nearest_k, dist.shape[0]) 
            assert k >= 1
            #print(f"\033[92m DEBUG: dist.shape={dist.shape}, k={k} \033[0m")
            _, nearest_inds = torch.topk(dist, k, dim=1, largest=False)    # (num_rois_of_batch, k)
            if k < self.nearest_k:
                m = int(np.ceil(self.nearest_k / k))
                nearest_inds = torch.cat([nearest_inds] * m, dim=1)[:, :self.nearest_k]
            # retrieve flat idx
            nearest_flat_idx_of_batch = flat_idx[select_mask][nearest_inds]       # (num_rois_of_batch, nearest_k)
            nearest_flat_idx[select_mask] = nearest_flat_idx_of_batch
        return nearest_flat_idx

    def find_nearest_neighbors_pyt3d(self, batch_size, idx):
        """
        Find KNN using pytorch3d API
        idx: (num_rois, 3). Indices into batch, height and width of lowres images
        """
        num_rois = idx.shape[0]
        flat_idx = torch.arange(num_rois, dtype=idx.dtype, device=idx.device)
        nearest_flat_idx = torch.zeros([num_rois, self.nearest_k], dtype=idx.dtype, device=idx.device)
        for i in range(batch_size):
            select_mask = (idx[:, 0]==i)
            yx = idx[select_mask, 1:].float()     # (num_rois_of_batch, 2)
            if yx.shape[0] == 0:
                continue
            # find k nearest neighbors
            k = min(self.nearest_k, yx.shape[0] - 1)    # eliminate self
            assert k >= 1
            ret = knn_points(yx.unsqueeze(0), yx.unsqueeze(0), K=k, return_nn=False, return_sorted=False)    
            nearest_inds = ret.idx[0]      # (num_rois_of_batch, k)
            if k < self.nearest_k:
                m = int(np.ceil(self.nearest_k / k))
                nearest_inds = torch.cat([nearest_inds] * m, dim=1)[:, :self.nearest_k]
            # retrieve flat idx
            nearest_flat_idx_of_batch = flat_idx[select_mask][nearest_inds]       # (num_rois_of_batch, nearest_k)
            nearest_flat_idx[select_mask] = nearest_flat_idx_of_batch
        return nearest_flat_idx

    def find_nearest_neighbors_kdtree(self, batch_size, idx):
        """
        idx: (num_rois, 3). Indices into batch, height and width of lowres images
        """
        num_rois = idx.shape[0]
        flat_idx = torch.arange(num_rois, dtype=idx.dtype, device=idx.device)
        nearest_flat_idx = torch.zeros([num_rois, self.nearest_k], dtype=idx.dtype, device=idx.device)
        for i in range(batch_size):
            select_mask = (idx[:, 0]==i)
            yx = idx[select_mask, 1:].float()     # (num_rois_of_batch, 2)
            if yx.shape[0] == 0:
                continue
            k = min(self.nearest_k, yx.shape[0] - 1) 
            assert k >= 1
            dist, nearest_inds = knn_search(yx, yx, k=k+1, workers=1)
            nearest_inds = nearest_inds[:, 1:]      # exclude self
            if k < self.nearest_k:
                m = int(np.ceil(self.nearest_k / k))
                nearest_inds = torch.cat([nearest_inds] * m, dim=1)[:, :self.nearest_k]
            # retrieve flat idx
            nearest_flat_idx_of_batch = flat_idx[select_mask][nearest_inds]       # (num_rois_of_batch, nearest_k)
            nearest_flat_idx[select_mask] = nearest_flat_idx_of_batch
        return nearest_flat_idx

    def point_sample(self, x, batch_idx, point_coords, **kwargs):
        """
        x: (b, c, h, w)
        batch_idx: (num_points, ). Index into batch dimenstion of x
        point_coords: (num_points, 2). Assumes point_coords are normalized to [0, 1]
        """
        batch_size = x.shape[0]
        output = torch.zeros([point_coords.shape[0], x.shape[1]], dtype=point_coords.dtype, device=point_coords.device)
        for i in range(batch_size):
            select_mask = (batch_idx==i)
            pts_xy = point_coords[select_mask]     # (num_points_of_batch, 2)
            if pts_xy.shape[0] == 0:
                continue
            pts_xy = pts_xy.unsqueeze(0).unsqueeze(2) * 2 - 1      # convert from [0, 1] to [-1, 1]
            sampled = F.grid_sample(x[i:i+1], pts_xy, **kwargs).squeeze(3).squeeze(0)   # (num_points_of_batch, c)
            output[select_mask] = sampled.transpose(0, 1)
        return output
    
    def sparse_upsample(self, lowres_src, select_idx, upsample_rate, roi_extension):
        """
        Upsample lowres_src to full resolution using point sampling
        """
        reso = 'x{}'.format(upsample_rate)
        upsample_shape = [s * upsample_rate for s in lowres_src.shape[2:]]
        if not hasattr(self, f'upsample_grid_{reso}') or (hasattr(self, f'upsample_grid_{reso}') and getattr(self, f'upsample_grid_{reso}').shape[2:] != torch.Size(upsample_shape)):
            upsample_grid = self.create_grid_coords(upsample_shape[0], upsample_shape[1], lowres_src.device)
            upsample_grid = upsample_grid.unsqueeze(0).repeat(lowres_src.shape[0], 1, 1, 1)
            upsample_grid = upsample_grid.permute(0, 3, 1, 2)
            self.register_buffer(f'upsample_grid_{reso}', upsample_grid)  # (b, 2, h, w)
        else:
            upsample_grid = getattr(self, f'upsample_grid_{reso}')
        region_grid = self.crop_regions(upsample_grid, select_idx, upsample_rate, roi_extension)   # (num_rois, 2, roi_size, roi_size)
        roi_shape = region_grid.shape[2:]
        grid_sampling_points = einops.rearrange(region_grid, 'b c h w -> (b h w) c')     # (num_rois*roi_size*roi_size, 2)
        grid_batch_idx = select_idx[:, 0].unsqueeze(1).unsqueeze(2).repeat(1, roi_shape[0], roi_shape[1]).view(-1)    # (num_rois*roi_size*roi_size, )
        sampled_src = self.point_sample(lowres_src, grid_batch_idx, grid_sampling_points, mode='bilinear', align_corners=False)   # (num_rois*roi_size*roi_size, c)
        sampled_src = einops.rearrange(sampled_src, '(b h w) c -> b c h w', h=roi_shape[0], w=roi_shape[1])     # (num_rois, c, roi_size, roi_size)
        return sampled_src
    
    def forward(self, inputs):
        ret = self.forward_lowres(inputs)

        lowres_seg = ret['lowres_segs'][-1]
        lowres_trimap = ret['lowres_trimaps'][-1]

        if 'overwrite_lowres_trimap' in inputs.keys():
            lowres_trimap = inputs['overwrite_lowres_trimap']
        if 'overwrite_lowres_seg' in inputs.keys():
            lowres_seg = inputs['overwrite_lowres_seg']

        if self.enable_highres_head:
            highres_inputs = {'highres_img': inputs['highres_img'], 'lowres_seg': lowres_seg, 'lowres_trimap': lowres_trimap}
            if 'highres_matte_gt' in inputs.keys():
                highres_inputs['highres_matte_gt'] = inputs['highres_matte_gt']
            highres_ret = self.forward_highres(highres_inputs)
            ret.update(highres_ret)

        return ret
    
    def forward_lowres(self, inputs):
        highres_img = inputs['highres_img']
        if ('lowres_img' not in inputs.keys()) or (inputs['lowres_img'].shape[2] * self.input_rate != highres_img.shape[2]) or (inputs['lowres_img'].shape[3] * self.input_rate != highres_img.shape[3]):
            if self.input_rate > 1:
                lowres_img = F.interpolate(highres_img, scale_factor=1/self.input_rate, mode='area')  
            else:
                lowres_img = highres_img
        else:
            lowres_img = inputs['lowres_img']

        with torch.autograd.profiler.record_function("lowres_model"):    # code context for profiler
            enc_outs, dec_outs = self.lowres_model(lowres_img)
        lowres_outs = dec_outs[-self.num_lowres_heads:]
        assert len(lowres_outs) == len(self.trimap_heads)
        assert len(lowres_outs) == len(self.seg_heads)

        lowres_trimap_logits = [None] * len(lowres_outs)
        lowres_trimaps = [None] * len(lowres_outs)
        lowres_seg_logits = [None] * len(lowres_outs)
        lowres_segs = [None] * len(lowres_outs)
        for i in range(len(lowres_outs)):
            lowres_trimap_logits[i] = self.trimap_heads[i](lowres_outs[i])
            lowres_seg_logits[i] = self.seg_heads[i](lowres_outs[i])
            if self.lowres_model.d2s_factor > 1:
                lowres_trimap_logits[i] = einops.rearrange(lowres_trimap_logits[i], 'b (c kh kw) h w -> b c (h kh) (w kw)', c=3, kh=self.lowres_model.d2s_factor, kw=self.lowres_model.d2s_factor)
                lowres_seg_logits[i] = einops.rearrange(lowres_seg_logits[i], 'b (c kh kw) h w -> b c (h kh) (w kw)', c=1, kh=self.lowres_model.d2s_factor, kw=self.lowres_model.d2s_factor)
            lowres_trimaps[i] = F.softmax(lowres_trimap_logits[i], dim=1)
            lowres_segs[i] = F.sigmoid(lowres_seg_logits[i])

        ret = {
            'lowres_trimap_logits': lowres_trimap_logits,
            'lowres_trimaps': lowres_trimaps,
            'lowres_seg_logits': lowres_seg_logits,
            'lowres_segs': lowres_segs,
        }
        return ret
    
    def forward_highres(self, inputs):
        highres_img = inputs['highres_img']
        lowres_seg = inputs['lowres_seg']
        lowres_trimap = inputs['lowres_trimap']

        # find & crop ROIs
        with torch.no_grad():
            idx = self.find_region_idx(lowres_trimap, self.adaptive_sampling)    # (num_rois, 3). lowres region idx

        up_lowres_seg = F.interpolate(
            lowres_seg,
            scale_factor=self.downsample_rate,
            mode='bilinear', align_corners=False,
        )
        highres_rgbm = torch.cat([highres_img, up_lowres_seg], dim=1)

        assert self.roi_unfold_size % 2 == 1, "roi_unfold_size must be odd number because the padding parameter in F.unfold only supports symmetric padding."
        b, c, h, w = highres_rgbm.shape
        rgbm_rois = self.crop_regions(
            F.unfold(highres_rgbm, kernel_size=self.roi_unfold_size, padding=(self.roi_unfold_size//2, self.roi_unfold_size//2)).view(b, -1, h, w), 
            idx, self.downsample_rate, 0)   # (num_rois, roi_unfold_size**2, roi_size, roi_size)
        
        if self.use_transformer:
            with torch.no_grad():
                # find k nearest neighbors
                batch_size = highres_img.shape[0]
                nearest_idx = self.find_nearest_neighbors(batch_size, idx)  # (num_rois, nearest_k)
                #nearest_idx = self.find_nearest_neighbors_pyt3d(batch_size, idx)  # (num_rois, nearest_k)
                #nearest_idx = self.find_nearest_neighbors_kdtree(batch_size, idx)  # (num_rois, nearest_k)

                # debug kdtree
                #try:
                #    assert torch.all(nearest_idx.sort(dim=-1)[0] == nearest_idx_kdtree.sort(dim=-1)[0])
                #except:
                #    pdb.set_trace()

                # find relative positional embedding
                knn_pos = idx[nearest_idx][..., 1:3]      # (num_rois, nearest_k, 2)
                knn_pos = torch.cat([idx[:, 1:3].unsqueeze(1), knn_pos], dim=1)     # (num_rois, nearest_k+1, 2)
                rel_pos = knn_pos.unsqueeze(1) - knn_pos.unsqueeze(2)    # (num_rois, nearest_k+1, nearest_k+1, 2)
                rel_pos += self.rel_pos_range - 1
                rel_pos_flat = rel_pos[..., 0] * (2 * self.rel_pos_range - 1) + rel_pos[..., 1]     # (num_rois, nearest_k+1, nearest_k+1)
                rel_pos_flat[rel_pos_flat < 0] = self.num_valid_rel_pos
                rel_pos_flat[rel_pos_flat >= self.num_valid_rel_pos] = self.num_valid_rel_pos
                rel_pos_bias = self.rel_pos_bias_table[rel_pos_flat].permute(0, 3, 1, 2)    # (num_rois, heads, nearest_k+1, nearest_k+1)
        else:
            nearest_idx = None
            rel_pos_bias = None
        
        # run roi head
        with torch.autograd.profiler.record_function("roi_head"):    # code context for profiler
            roi_enc_outs, roi_dec_outs = self.roi_head(rgbm_rois, rel_pos_bias, nearest_idx)
        roi_matte = self.roi_matte_head(roi_dec_outs[-1]).clamp_(0, 1)

        # replace in source image with lowres_seg
        up_lowres_seg = F.interpolate(
            lowres_seg,
            scale_factor=self.downsample_rate,
            #mode='bilinear', align_corners=False,
            mode='nearest',
        )
        ## replace in source image with lowres_trimap
        #up_lowres_seg = F.interpolate(
        #    (torch.argmax(lowres_trimap, dim=1) == 2).float().unsqueeze(1),
        #    scale_factor=self.downsample_rate,
        #    mode='nearest',
        #)
        highres_matte = self.replace_region(up_lowres_seg, roi_matte, idx, 0)

        ret = dict()
        ret['highres_matte'] = highres_matte

        # create a sampling mask image for visualization
        with torch.no_grad():
            sampling_mask = torch.zeros_like(lowres_seg)
            sampling_mask[idx[:, 0], :, idx[:, 1], idx[:, 2]] = 1
        ret['sampling_mask'] = sampling_mask

        # debug
        ret['roi_matte'] = roi_matte
        if 'highres_matte_gt' in inputs.keys():
            ret['roi_matte_gt'] = self.crop_regions(inputs['highres_matte_gt'], idx, self.downsample_rate, 0)
        
        return ret