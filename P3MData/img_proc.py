import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

import kornia
import cv2
import numpy as np
import random
import math
import functools

import sys
sys.path.append('..')
from utils import check_cuda_memory_error
sys.path.pop()

import pdb

def make_coordinate_grid_2d(spatial_shape, dtype, device, normalize):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_shape.

    The reason to use normalize=True is that F.grid_sample() requires input sampling gird to be in [-1, 1]
    """
    h, w = spatial_shape
    y = torch.arange(h, dtype=dtype, device=device)
    x = torch.arange(w, dtype=dtype, device=device)

    if normalize: 
        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)
       
    yy, xx = torch.meshgrid(y, x)
    meshed = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
    return meshed

def get_affine_matrix(center, angle, translate, scale, shear):
    """
    modified from torchvision.transforms.functional._get_inverse_affine_matrix
    """
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    matrix = [a, b, 0.0, c, d, 0.0]
    matrix = [x * scale for x in matrix]
    # Apply inverse of center translation: RSS * C^-1
    matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    # Apply translation and center : T * C * RSS * C^-1
    matrix[2] += cx + tx
    matrix[5] += cy + ty

    matrix = np.array(matrix + [0, 0, 1], dtype=np.float32).reshape(3, 3)      # (3, 3)
    return matrix

def gen_trimap_with_dilation(alpha, kernel_size):	
    kernel = torch.as_tensor( cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), dtype=torch.float32, device=alpha.device )
    ndim = alpha.ndim
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(0)
    fg_and_unknown = (alpha != 0).to(torch.float32)
    fg = (alpha == 1).to(torch.float32)
    dilate =  kornia.morphology.dilation(fg_and_unknown, kernel)
    erode = kornia.morphology.erosion(fg, kernel)
    trimap = erode * 2 + (dilate - erode) * 1
    if ndim == 3:
        trimap = trimap.squeeze(0)
    return trimap

def gen_trimap_with_threshold(alpha, kernel_size, thres):	
    kernel = torch.as_tensor( cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), dtype=torch.float32, device=alpha.device )
    ndim = alpha.ndim
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(0)
    fg = alpha > 0.5 + thres
    #bg = alpha < 0.5 - thres
    unknown = torch.abs(alpha - 0.5) <= thres
    fg_and_unknown = torch.logical_or(fg, unknown)
    dilate =  kornia.morphology.dilation(fg_and_unknown.float(), kernel)
    erode = kornia.morphology.erosion(fg.float(), kernel)
    trimap = erode * 2 + (dilate - erode) * 1
    if ndim == 3:
        trimap = trimap.squeeze(0)
    return trimap

class ColorJitter(torchvision.transforms.ColorJitter):
    """
    compared to torchvision.transforms.ColorJitter, this class can cache the previous params used for color jitter.
    """
    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
        return fn_idx, b, c, s, h

    def forward(self, img, transform_params=None):
        if transform_params is None:
            transform_params = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transform_params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue(img, hue_factor)
        return img

class BaseImageProcessor():
    def __init__(self, device, **kwargs):
        self.device = device

        for k, v in kwargs.items():
            assert hasattr(self, '_{}'.format(k)), "No attribute named {} found".format(k)
            setattr(self, '_{}'.format(k), v) 

    def __call__(self, x):
        """
        img: np.ndarray, uint8, 0-255
        matte: np.ndarray, uint8, 0-255
        """
        #print(f"calling ImageProcessor on device: {self.device}")
        for k in ['img', 'bg_img', 'matte', 'fg', 'bg']:
            if k not in x.keys():
                continue
            im = x[k]
            assert isinstance(im, np.ndarray) and im.dtype == np.uint8
            if im.ndim == 2:
                im = im[..., np.newaxis]
            im = torch.as_tensor(im / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

            # BGR to RGB
            if k in ['img', 'bg_img', 'fg', 'bg']:
                im[...] = im[[2, 1, 0], ...]    # do in-place update to save memory
            
            x[k] = im

        # compute trimap online
        if 'matte' in x.keys() and 'trimap' not in x.keys():
            #kernel_size = random.randint(15, 30)
            trimap = gen_trimap_with_dilation(x['matte'], kernel_size=7)
            #trimap = gen_trimap_with_threshold(x['matte'], kernel_size=5, thres=0.499)
            x['trimap'] = trimap

        return x

class P3MImageProcessor(BaseImageProcessor):

    _num_tps_points = 5
    _tps_sigma = 0.007

    _bgt = 0.2
    _cst = 0.1
    _sat = 0.1
    _hue = 0.1

    def __init__(self, img_size, device, **kwargs):
        super(P3MImageProcessor, self).__init__(device, **kwargs)

        self.img_size = img_size
        self.dst_pts = make_coordinate_grid_2d(self.img_size[::-1], torch.float32, self.device, normalize=True).view(-1, 2)  # xy of shape (h*w, 2)
        self.control_pts = make_coordinate_grid_2d((self._num_tps_points, self._num_tps_points), torch.float32, self.device, normalize=True).view(-1, 2)  # xy of shape (num_tps*num_tps, 2)

        self.color_jit = ColorJitter(brightness=self._bgt, contrast=self._cst, saturation=self._sat, hue=self._hue) 

    def get_transform(self, trimap):
        raise NotImplementedError("To be implemented in subclass.")
    
    def get_deform_grid(self, ori_size, trans_matx):
        w, h = ori_size

        # adapt transform matrix to normalized coordinates in [-1, 1]
        src_norm_matx = torch.tensor([[2 / (w - 1), 0, -1], [0, 2 / (h - 1), -1], [0, 0, 1]], dtype=torch.float32, device=trans_matx.device)
        dst_norm_matx = torch.tensor([[2 / (self.img_size[0] - 1), 0, -1], [0, 2 / (self.img_size[1] - 1), -1], [0, 0, 1]], dtype=torch.float32, device=trans_matx.device)
        trans_matx = torch.matmul(dst_norm_matx, torch.matmul(trans_matx, torch.inverse(src_norm_matx)))

        # do all transformation
        trans_matx = torch.inverse(trans_matx).view(1, 3, 3)
        transformed_pts = torch.matmul(trans_matx[:, :2, :2], self.dst_pts.unsqueeze(-1)) + trans_matx[:, :2, 2:]  # (num_pts, 2, 1)
        transformed_pts = transformed_pts.squeeze(-1)   # (num_pts, 2)
        #transformed_pts = F.affine_grid(trans_matx, [1, 3, self.img_size[1], self.img_size[0]]).view(-1, 2)
        
        # get TPS transformation
        control_params = torch.randn(1, self._num_tps_points ** 2).to(self.device)
        control_params *= self._tps_sigma
        d = self.dst_pts.view(-1, 1, 2) - self.control_pts.view(1, -1, 2)    # (h*w, num_tps * num_tps, 2)
        norm2 = torch.sum(d * d, dim=-1)  # (h*w, num_tps * num_tps)
        norm = torch.sqrt(norm2)
        kernel = norm2 * torch.log(norm + 1e-6)     # (h*w, num_tps * num_tps)
        tps_transformed = torch.sum(kernel * control_params, dim=-1, keepdim=True)    # (h*w, 1)
        transformed_pts = transformed_pts + tps_transformed

        grid = transformed_pts.view(1, self.img_size[1], self.img_size[0], 2)
        return grid

    @check_cuda_memory_error
    def __call__(self, x):
        """
        img: np.ndarray, uint8, 0-255
        matte: np.ndarray, uint8, 0-255
        """
        # make a copy so that the in-place modification later won't affect the original input
        # we need to retain the original input because this function will be called multiple times if cuda_memory_error is raised
        x = dict([(k, v.copy() if isinstance(v, np.ndarray) else v) for k, v in x.items()])

        x = super(P3MImageProcessor, self).__call__(x)

        jit_params = self.color_jit.get_params(self.color_jit.brightness, self.color_jit.contrast, self.color_jit.saturation, self.color_jit.hue)

        grid = None
        for k in ['img', 'bg_img', 'matte', 'trimap', 'fg', 'bg']:
            if k not in x.keys():
                continue

            im = x[k]
            assert isinstance(im, torch.Tensor)
            h, w = im.shape[1:]
            if grid is None:
                trans_matx = self.get_transform(x['trimap'] if 'trimap' in x.keys() else torch.zeros_like(x[k][:1]))
                grid = self.get_deform_grid((w, h), trans_matx)

            if k in ['img', 'bg_img', 'fg', 'bg']:
                # automatically choose when to do color jitter
                color_jit_done = False
                if h * w < self.img_size[0] * self.img_size[1]:
                    im[...] = self.color_jit(im, jit_params)    # do in-place update to save memory
                    color_jit_done = True

            im = F.grid_sample(im.unsqueeze(0), grid, mode='bilinear' if k != 'trimap' else 'nearest').squeeze(0)

            if k in ['img', 'bg_img', 'fg', 'bg']:
                if not color_jit_done:
                    im[...] = self.color_jit(im, jit_params)    # do in-place update to save memory
                    color_jit_done = True
            
            # free up memory ASAP
            del x[k]
            torch.cuda.empty_cache()

            x[k] = im
            
        return x

class P3MImageProcessorV1(P3MImageProcessor):
    """
    Follow exactly the same image preprocessing in the original P3M paper
    """
    _crop_sizes = [512, 768, 1024]
    #_crop_ratios = (0.7, 1)

    def get_transform(self, trimap):
        h, w = trimap.shape[1:]
        crop_size = random.choice(self._crop_sizes)
        crop_size = crop_size if crop_size < min(h, w) else 512
        #crop_ratio = random.uniform(self._crop_ratios[0], self._crop_ratios[1])
        #crop_size = int(crop_ratio * math.sqrt(h * w))
        if crop_size >= min(h, w):
            crop_size = min(h, w) - 1

        trimap_crop = trimap[0, :h-crop_size, :w-crop_size]
        target = torch.where(trimap_crop == 1) if random.random() < 0.5 else torch.where(trimap_crop > -100)
        if len(target[0])==0:
            target = torch.where(trimap_crop > -100)
        rand_ind = torch.randint(target[0].shape[0], size=(1,))[0]
        cropx, cropy = target[1][rand_ind], target[0][rand_ind]
        #print(f"img size: {(w, h)}, crop xy: {(cropx, cropy)}")

        crop_matx = torch.tensor([[1, 0, -cropx], [0, 1, -cropy], [0, 0, 1]], dtype=torch.float32, device=trimap.device)
        resize_matx = torch.tensor([[self.img_size[0]/crop_size, 0, 0], [0, self.img_size[1]/crop_size, 0], [0, 0, 1]], dtype=torch.float32, device=trimap.device)

        flip_flag =  random.random() < 0.5 
        if flip_flag:
            flip_matx = torch.tensor([[-1, 0, self.img_size[0]], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=trimap.device)
        else:
            flip_matx = torch.eye(3, dtype=torch.float32, device=trimap.device)

        trans_matx = functools.reduce(lambda x, y: torch.matmul(y, x), [crop_matx, resize_matx, flip_matx])
        return trans_matx
    
class P3MImageProcessorV2(P3MImageProcessor):
    """
    Follow image processing in BGMv2 paper
    """
    _degrees = (-5, 5)
    _translate = (0.1, 0.1)
    _scale = (0.5, 1.5)     # original paper uses (0.4, 1)
    _shear = (-5, 5)
    _aspect_ratio = (0.7, 1.3)

    def get_transform(self, trimap):
        h, w = trimap.shape[1:]

        # flip
        if np.random.uniform() < 0.5:
            flip_matx = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        else:
            flip_matx = np.eye(3, dtype=np.float32)
        
        # pad
        w_padded = max(w, self.img_size[0])
        h_padded = max(h, self.img_size[1])
        pad_h = (h_padded - h)
        pad_w = (w_padded - w)
        pad_matx = np.array([[1, 0, pad_w/2], [0, 1, pad_h/2], [0, 0, 1]], dtype=np.float32)

        # affine
        scale_factor = max(self.img_size[0] / w, self.img_size[1] / h)
        _scale = self._scale[0] * scale_factor, self._scale[1] * scale_factor
        _translate = self._translate[0] * scale_factor, self._translate[1] * scale_factor
        angle, translate, scale, shear = torchvision.transforms.RandomAffine.get_params(self._degrees, _translate, _scale, self._shear, (w_padded, h_padded))
        center = [w_padded*0.5, h_padded*0.5]
        affine_matx = get_affine_matrix(center, angle, translate, scale, shear)     # (3, 3)

        # aspect ratio
        ratio = np.random.uniform(self._aspect_ratio[0], self._aspect_ratio[1])
        affine_matx[0, 0] *= ratio

        # center crop of img_size
        crop_matx = np.array([[1, 0, -(w_padded-self.img_size[0])/2], [0, 1, -(h_padded-self.img_size[1])/2], [0, 0, 1]], dtype=np.float32)

        # compose matrices
        trans_matx = functools.reduce(lambda t1, t2: np.matmul(t2, t1), [flip_matx, pad_matx, affine_matx, crop_matx])
    
        return torch.as_tensor(trans_matx, dtype=torch.float32, device=self.device)