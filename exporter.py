import torch
from torch import nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis, parameter_count

import os

from common import ModelContainer

import pdb

class TracingWrapperFull(nn.Module):
    def __init__(self, mat_model):
        super(TracingWrapperFull, self).__init__()
        self.mat_model = mat_model
    
    def forward(self, *inputs):
        with torch.no_grad():
            assert self.mat_model.enable_highres_head

            highres_img = inputs[0]
            if hasattr(self.mat_model, 'input_rate'):
                lowres_img = F.interpolate(highres_img, scale_factor=1./self.mat_model.input_rate, mode='area')
            else:
                lowres_img = F.interpolate(highres_img, scale_factor=1./self.mat_model.downsample_rate, mode='area')

            ret = self.mat_model({'highres_img': highres_img, 'lowres_img': lowres_img})

            return ret['highres_matte']

class TracingWrapperHighres(nn.Module):
    def __init__(self, mat_model):
        super(TracingWrapperHighres, self).__init__()
        self.mat_model = mat_model
    
    def forward(self, *inputs):
        with torch.no_grad():
            assert self.mat_model.enable_highres_head
            highres_img, lowres_seg, lowres_trimap = inputs
            ret = self.mat_model.forward_highres({'highres_img': highres_img, 'lowres_seg': lowres_seg, 'lowres_trimap': lowres_trimap})
            return ret['highres_matte']

class Exporter():
    def __init__(self, test_cfg, model_cfg, gpu_id, init_ckpt_file=''):
        self.test_cfg = test_cfg
        self.model_cfg = model_cfg

        self.global_step = 0

        # create models
        self.container = ModelContainer(model_cfg, 'test')
        device0 = 'cuda:{}'.format(gpu_id) if gpu_id is not None and gpu_id >= 0 else 'cpu'
        self.device0 = device0

        if len(init_ckpt_file) > 0 and os.path.isfile(init_ckpt_file):
            self.load_ckpt(init_ckpt_file, map_location='cpu')

        self.container.mat_model.eval().to(device0)

    def load_ckpt(self, ckpt_file, map_location):
        assert ckpt_file[-3:] == 'pth'
        assert os.path.isfile(ckpt_file)
        state_dict = torch.load(ckpt_file, map_location=map_location)

        ## ignore missing weights and mismatched shape from saved file
        #new_state_dict = self.container.mat_model.state_dict()
        #for k, v in new_state_dict.items():
        #    if k in state_dict['mat_model'].keys() and v.shape == state_dict['mat_model'][k].shape:
        #        new_state_dict[k] = state_dict['mat_model'][k]
        #    elif k not in state_dict['mat_model'].keys():
        #        print(f"\033[92m{k} is not available in state_dict of mat_model. Skipped.\033[0m")
        #    elif v.shape == state_dict['mat_model'][k].shape:
        #        print(f"\033[92mshape of {k} does not match the shape in state_dict of mat_model. Skipped.\033[0m")
        #self.container.mat_model.load_state_dict(new_state_dict)

        # ignore buffer 'attn_mask', which is not trainable and input shape independent
        new_state_dict = self.container.mat_model.state_dict()
        for k, v in new_state_dict.items():
            if 'attn_mask' not in k:
                new_state_dict[k] = state_dict['mat_model'][k]
        self.container.mat_model.load_state_dict(new_state_dict)

        #self.container.mat_model.load_state_dict(state_dict['mat_model'])

        print("Model successfully loaded from {}".format(ckpt_file))

    def export(self, input_size, dst_dir=None):
        """
        export full model
        """
        wrap_g = TracingWrapperFull(self.container.mat_model).eval()

        # better to use a real image as input because the refinement net depends on image content
        import cv2
        import numpy as np
        img = cv2.imread('assets/example.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(input_size))
        img = torch.as_tensor(img.astype(np.float32) / 255, dtype=torch.float32, device=self.device0).permute(2, 0, 1).unsqueeze(0)

        #img = torch.rand(1, 3, input_size[1], input_size[0], dtype=torch.float32).to(self.device0)

        if hasattr(wrap_g.mat_model, 'input_rate'):
            input_shape = [s // wrap_g.mat_model.input_rate for s in input_size[::-1]]
        else:
            input_shape = input_size[::-1]
        
        if hasattr(wrap_g.mat_model, 'lowres_model'):
            m = wrap_g.mat_model.lowres_model
        else:
            m = wrap_g.mat_model
        m.encoder.model.update_input_resolution(input_shape)
        
        script_model = torch.jit.trace(wrap_g, [img])
        
        if dst_dir is not None:
            model_path = os.path.join(dst_dir, f'full_model_{input_size[0]}x{input_size[1]}.pt' )
            script_model.save(model_path)
            print("Full model successfully exported to ", model_path)

        return script_model

    def export_highres(self, input_size, dst_dir=None):
        """
        export ROI extraction, ROI head and ROI replacement
        """
        wrap_g = TracingWrapperHighres(self.container.mat_model).eval()

        # better to use a real image as input because the refinement net depends on image content
        import cv2
        import numpy as np
        img = cv2.imread('assets/example.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = torch.as_tensor(img.astype(np.float32) / 255, dtype=torch.float32, device=self.device0).permute(2, 0, 1).unsqueeze(0)

        #img = torch.rand(1, 3, input_size[1], input_size[0], dtype=torch.float32).to(self.device0)

        if hasattr(wrap_g.mat_model, 'input_rate'):
            input_shape = [s // wrap_g.mat_model.input_rate for s in input_size[::-1]]
        else:
            input_shape = input_size[::-1]
        
        if hasattr(wrap_g.mat_model, 'lowres_model'):
            m = wrap_g.mat_model.lowres_model
        else:
            m = wrap_g.mat_model
        m.encoder.model.update_input_resolution(input_shape)

        lowres_outputs = wrap_g.mat_model.forward_lowres({'highres_img': img})
        
        script_model = torch.jit.trace(wrap_g, [img, lowres_outputs['lowres_segs'][-1], lowres_outputs['lowres_trimaps'][-1]])
        
        if dst_dir is not None:
            model_path = os.path.join(dst_dir, f'highres_model_{input_size[0]}x{input_size[1]}.pt' )
            script_model.save(model_path)
            print("Full model successfully exported to ", model_path)

        return script_model

    def count_flops(self, img):
        wrap_g = TracingWrapperFull(self.container.mat_model).eval()
        flops = FlopCountAnalysis(wrap_g, img)
        return flops.total()

    def count_params(self):
        wrap_g = TracingWrapperFull(self.container.mat_model).eval()
        num_params = parameter_count(wrap_g)[""]
        return num_params