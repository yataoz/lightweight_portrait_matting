import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import FlopCountAnalysis, parameter_count

import os

from common import ModelContainer
from P3MData.img_proc import gen_trimap_with_dilation, gen_trimap_with_threshold

import pdb

class FullGenerator(nn.Module):
    def __init__(self, mat_model, test_cfg):
        super(FullGenerator, self).__init__()
        self.test_cfg = test_cfg
        self.mat_model = mat_model
        self.mat_model.eval()

    def forward(self, inputs):
        """
        highres_img: (b, 3, h, w)

        input image is channel first and must have been normalized to [0, 1] already.
        """
        highres_img = inputs['img']
        if hasattr(self.mat_model, 'input_rate'):
            lowres_img = F.interpolate(highres_img, scale_factor=1./self.mat_model.input_rate, mode='area')
        else:
            lowres_img = F.interpolate(highres_img, scale_factor=1./self.mat_model.downsample_rate, mode='area')

        with torch.autograd.profiler.record_function("model_inference"):    # code context for profiler
            inputs['highres_img'] = highres_img
            inputs['lowres_img'] = lowres_img
            ret = self.mat_model(inputs)

        ret['lowres_img'] = lowres_img

        return ret

class TracingWrapper(nn.Module):
    """
    A wrapper class that make model take tuple of tensors as input and tuple of tensors as output.
    """
    def __init__(self, full_g):
        super(TracingWrapper, self).__init__()
        self.full_g = full_g
    
    def forward(self, inputs):
        assert isinstance(inputs, (tuple, list))
        dict_inputs = {'img': inputs[0]}
        dict_outputs = self.full_g(dict_inputs)
        outputs = [v for v in dict_outputs.values()]
        return outputs
    

class Tester():
    def __init__(self, test_cfg, model_cfg, gpu_id, init_ckpt_file='', init_lowres_ckpt_file=''):
        self.test_cfg = test_cfg
        self.model_cfg = model_cfg

        self.global_step = 0

        # create models
        self.container = ModelContainer(model_cfg, 'test')
        device0 = 'cuda:{}'.format(gpu_id) if gpu_id is not None and gpu_id >= 0 else 'cpu'
        self.device0 = device0

        self.full_g = FullGenerator(
            self.container.mat_model, 
            test_cfg)
        self.full_g.to(device0)

        self.wrap_g = TracingWrapper(self.full_g)
        self.wrap_g.to(device0)

        # need to load a checkpoint from sparse training to figure out the weights to prune
        if hasattr(model_cfg, 'PRUNE') and model_cfg.PRUNE.ENABLE:
            assert os.path.isfile(init_ckpt_file)
            self.load_ckpt(init_ckpt_file, map_location=device0)
            self.container.mat_model.prune()
        # must initialize model before calling DataParallel()
        elif len(init_ckpt_file) > 0 and os.path.isfile(init_ckpt_file):
            self.load_ckpt(init_ckpt_file, map_location=device0)

            if len(init_lowres_ckpt_file) > 0 and os.path.isfile(init_lowres_ckpt_file):
                self.load_lowres_ckpt(init_lowres_ckpt_file, map_location=device0)

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

    def load_lowres_ckpt(self, lowres_ckpt_file, map_location):
        assert lowres_ckpt_file[-3:] == 'pth'
        assert os.path.isfile(lowres_ckpt_file)
        state_dict = torch.load(lowres_ckpt_file, map_location=map_location)

        # ignore buffer 'attn_mask', which is not trainable and input shape independent
        new_state_dict = self.container.mat_model.state_dict()
        for k, v in new_state_dict.items():
            if 'attn_mask' not in k:
                new_state_dict[k] = state_dict['mat_model'][k]
        
        for name in ['lowres_model', 'seg_heads', 'trimap_heads']:
            getattr(self.container.mat_model, name).load_state_dict(dict([(k.replace(name+'.', ''), v) for k, v in new_state_dict.items() if name in k]))

        print("Lowres model successfully loaded from {}".format(lowres_ckpt_file))
    
    def run(self, inputs):
        self.global_step += 1
        return self.full_g(inputs)
    
    def count_flops(self, inputs):
        list_inputs = [inputs['img']]
        flops = FlopCountAnalysis(self.wrap_g, list_inputs)
        return flops.total()

        ## debug
        #debug_size = 224
        #list_inputs = [F.interpolate(inputs['img'], (debug_size, debug_size))]
        #try:
        #    eval('self.full_g.mat_model.encoder.model.update_input_resolution(debug_size)')
        #except torch.nn.modules.module.ModuleAttributeError:   # non existing udate_input_resolution() function
        #    pass
        #flops = FlopCountAnalysis(self.wrap_g, list_inputs)
        #aa = flops.by_module()
        #pdb.set_trace()
    
    def count_params(self):
        return parameter_count(self.full_g)[""]