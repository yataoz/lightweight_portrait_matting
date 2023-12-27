import torch 
from torch import nn
import functools
import importlib

from Model.layers import NORM, SN, BS
from Model.discriminator import MultiScaleDiscriminator

import pdb

class ModelContainer():
    def __init__(self, model_cfg, phase):
        assert phase in ['train', 'test']
        self.model_cfg = model_cfg
        self.phase = phase

        # check NORM & SN are consistent with preset
        assert self.model_cfg.SPECTRAL_NORM == SN
        assert self.model_cfg.NORM_METHOD == NORM
        assert self.model_cfg.ENFORCE_BIAS == BS

        # import corresponding model version
        module = importlib.import_module('Model.{}'.format(self.model_cfg.MODEL_VERSION))

        # create models
        self.mat_model = module.MattingModel(self.model_cfg, phase=self.phase)

        if self.phase == 'train':
            self.discriminator = MultiScaleDiscriminator(self.model_cfg.D_SCALES, self.model_cfg.D_LOGITS_LAYERS, self.model_cfg.D_DROPOUT) 
        else:
            self.mat_model.eval()

    @property
    def g_params(self):
        models = [self.mat_model]
        g_params = functools.reduce(
            lambda x, y: x + y, 
            [list(m.parameters()) for m in models],
        )  
        return g_params
    
    @property
    def d_params(self):
        if self.phase == 'test':
            raise ValueError("d_params not available when phase='test'.")
        d_params = self.discriminator.parameters()
        return d_params

    def state_dict(self):
        res = dict()
        res['mat_model'] = self.mat_model.state_dict()
        if self.phase == 'train':
            res['discriminator'] = self.discriminator.state_dict()
        return res

