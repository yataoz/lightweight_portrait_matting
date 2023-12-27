import os
import pprint
import socket
import platform
import copy
import math
import pdb

class AttrDict():
    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        if name.startswith('_'):
            # Do not mess with internals. Otherwise copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """
        Update from command line args. 
        E.g., args = [TRAIN.BATCH_SIZE=1,TRAIN.INIT_LR=0.1]
        """
        assert isinstance(args, (tuple, list))
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config     # short alias to avoid coding

# training
_C.TRAIN.DEFAULT_LOG_ROOT = './train_root'
_C.TRAIN.SEED = 100
_C.TRAIN.MAX_EPOCH = 4
_C.TRAIN.LOSS_WEIGHTS = {
    'lowres_seg': 1,
    'lowres_trimap': 1,
    'highres_matte': 1,
    'highres_lap': 1,
    'highres_comp': 1,
    'discrim': 1,
    'distill': 0.1,
} 

_C.TRAIN.STEPS_PER_EPOCH = 100000
_C.TRAIN.SAVE_PER_K_EPOCHS = 1
_C.TRAIN.SUMMARY_PERIOD = 1000

_C.TRAIN.INIT_G_LR = 1.e-4
_C.TRAIN.INIT_D_LR = 1.e-4
_C.TRAIN.G_PERIOD = 1
_C.TRAIN.D_PERIOD = 1
_C.TRAIN.WEIGHT_DECAY = 1.e-6
_C.TRAIN.LR_SCHEDULES = [120000, 240000]

_C.TRAIN.IMG_SIZE = (896, 896)
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.BG_AUG = False
_C.TRAIN.GAN_LOSS = None
_C.TRAIN.IMG_PROC_VERSION = 'v2'

# Model
_C.MODEL.DOWNSAMPLE_RATE = 8
_C.MODEL.SPECTRAL_NORM = False
_C.MODEL.NORM_METHOD = 'BN'
_C.MODEL.ENFORCE_BIAS = True
_C.MODEL.D_LOGITS_LAYERS = [-1]
_C.MODEL.D_SCALES = ['x1']
_C.MODEL.D_DROPOUT = False
_C.MODEL.ENCODER_ARCH = 'Swin-T'
_C.MODEL.ROI_HEAD.ENABLE = True
_C.MODEL.ROI_HEAD.ADAPTIVE_SAMPLING = True
_C.MODEL.ROI_HEAD.TRANSFORMER.NEAREST_K = 8
_C.MODEL.ROI_HEAD.TRANSFORMER.ENABLE = True
_C.MODEL.ROI_HEAD.TRANSFORMER.REL_POS_EMBED = False
_C.MODEL.ROI_HEAD.TRANSFORMER.REL_POS_RANGE = 4
_C.MODEL.ROI_HEAD.UNFOLD_SIZE = 1
_C.MODEL.INPUT_RATE = 1
_C.MODEL.INPUT_RESOLUTION = 896
_C.MODEL.PATCH_SIZE = 16
_C.MODEL.D2S = True
_C.MODEL.SHRINK_RATIO = 1

# testing
_C.TEST.DEFAULT_LOG_ROOT = './train_root'
_C.TEST.SEED = 100
_C.TEST.PROFILE.ENABLED = True
_C.TEST.PROFILE.PRECISION = 'fp32'

_C.TEST.VIDEO_DEMO.IMG_SIZE = (512, 512)
_C.TEST.VIDEO_DEMO.BATCH_SIZE = 1
_C.TEST.VIDEO_DEMO.FORCE_MULTIPLES_OF_32 = True

_C.TEST.BENCHMARK.MAX_AREA = None
_C.TEST.BENCHMARK.RESIZE_FACTOR = 1

_C.freeze()
