import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import ColorJitter
import torch.distributed as dist

import numpy as np
import os
import functools
from scipy.special import comb
from collections import OrderedDict

from utils import SummaryCollector, check_cuda_memory_error
from common import ModelContainer
from losses import weighted_loss, appx_l1_loss, focal_loss, laplacian_loss

import pdb

# for debug use
def debug_print(x, name, color):
    print("\033[{}m {}: min={}, max={}, mean={} \033[0m".format(color, name, x.min(), x.max(), x.mean()))

def nan_inf_hook(self, inp, output):
    def iterate(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from iterate(v)
        elif isinstance(x, (tuple, list)):
            for v in x:
                yield from iterate(v)
        else:
            yield x

    for i, out in enumerate(iterate(output)):
        mask = torch.logical_or( torch.isnan(out) , torch.isinf(out) )
        if mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN/INF in output {i} at indices: ", mask.nonzero(), "where:", out[mask.nonzero()[:, 0].unique(sorted=True)])

class FullDiscriminator(nn.Module):
    def __init__(self, discriminator, train_cfg, summary_collector):
        super(FullDiscriminator, self).__init__()

        self.train_cfg = train_cfg
        self.discriminator = discriminator
        self.summary_collector = summary_collector

    def D_loss_fn(self, real_logit, fake_logit):
        if self.train_cfg.GAN_LOSS == 'Hinge':
            return F.relu(1 + fake_logit) + F.relu(1 - real_logit)      # convergence value ~2
        elif self.train_cfg.GAN_LOSS == 'LS':
            return torch.square(1 - real_logit) + torch.square(fake_logit)  # convergence value ~1
        else:
            raise NotImplementedError("Unrecognized GAN loss: {}".format(self.train_cfg.GAN_LOSS))

    def forward(self, img_pd, img_gt, cond, reso=None, name_scope='discrim'):
        loss_weights = self.train_cfg.LOSS_WEIGHTS

        if reso is None:
            # use multi-scale discriminator
            real_feats, real_logits = self.discriminator(img_gt, cond)
            fake_feats, fake_logits = self.discriminator(img_pd.detach(), cond)
        else:
            # use a discriminator at specific resolution
            real_feats, real_logits = self.discriminator[reso](img_gt, cond)
            fake_feats, fake_logits = self.discriminator[reso](img_pd.detach(), cond)

        # discriminator hinge loss 
        losses = []
        for real_logit, fake_logit in zip(real_logits, fake_logits):
            loss = torch.mean(self.D_loss_fn(real_logit, fake_logit)) * loss_weights['discrim']
            losses.append(loss)
            self.summary_collector.add('scalar', '{}/gan_loss_for_discrim'.format(name_scope), loss)
        gan_loss_for_discrim = sum(losses)
        return gan_loss_for_discrim

class FullGenerator(nn.Module):
    def __init__(self, container, train_cfg, summary_collector):
        super(FullGenerator, self).__init__()

        self.train_cfg = train_cfg
        self.summary_collector = summary_collector

        self.enable_d = self.train_cfg.GAN_LOSS not in ['None', None, False]

        self.mat_model = container.mat_model
        self.discriminator = container.discriminator

        self.teacher_model = getattr(container, 'teacher_model', None)
        self.dec_adapter = getattr(container, 'dec_adapter', None)
        self.roi_enc_adapter = getattr(container, 'roi_enc_adapter', None)
        self.roi_dec_adapter = getattr(container, 'roi_dec_adapter', None)
        
        ## debug 
        #for m in self.mat_model.modules():
        #    m.register_forward_hook(nan_inf_hook)

    def G_loss_fn(self, fake_logit):
        if self.train_cfg.GAN_LOSS == 'Hinge':
            return -fake_logit      # convergence value ~0
        elif self.train_cfg.GAN_LOSS == 'LS':
            return torch.square(1 - fake_logit)     # convergence value ~0.5
        else:
            raise NotImplementedError("Unrecognized GAN loss: {}".format(self.train_cfg.GAN_LOSS))
    
    def forward(self, inputs):
        """
        img: (b, 3, h, w)
        seg: (b, 1, h, w)
        """
        self.inputs = inputs
        if self.train_cfg.BG_AUG:
            inputs['img'][...] = inputs['img'] * inputs['matte'] + inputs['bg_img'] * (1 - inputs['matte'])     # do in-place update to save memory
            inputs['bg'] = inputs['bg_img']

        highres_img = inputs['img']
        highres_matte_gt = inputs['matte']
        highres_trimap_cls_gt = inputs['trimap'].long()
        highres_trimap_gt = F.one_hot(highres_trimap_cls_gt.squeeze(1), num_classes=3).permute(0, 3, 1, 2).float()    # categorical to one-hot probability

        lowres_img = F.interpolate(highres_img, scale_factor=1./self.mat_model.downsample_rate, mode='area')
        lowres_seg_gt = F.interpolate(highres_matte_gt, scale_factor=1./self.mat_model.downsample_rate, mode='area')
        lowres_trimap_gt = F.interpolate(highres_trimap_gt, scale_factor=1./self.mat_model.downsample_rate, mode='area')
        lowres_trimap_cls_gt = torch.argmax(lowres_trimap_gt, dim=1).unsqueeze(1)

        # run model
        ret = self.mat_model(
            {
                'lowres_img': lowres_img, 
                'highres_img': highres_img, 
                'highres_matte_gt': highres_matte_gt, 
            }
        )

        # losses
        loss_weights = self.train_cfg.LOSS_WEIGHTS
        self.losses = dict()

        # lowres seg loss
        # prefer binary cross entropy over L1 if possible
        if not isinstance(ret['lowres_segs'], (list, tuple)):
            ret['lowres_segs'] = [ret['lowres_segs']]
        if 'lowres_seg_logits' in ret.keys():
            if not isinstance(ret['lowres_seg_logits'], (list, tuple)):
                ret['lowres_seg_logits'] = [ret['lowres_seg_logits']]
            for i, lowres_seg_logit in enumerate(ret['lowres_seg_logits']):
                lowres_seg_logit = F.interpolate(lowres_seg_logit, lowres_seg_gt.shape[2:], mode='bilinear', align_corners=False) if lowres_seg_gt.shape[2:] != lowres_seg_logit.shape[2:] else lowres_seg_logit
                lowres_seg_loss = F.binary_cross_entropy_with_logits(lowres_seg_logit, lowres_seg_gt) * loss_weights['lowres_seg']
                self.losses[f'lowres_seg_loss_{i}'] = lowres_seg_loss
        else:
            for i, lowres_seg in enumerate(ret['lowres_segs']):
                lowres_seg = F.interpolate(lowres_seg, lowres_seg_gt.shape[2:], mode='bilinear', align_corners=False) if lowres_seg_gt.shape[2:] != lowres_seg.shape[2:] else lowres_seg
                lowres_seg_loss = appx_l1_loss(lowres_seg, lowres_seg_gt) * loss_weights['lowres_seg']
                self.losses[f'lowres_seg_loss_{i}'] = lowres_seg_loss

        ## lowres trimap loss
        cls_cnt = torch.bincount(lowres_trimap_cls_gt.view(-1)).float()
        cls_weight = cls_cnt ** (-0.5)
        cls_weight = cls_weight / (cls_weight.sum() + 1.e-8)
        if not isinstance(ret['lowres_trimaps'], (list, tuple)):
            ret['lowres_trimaps'] = [ret['lowres_trimaps']]
        if 'lowres_trimap_logits' in ret.keys():
            if not isinstance(ret['lowres_trimap_logits'], (list, tuple)):
                ret['lowres_trimap_logits'] = [ret['lowres_trimap_logits']]
            for i, lowres_trimap_logit in enumerate(ret['lowres_trimap_logits']):
                lowres_trimap_logit = F.interpolate(lowres_trimap_logit, lowres_seg_gt.shape[2:], mode='bilinear', align_corners=False) if lowres_seg_gt.shape[2:] != lowres_trimap_logit.shape[2:] else lowres_trimap_logit
                #lowres_trimap_loss = F.cross_entropy(
                #    lowres_trimap_logit,
                #    lowres_trimap_cls_gt.squeeze(1),
                #    cls_weight) * loss_weights['lowres_trimap']
                lowres_trimap_loss = focal_loss(
                    lowres_trimap_logit,
                    lowres_trimap_gt,
                    cls_weight) * loss_weights['lowres_trimap']
                self.losses[f'lowres_trimap_loss_{i}'] = lowres_trimap_loss
        
        if self.mat_model.enable_highres_head:
            ## highres matte loss
            unknown_id = 1
            unknown_mask = (highres_trimap_cls_gt == unknown_id)
            cls_weight = cls_cnt ** (-2)
            cls_weight = cls_weight / (cls_weight.sum() + 1.e-8)
            weight_mask = cls_weight[unknown_id] * unknown_mask.float() + (1 - cls_weight[unknown_id]) * torch.logical_not(unknown_mask)
            highres_matte_loss = weighted_loss(appx_l1_loss, ret['highres_matte'], highres_matte_gt, weight_mask) * loss_weights['highres_matte']
            self.losses['highres_matte_loss'] = highres_matte_loss

            # highres laplacian loss
            highres_lap_loss = laplacian_loss(ret['highres_matte'], highres_matte_gt, weight_mask) * loss_weights['highres_lap']
            self.losses['highres_lap_loss'] = highres_lap_loss

            # highres composition loss
            highres_comp = inputs['fg'] * ret['highres_matte'] + inputs['bg'] * (1 - ret['highres_matte'])
            highres_comp_loss = weighted_loss(appx_l1_loss, highres_comp, highres_img, weight_mask) * loss_weights['highres_comp']
            self.losses['highres_comp_loss'] = highres_comp_loss

            # debug
            if 'roi_matte' in ret.keys():
                self.summary_collector.add('histogram', 'histogram/roi_matte', ret['roi_matte'])
                self.summary_collector.add('histogram', 'histogram/roi_matte_gt', ret['roi_matte_gt'])

        # teacher supervision loss
        if self.teacher_model is not None:
            t_ret = self.teacher_model(highres_img, ret.get('roi_idx', None))

            # logits loss
            for i, (t_logits, s_logits) in enumerate(zip(t_ret['lowres_seg_logits'], ret['lowres_seg_logits'])):
                t_seg = t_ret['lowres_segs'][i]
                s_seg = ret['lowres_segs'][i]
                gt_seg = F.interpolate(lowres_seg_gt, s_seg.shape[2:], mode='area') if lowres_seg_gt.shape[2:] != s_seg.shape[2:] else lowres_seg_gt
                # get the mask where student prediction is worse than teacher prediction
                s_bce = F.binary_cross_entropy(s_seg, gt_seg)
                t_bce = F.binary_cross_entropy(t_seg, gt_seg)
                mask = (s_bce > t_bce).float().detach()
                s_log_prob = F.logsigmoid(s_logits)
                t_prob = t_seg
                # only learn from teacher if student prediction is worse
                self.losses[f'distill_lowres_seg_loss_{i}'] = weighted_loss(F.kl_div, s_log_prob, t_prob, mask) * loss_weights['distill']
            for i, (t_logits, s_logits) in enumerate(zip(t_ret['lowres_trimap_logits'], ret['lowres_trimap_logits'])):
                gt_trimap = F.interpolate(lowres_trimap_gt, s_logits.shape[2:], mode='area') if lowres_trimap_gt.shape[2:] != s_logits.shape[2:] else lowres_trimap_gt
                # get the mask where student prediction is worse than teacher prediction
                s_log_prob = F.log_softmax(s_logits, dim=1)
                t_log_prob = F.log_softmax(t_logits, dim=1)
                s_ce = torch.sum(-gt_trimap * s_log_prob, dim=1, keepdim=True)   # (B, 1, ...)
                t_ce = torch.sum(-gt_trimap * t_log_prob, dim=1, keepdim=True)   # (B, 1, ...)
                mask = (s_ce > t_ce).float().detach()
                t_prob = t_ret['lowres_trimaps'][i]
                # only learn from teacher if student prediction is worse
                self.losses[f'distill_lowres_trimap_loss_{i}'] = weighted_loss(F.kl_div, s_log_prob, t_prob, mask) * loss_weights['distill']

            if self.mat_model.enable_highres_head:
                t_dec_feats = t_ret['dec_feats']
                t_roi_enc_feats = t_ret['roi_enc_feats']
                t_roi_dec_feats = t_ret['roi_dec_feats']

                s_dec_feats = ret['dec_feats'][-len(t_dec_feats):] 
                s_roi_enc_feats = ret['roi_enc_feats'][-len(t_roi_enc_feats):]
                s_roi_dec_feats = ret['roi_dec_feats'][:-2] + ret['roi_dec_feats'][-1:]
                t_roi_dec_feats = t_roi_dec_feats[-len(s_roi_dec_feats):]

                t_dec_feats = self.dec_adapter(t_dec_feats)
                t_roi_enc_feats = self.roi_enc_adapter(t_roi_enc_feats)
                t_roi_dec_feats = self.roi_dec_adapter(t_roi_dec_feats)
                
                # features loss
                for i, (tx, sx) in enumerate(zip(t_dec_feats, s_dec_feats)):
                    self.losses[f'distill_lowres_dec_loss_{i}'] = F.mse_loss(sx, tx) * loss_weights['distill']
                for i, (tx, sx) in enumerate(zip(t_roi_enc_feats, s_roi_enc_feats)):
                    self.losses[f'distill_roi_enc_loss_{i}'] = F.mse_loss(sx, tx) * loss_weights['distill']
                for i, (tx, sx) in enumerate(zip(t_roi_dec_feats, s_roi_dec_feats)):
                    self.losses[f'distill_roi_dec_loss_{i}'] = F.mse_loss(sx, tx) * loss_weights['distill']

                # matte loss
                roi_matte_gt = ret['roi_matte_gt']
                s_roi_matte = ret['roi_matte']
                t_roi_matte = t_ret['roi_matte']
                mask = (F.l1_loss(s_roi_matte, roi_matte_gt) > F.l1_loss(t_roi_matte, roi_matte_gt)).float().detach()
                # only learn from teacher if student prediction is worse
                self.losses[f'distill_roi_matte_loss_{i}'] = weighted_loss(appx_l1_loss, s_roi_matte, t_roi_matte, mask) * loss_weights['distill']

                #self.losses['distill_lowres_feat_loss'] = F.mse_loss(ret['lowres_feat'], t_ret['lowres_feat']) * loss_weights['distill']

        g_loss = sum([l for l in self.losses.values()])

        ## debug 
        #for k, l in self.losses.items():
        #    print(k, l)

        # regularization
        reg_loss = 0
        for name, param in self.mat_model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.sum(torch.square(param))
        reg_loss = reg_loss * self.train_cfg.WEIGHT_DECAY
        self.losses['reg_loss'] = reg_loss
        g_loss += reg_loss
        
        self.losses['g_loss'] = g_loss

        # save output images
        self.visuals = OrderedDict()
        self.visuals['highres_img'] = highres_img
        self.visuals['lowres_img'] = lowres_img
        self.visuals['highres_matte_gt'] = highres_matte_gt
        self.visuals['lowres_seg'] = ret['lowres_segs'][-1]
        self.visuals['highres_trimap_gt'] = highres_trimap_gt
        self.visuals['lowres_trimap'] = ret['lowres_trimaps'][-1]
        if self.teacher_model is not None:
            self.visuals['teacher_lowres_seg'] = t_ret['lowres_segs'][-1]
            self.visuals['teacher_lowres_trimap'] = t_ret['lowres_trimaps'][-1]
        if self.mat_model.enable_highres_head:
            self.visuals['highres_matte'] = ret['highres_matte']
            if self.teacher_model is not None:
                self.visuals['teacher_highres_matte'] = t_ret['highres_matte']
            self.visuals['highres_comp'] = highres_comp
            if 'sampling_mask' in ret.keys():
                self.visuals['sampling_mask'] = ret['sampling_mask']


        return g_loss
        
    def dump_summaries(self):
        name_scope = 'losses'
        for k, loss in self.losses.items():
            self.summary_collector.add('scalar', '{}/{}'.format(name_scope, k), self.losses[k])

        name_scope = 'visuals'
        sources = set(self.inputs['source'])
        self.summary_collector.add('meta', '{}/data_sources'.format(name_scope), sources)
        pic_names = [k for k in self.visuals.keys()] 
        self.summary_collector.add('meta', '{}/pic_names'.format(name_scope), pic_names)
        self.summary_collector.add('meta', '{}/highres_img_size'.format(name_scope), self.visuals['highres_img'].shape[2:][::-1])

        for source in sources:
            select_mask = torch.tensor([s == source for s in self.inputs['source']], device=self.inputs['img'].device)
            #print("source {} ratio {}".format(source, select_mask.float().sum() / select_mask.shape[0]))
            for k, v in self.visuals.items():
                ## normalize some images for visualization
                #if k in ['lowres_seg_grad']:
                #    vmin, vmax = v.amin(dim=(2, 3), keepdim=True), v.amax(dim=(2, 3), keepdim=True)
                #    v = (v - vmin) / (vmax - vmin + 1.e-8)
                self.summary_collector.add('image', '{}/{}/{}'.format(name_scope, source, k), v[select_mask])

class Trainer():
    def __init__(self, train_cfg, model_cfg, rank, init_ckpt_file='', init_lowres_ckpt_file=''):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.rank = rank
        self.distributed = dist.is_available() and dist.is_initialized()

        self.enable_d = self.train_cfg.GAN_LOSS not in ['None', None, False]

        self.g_period = train_cfg.G_PERIOD
        self.d_period = train_cfg.D_PERIOD

        self.global_step = 0

        # create models
        self.summary_collector = SummaryCollector()
        self.container = ModelContainer(model_cfg, 'train')
        device0 = 'cuda:{}'.format(rank)

        self.full_g = FullGenerator(
            self.container,
            train_cfg, self.summary_collector)
        self.full_g.to(device0)

        self.full_d = FullDiscriminator(
            self.container.discriminator, 
            train_cfg, self.summary_collector)
        self.full_d.to(device0)

        self.g_optimizer = torch.optim.Adam(self.container.g_params, lr=train_cfg.INIT_G_LR, betas=(0.9, 0.999))
        self.d_optimizer = torch.optim.Adam(self.container.d_params, lr=train_cfg.INIT_D_LR, betas=(0.9, 0.999))

        # must initialize model before calling DataParallel()
        if len(init_ckpt_file) > 0 and os.path.isfile(init_ckpt_file):
            self.load_ckpt(init_ckpt_file, map_location=device0)

            if len(init_lowres_ckpt_file) > 0 and os.path.isfile(init_lowres_ckpt_file):
                self.load_lowres_ckpt(init_lowres_ckpt_file, map_location=device0)

        if self.distributed:
            self.full_g = nn.parallel.DistributedDataParallel(self.full_g, device_ids=[rank], find_unused_parameters=True)
            self.full_d = nn.parallel.DistributedDataParallel(self.full_d, device_ids=[rank], find_unused_parameters=True)

    def load_ckpt(self, ckpt_file, map_location):
        assert ckpt_file[-3:] == 'pth'
        assert os.path.isfile(ckpt_file)
        state_dict = torch.load(ckpt_file, map_location=map_location)

        ## ignore ROIHead; only init from lowres model
        #new_state_dict = self.container.mat_model.state_dict()
        #for k, v in new_state_dict.items():
        #    if 'roi' not in k:
        #        new_state_dict[k] = state_dict['mat_model'][k]
        #self.container.mat_model.load_state_dict(new_state_dict)

        # ignore buffer 'attn_mask', which is not trainable and input shape independent
        new_state_dict = self.container.mat_model.state_dict()
        for k, v in new_state_dict.items():
            if 'attn_mask' not in k:
                new_state_dict[k] = state_dict['mat_model'][k]
        self.container.mat_model.load_state_dict(new_state_dict)

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

        ## convert first 7x7 conv kernel to 3x3
        #new_state_dict = self.container.mat_model.state_dict()
        #for k, v in new_state_dict.items():
        #    if k in state_dict['mat_model'].keys() and v.shape == state_dict['mat_model'][k].shape:
        #        new_state_dict[k] = state_dict['mat_model'][k]
        #    elif k in state_dict['mat_model'].keys() and tuple(v.shape[:2]) == (7, 7) and tuple(state_dict['mat_model'][k].shape) == (3, 3):
        #        new_state_dict[k] = state_dict['mat_model'][k][2:5, 2:5]
        #self.container.mat_model.load_state_dict(new_state_dict)

        ## ignore missing weights and mismatched shape from saved file
        #new_state_dict = self.container.discriminator.state_dict()
        #for k, v in new_state_dict.items():
        #    if k in state_dict['discriminator'].keys() and v.shape == state_dict['discriminator'][k].shape:
        #        new_state_dict[k] = state_dict['discriminator'][k]
        #self.container.discriminator.load_state_dict(new_state_dict)
        #self.container.discriminator.load_state_dict(state_dict['discriminator'])

        ##self.container.mat_model.load_state_dict(state_dict['mat_model'])
        #if hasattr(self.container, 'dec_adapter'):
        #    self.container.dec_adapter.load_state_dict(state_dict['dec_adapter'])
        #if hasattr(self.container, 'roi_enc_adapter'):
        #    self.container.roi_enc_adapter.load_state_dict(state_dict['roi_enc_adapter'])
        #if hasattr(self.container, 'roi_dec_adapter'):
        #    self.container.roi_dec_adapter.load_state_dict(state_dict['roi_dec_adapter'])

        self.global_step = state_dict['global_step']
        #self.g_optimizer.load_state_dict(state_dict['g_optimizer'])
        #self.d_optimizer.load_state_dict(state_dict['d_optimizer'])
        
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

    def save_ckpt(self, ckpt_file):
        assert ckpt_file[-3:] == 'pth'
        state_dict = self.container.state_dict() 

        state_dict['global_step'] = self.global_step
        #state_dict['g_optimizer'] = self.g_optimizer.state_dict() 
        #state_dict['d_optimizer'] = self.d_optimizer.state_dict() 

        torch.save(state_dict, ckpt_file)
        print("Model successfully saved to {}".format(ckpt_file))

    @check_cuda_memory_error
    def step(self, inputs):
        self.summary_collector.clear()
        update_g = self.global_step % self.g_period == 0
        update_d = self.global_step % self.d_period == 0

        if update_g:
            g_loss = self.full_g(inputs)
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
        if update_d and self.enable_d:
            cond = torch.cat([inputs['img'], highres_seg], dim=1).detach()
            d_loss = self.full_d(highres_matte, inputs['matte'], cond)

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        self.global_step += 1
        g_lr = self.get_g_lr()
        d_lr = self.get_d_lr()
        self.summary_collector.add('scalar', 'optimizer/g_lr', g_lr)
        self.summary_collector.add('scalar', 'optimizer/d_lr', d_lr)
    
    def dump_summaries(self):
        if self.distributed:
            self.full_g.module.dump_summaries()
        else:
            self.full_g.dump_summaries()
    
    def set_g_lr(self, g_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = float(g_lr)

    def set_d_lr(self, d_lr):
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = float(d_lr)

    def get_g_lr(self):
        for param_group in self.g_optimizer.param_groups:
            g_lr = param_group['lr']
        return g_lr

    def get_d_lr(self):
        for param_group in self.d_optimizer.param_groups:
            d_lr = param_group['lr']
        return d_lr