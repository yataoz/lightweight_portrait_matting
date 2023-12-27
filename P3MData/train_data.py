import torch
import tensorpack as tp
import numpy as np
import cv2 
import os
import functools
import itertools
import random

from .img_proc import BaseImageProcessor, P3MImageProcessorV1, P3MImageProcessorV2
from .data_loader import P3M10KTrain, BG20K
from .utils import RandomSwitch

import pdb

def create_dataflow(data_sources, img_size, img_proc_version, device, bg_aug):
    ImageProcessor = eval(f'P3MImageProcessor{img_proc_version.upper()}')

    all_sources ={
        'P3M-10K': {'path': './Datasets/P3M-10K', 'chance': 1}, 
    }

    if not isinstance(data_sources, (tuple, list)):
        data_sources = [data_sources]

    dfs = []
    chances = []
    for src in data_sources:
        if src not in all_sources:
            continue
        datasets = all_sources[src]
        if not isinstance(datasets, (tuple, list)):
            datasets = [datasets]
        
        for dataset in datasets:
            p = dataset['path']
            if 'P3M-10K' in p:
                df = P3M10KTrain(p, subset='train')
            else:
                raise ValueError("Unrecognized data source {}".format(p))
            
            dfs.append(df)
            chances.append(dataset['chance'])

    chances = [s / sum(chances) for s in chances]
    df = RandomSwitch(dfs, chances)
    img_processor = ImageProcessor(img_size, device)
    #img_processor = BaseImageProcessor(device)
    df = tp.dataflow.MapData(df, img_processor) 

    if bg_aug:
        bg_df = FullResBg('./Datasets/BG-20K/train')
        bg_processor = ImageProcessor(img_size, device)
        bg_df = tp.dataflow.MapData(bg_df, bg_processor)
        df = tp.dataflow.JoinData([df, bg_df])

    return df

def debug_Dataset():
    from skimage.transform import resize

    data_sources = [
        'P3M-10K',
        ]
    img_size = (512, 512)
    #device = 'cuda:7'
    device = 'cpu'
    bg_aug = False
    img_proc_version = 'V2'

    df = create_dataflow(data_sources, img_size, img_proc_version, device, bg_aug)
    df.reset_state()

    ## test speed
    #tp.dataflow.TestDataSpeed(df, size=1000).start()
    #return

    # visualize
    for dp in df:
        print("source: ", dp['source'])

        img = dp['img'] 
        img = cv2.normalize(img.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('tmp/img.jpg', img)
        #cv2.imshow('img', img)

        matte = dp['matte'] 
        matte = cv2.normalize(matte.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite('tmp/mask.jpg', matte)
        #cv2.imshow('matte', matte)

        trimap = dp['trimap'] 
        trimap = cv2.normalize(trimap.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite('tmp/trimap.jpg', trimap)

        bg = dp['bg'] 
        bg = cv2.normalize(bg.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite('tmp/bg.jpg', bg)

        fg = dp['fg'] 
        fg = cv2.normalize(fg.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
        cv2.imwrite('tmp/fg.jpg', fg)

        #bg = dp['bg_img'] 
        #bg = cv2.normalize(bg.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        ##cv2.imshow('bg', bg)

        #comp = dp['img'] * dp['matte'] + dp['bg_img'] * (1 - dp['matte'])
        #comp = cv2.normalize(comp.permute(1, 2, 0).detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
        #comp = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('tmp/img.jpg', comp)
        ##cv2.imshow('comp', comp)

        pyramid = [img]
        for s in [2, 4, 8]:
            #img_s = cv2.resize(img, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_AREA)
            img_s = resize(img.astype(np.float32) / 255, (img.shape[0] // s, img.shape[1] // s), anti_aliasing=True)
            img_s = cv2.resize(img_s, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            img_s = np.cast[np.uint8](img_s * 255) if img_s.dtype == np.float32 else img_s
            pyramid.append(img_s)
        pyramid = tp.utils.viz.stack_patches(pyramid, 2, 2, border=2)
        cv2.imwrite(f'tmp/pyramid.jpg', pyramid)

        #cv2.waitKey(0)
        pdb.set_trace()
    

if __name__ == '__main__':
    seed = 200
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    debug_Dataset()