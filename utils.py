from torch.utils.tensorboard import SummaryWriter
import torch
import os
import cv2 
from shutil import rmtree, copytree
import sys
import glob
import random
import numpy as np
from math import cos, sin
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools
import time

from tensorpack.utils.utils import get_tqdm_kwargs, fix_rng_seed, logger
from tensorpack.utils.viz import stack_patches
import tqdm

import pdb

matplotlib.use('Agg')   # for plt to work with ssh

def create_tqdm_bar(total, **kwargs):
    tqdm_args = get_tqdm_kwargs(leave=True, **kwargs)
    bar = tqdm.trange(total, **tqdm_args)
    return bar

def get_latest_checkpoint(log_dir):
    ckpt_files =[f for f in glob.glob(os.path.join(log_dir, '*.pth'))]
    ckpt_files =sorted(ckpt_files)
    return ckpt_files[-1]

def clean_folders(folders, force=False):
    for folder in folders:
        if os.path.exists(folder):
            if force:
                rmtree(folder)
            else:
                print("Are you sure you want to delete the folder {}?".format(folder))
                choice = input("\033[93mPress y to delete and n to cancel.\033[0m\n")
                if choice == 'y':
                    rmtree(folder)
                else:
                    sys.exit()

def set_rand_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    fix_rng_seed(seed)

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def draw_points_on_image(img, points, radius, color, thickness):
    for x, y in points:
        x = int(round(x))
        y = int(round(y))
        cv2.circle(img, (x, y), radius, color, thickness)
    return img

def check_cuda_memory_error(func):
    @functools.wraps(func)
    def inner_func(*args, **kwargs):
        num_trials = 10
        wait_time = 2   # in seconds
        cnt = 0
        while 1:
            try:
                if cnt < num_trials:
                    return func(*args, **kwargs)
                else:
                    raise RuntimeError("Checking CUDA out of memory error reached max number of trials.")
            except RuntimeError as e:
                msg = str(e)
                if 'CUDA out of memory' in msg and 'max number of trials' not in msg:
                    logger.error("Got CUDA out of memory error. {} trials failed. Wait for {}s for next trial...".format(cnt, wait_time))
                    time.sleep(wait_time)
                else:
                    raise e
            else:   # terminate when successfully run func() without error
                break
            finally:
                cnt += 1    # increase cnt whether it's a failed or successful run
    return inner_func

class SummaryCollector():
    def __init__(self):
        self.collections = defaultdict(list)

    def add(self, summary_type, name, value, append_if_exists=True):
        assert summary_type in ['scalar', 'image', 'histogram', 'tensor', 'meta']
        if name in self.collections.keys() and not append_if_exists:
            raise RuntimeError("An item with name={} already exists in the collection. You can't add a new item with the same name when 'append_if_exists' is disable.".format(name))
        self.collections[name].append({'summary_type': summary_type, 'value': value})

    def clear(self):
        # only clear collections
        self.collections.clear()

class BaseSummaryLogger():
    def __init__(self, log_dir):
        self.logger = SummaryWriter(log_dir)
    
    def write_graph(self, graph, inputs):
        self.logger.add_graph(graph, inputs)

    def write_summaries(self, *args, **kwargs):
        raise NotImplementedError("To be implemented in sublcass.")

    def close(self):
        self.logger.close()

    def __del__(self):
        self.logger.close()

class SummaryLogger(BaseSummaryLogger):
    def write_summaries(self, collector, global_step, first_num_imgs=None):
        """
        collector: collector created in the forward pass of a model

        first_num_img: only show first several imgs
        """
        assert isinstance(collector, SummaryCollector)

        # simple logging for existing primitives such as scalar and histogram
        data = collector.collections

        for name, item_list in data.items():
            for i, item in enumerate(item_list):
                summary_type, value = item['summary_type'], item['value']
                unq_name = '{}_{}'.format(name, i)
                if summary_type == 'scalar':
                    self.logger.add_scalar(unq_name, value, global_step=global_step)
                elif summary_type == 'histogram':
                    try:
                        self.logger.add_histogram(unq_name, value, global_step=global_step)
                    except Exception as e:
                        print(e)
                        print("\nContinue training...")
                elif summary_type == 'image':
                    item['value'] = value.permute(0, 2, 3, 1)   # channel first to channel last
                    #self.logger.add_image(unq_name, value)
                #elif summary_type == 'tensor':
                #    other_tensors[unq_name] = value

        data_sources = data['visuals/data_sources'][0]['value']
        pic_names = data['visuals/pic_names'][0]['value']
        highres_img_size = data['visuals/highres_img_size'][0]['value']
        figs = dict()
        inches_per_256_pix = 3
        for source in data_sources:
            bs, h, w, _ = data['visuals/{}/highres_img'.format(source)][0]['value'].shape
            show_bs = min(first_num_imgs, bs) if first_num_imgs else bs
            nrows = show_bs
            ncols = len(pic_names)
            fig, axarr = plt.subplots(nrows, ncols)
            axarr = axarr.reshape((nrows, ncols))
            fig.suptitle(source)
            fig.set_size_inches(inches_per_256_pix * (w // 256) * ncols, inches_per_256_pix * (h // 256) * nrows, forward=True)
            for i in range(nrows):
                for j in range(ncols):
                    name = pic_names[j]
                    bat_pics = data['visuals/{}/{}'.format(source, name)][0]['value'].detach().cpu().numpy()
                    pic = (bat_pics[i] * 255).astype(np.uint8)
                    pic[0, 0] = 0   # just to make sure matplotlit correct display pic when all pixels are 1 or 255. When all pixels in pic have the same value, it will show a black image
                    axarr[i, j].imshow(pic.squeeze(-1) if pic.shape[-1] == 1 else pic)
                    if i == 0:
                        axarr[i, j].set_title(name)
            figs[source] = fig

        for source, fig in figs.items():
            self.logger.add_figure(source, fig, global_step=global_step)

        plt.close('all')
        self.logger.flush()
