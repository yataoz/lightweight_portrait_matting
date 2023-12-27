import torch
from torch import nn
import torch.nn.functional as F

import os
import cv2
import argparse
import numpy as np    

from config import config as cfg
from exporter import Exporter
from utils import get_latest_checkpoint, create_tqdm_bar
from PhotoMatteData.data_loader import PhotoMatte85Eval

#from fvcore.nn import FlopCountAnalysis, parameter_count
import cProfile
import pstats
import io
from pstats import SortKey
import time

import pdb

def main(args):
    model_cfg = getattr(cfg, args.model_version.upper())
    export_cfg = getattr(cfg, 'TEST')

    # update config
    cfg.freeze(False)
    if len(args.config) > 0:
        cfg.update_args(args.config.split(';'))
    model_cfg.MODEL_VERSION = args.model_version.lower()
    cfg.freeze()
    print("Config: ------------------------------------------\n" + str(cfg) + '\n')

    if len(args.src_train_dir) > 0:
        src_train_dir = args.src_train_dir
    else:
        src_train_dir = os.path.join(export_cfg.DEFAULT_LOG_ROOT, args.src_train_folder)
    init_ckpt_file = args.init_ckpt_file if len(args.init_ckpt_file) > 0 else get_latest_checkpoint(os.path.join(src_train_dir, 'logger'))

    if args.gpus is None:
        args.gpus = [0]
    assert len(args.gpus) == 1
    device = f'cuda:{args.gpus[0]}'

    exporter = Exporter(export_cfg, model_cfg, args.gpus[0], init_ckpt_file)
    input_size = [int(s) for s in args.input_size]
    script_model = exporter.export(input_size)

    # Creating profile object
    ob = cProfile.Profile()

    df = PhotoMatte85Eval('./Datasets/PhotoMatte85')
    df.reset_state()

    pbar = create_tqdm_bar(total=len(df), desc=f'profile on PhotoMatte85')

    times = []
    flops = []
    for i, data in enumerate(df):
        if args.max_imgs is not None and i >= args.max_imgs:
            break

        pbar.update()

        img = cv2.resize(data['img'], tuple(input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_t = torch.tensor(img.astype(np.float32) / 255, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

        if i >= args.warmup_imgs:
            ob.enable()
            t0 = time.time()

        # for fps
        outputs = script_model(img_t)

        if i >= args.warmup_imgs:
            t1 = time.time()
            times.append(t1 - t0)

        # for flops
        if args.get_flops:
            flops.append( exporter.count_flops() )

    pbar.close()

    ob.disable()
    sec = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    ps.print_stats()
    print("\033[92m\ncProfile results:\n\033[0m", sec.getvalue())
    print("\033[92m\ntime.time() results:\n\033[0m", f"avg time per inference: {sum(times) / len(times)}; fps: {len(times) / sum(times)}")

    num_params = exporter.count_params()
    print("\033[92m\nNumber of parameters: {}\n\033[0m".format(num_params))

    if args.get_flops:
        print("\033[92m\nNumber of total flops: {:.2f}M; number of average flops per image: {:.2f}M\n\033[0m".format(np.sum(flops) / 1.e6, np.mean(flops) / 1.e6))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--src_train_dir', default='', help="Directory where to write training log.")
    parser.add_argument('--src_train_folder', default='xx', help="If train_dir is not provided, this param will be used with DEFAULT_LOG_ROOT to create default train_dir. If train_dir is provided, this param will be ignored.")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--init_lowres_ckpt_file', default='', help="Init checkpoint file for low resolution network.")
    parser.add_argument('--input_size', nargs=2, help="Target test image resolution (w, h).")
    parser.add_argument('--max_imgs', type=int, default=100, help="Max number of test images.")
    parser.add_argument('--warmup_imgs', type=int, default=10, help="Number of test images used for warmup before doing the actual profiling.")
    parser.add_argument('--get_flops', type=int, default=0, help="Whether to get flops from the inference.")
    parser.set_defaults(verbose=False)

    main(parser.parse_args()) 