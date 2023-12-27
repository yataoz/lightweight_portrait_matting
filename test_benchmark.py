import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import os
import cv2
import argparse
from shutil import rmtree, copytree
import numpy as np    
from skimage.transform import resize
import pandas as pd
from collections import defaultdict
from tensorpack.utils import viz

from config import config as cfg
from tester import Tester
from P3MData.data_loader import P3M10KEval
from PPMData.data_loader import PPM100Eval
from PhotoMatteData.data_loader import PhotoMatte85Eval
from utils import create_tqdm_bar, clean_folders, set_rand_seed, get_latest_checkpoint
from eval_metrics import *

import pdb

def inference_once(tester, scale_img, device, get_flops):
    tensor_img = torch.from_numpy(scale_img.astype(np.float32)).permute(2, 0, 1).to(device)
    inputs = dict()
    inputs['img'] = tensor_img.unsqueeze(0)
    if hasattr(tester.full_g.mat_model, 'input_rate'):
        input_shape = [s // tester.full_g.mat_model.input_rate for s in scale_img.shape[:2]]
    else:
        input_shape = scale_img.shape[:2]
    if hasattr(tester.full_g.mat_model, 'lowres_model'):
        m = tester.full_g.mat_model.lowres_model
    else:
        m = tester.full_g.mat_model
    m.encoder.model.update_input_resolution(input_shape)
    outputs = tester.run(inputs)
    if not isinstance(outputs['lowres_segs'], (list, tuple)):
        outputs['lowres_segs'] = [outputs['lowres_segs']]
    if not isinstance(outputs['lowres_trimaps'], (list, tuple)):
        outputs['lowres_trimaps'] = [outputs['lowres_trimaps']]
    flops = tester.count_flops(inputs) if get_flops else 0
    ret = [ 
        outputs['highres_matte'][0, 0].detach().cpu().numpy(), 
        outputs['lowres_segs'][-1][0, 0].detach().cpu().numpy(),
        outputs['lowres_trimaps'][-1][0].detach().cpu().numpy(),
        flops,
    ]
    return ret

def inference_img(tester, img, device, get_flops, multiples_base, resize_factor, max_area, keep_aspect_ratio=False):
    h, w, c = img.shape
    new_h = h * resize_factor
    new_w = w * resize_factor
    if max_area is not None and new_h * new_w > max_area:
        s = np.sqrt(max_area / (new_h * new_w))
        new_h *= s
        new_w *= s
    new_h = int(new_h)
    new_w = int(new_w)
    if keep_aspect_ratio:
        pad_h = multiples_base - new_h % multiples_base
        pad_w = multiples_base - new_w % multiples_base
        scale_img = resize(img,(new_h,new_w))   # output of resize is normalized to [0, 1]
        scale_img = np.pad(scale_img, [[pad_h//2, pad_h-pad_h//2], [pad_w//2, pad_w-pad_w//2], [0, 0]])
    else:
        _h = new_h - (new_h % multiples_base)
        _w = new_w - (new_w % multiples_base)
        best_size = _w, _h
        if max_area is not None:
            for ext_h, ext_w in [(1, 1), (1, 0), (0, 1)]:
                _h = new_h - (new_h % multiples_base) + ext_h * multiples_base
                _w = new_w - (new_w % multiples_base) + ext_w * multiples_base
                _area = _h * _w
                if (_area < max_area) and (abs(_area - max_area) < abs(best_size[0] * best_size[1] - max_area)):
                    best_size = (_w, _h)
        new_w, new_h = best_size
        scale_img = resize(img,(new_h,new_w))   # output of resize is normalized to [0, 1]
    highres_matte, lowres_seg, lowres_trimap, flops = inference_once(tester, scale_img, device, get_flops)
    if keep_aspect_ratio:
        highres_matte = highres_matte[pad_h:pad_h+h, pad_w:pad_w+w]
    highres_matte = resize(highres_matte,(h,w))
    return highres_matte, lowres_seg, lowres_trimap, new_h*new_w, flops

def main(args):
    model_cfg = getattr(cfg, args.model_version.upper())
    test_cfg = getattr(cfg, 'TEST')

    if len(args.src_train_dir) > 0:
        src_train_dir = args.src_train_dir
    else:
        src_train_dir = os.path.join(test_cfg.DEFAULT_LOG_ROOT, args.src_train_folder)

    log_dir = os.path.join(src_train_dir, f'test_{args.dataset}')
    #clean_folders([log_dir], force=True)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    ## copy code
    #if not args.profile_only:
    #    clean_folders([os.path.join(log_dir, 'code')], force=True)
    #    ignore_func = lambda d, files: [f for f in files if os.path.isfile(os.path.join(d, f)) and os.path.basename(f).split('.')[-1] not in ['py', 'so', 'cc', 'cpp', 'c', 'txt', 'json', 'sh']]
    #    copytree('.', os.path.join(log_dir, 'code'), ignore=ignore_func)

    # update config
    cfg.freeze(False)
    if len(args.config) > 0:
        cfg.update_args(args.config.split(';'))
    model_cfg.MODEL_VERSION = args.model_version.lower()
    cfg.freeze()
    print("Config: ------------------------------------------\n" + str(cfg) + '\n')

    # fix random seed
    set_rand_seed(test_cfg.SEED)

    # ============== setup model
    if args.gpus is None:
        args.gpus = [0]
    assert len(args.gpus) == 1
    #os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])
    init_ckpt_file = args.init_ckpt_file if len(args.init_ckpt_file) > 0 else get_latest_checkpoint(os.path.join(src_train_dir, 'logger'))
    tester = Tester(test_cfg, model_cfg, args.gpus[0], init_ckpt_file, args.init_lowres_ckpt_file)
    device0 = 'cuda:{}'.format(args.gpus[0])

    if args.dataset == 'P3M':
        dataset = 'P3M-500-{}'.format('P' if args.privacy_data else 'NP')
        df = P3M10KEval('./Datasets/P3M-10K', subset='validation', use_privacy=args.privacy_data)
    elif args.dataset == 'PPM':
        dataset = 'PPM-100'
        df = PPM100Eval('./Datasets/PPM-100')
    elif args.dataset == 'PhotoMatte':
        dataset = 'PhotoMatte85'
        df = PhotoMatte85Eval('./Datasets/PhotoMatte85')
    else:
        raise ValueError(f"Unrecognized dataset: {args.dataset}")
    if not args.profile_only:
        ckpt_iters = os.path.basename(init_ckpt_file).replace('.pth', '')
        if os.path.isfile(args.init_lowres_ckpt_file):
            lowres_ckpt = '-'.join(args.init_lowres_ckpt_file.replace('logger/', '').replace('.pth', '').split('/')[-2:])
            lowres_ckpt = f'_lowres-{lowres_ckpt}'
        else:
            lowres_ckpt = ''
        test_result_dir = os.path.join(log_dir, f'test_results-{ckpt_iters}{lowres_ckpt}_x{test_cfg.BENCHMARK.RESIZE_FACTOR}', dataset)
        if os.path.exists(test_result_dir):
            clean_folders([test_result_dir], force=True)
        os.makedirs(test_result_dir)


    multiples_base = 32 * (model_cfg.PATCH_SIZE // 4) * model_cfg.INPUT_RATE if 'Swin' in model_cfg.ENCODER_ARCH else 16 * 8

    pbar = create_tqdm_bar(total=len(df), desc=f'test {dataset}')

    metrics = defaultdict(list)
    count_ops = defaultdict(list)
    df.reset_state()
    with torch.autograd.profiler.profile(enabled=args.profile_only, use_cuda=True, record_shapes=True, profile_memory=True) as prof:
        for i, data in enumerate(df):
            #if args.profile_only:
            #    if i >= 10:
            #        break

            #torch.cuda.empty_cache()
            pbar.update()

            highres_matte, lowres_seg, lowres_trimap, pixels, flops = inference_img(
                tester, 
                cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB), 
                device0,
                args.profile_only and args.get_flops,   # note that get_flops will slow down the inference, resulting in incorrect inference speed
                multiples_base, 
                test_cfg.BENCHMARK.RESIZE_FACTOR, 
                test_cfg.BENCHMARK.MAX_AREA, 
            )

            if args.profile_only:
                count_ops['pixels'].append(pixels)
                count_ops['flops'].append(flops)
                continue

            else:
                lowres_trimap = np.argmax(lowres_trimap, axis=0)
                highres_matte_gt = data['matte'].astype(np.float32) / 255.
                trimap = data['trimap']
                img_name = data['img_name']

                sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(highres_matte, highres_matte_gt, trimap)
                sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(highres_matte, highres_matte_gt)
                sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(highres_matte, highres_matte_gt, trimap)
                conn_diff = compute_connectivity_loss_whole_image(highres_matte, highres_matte_gt)
                grad_diff = compute_gradient_whole_image(highres_matte, highres_matte_gt)

                metrics['img_name'].append(img_name)
                metrics['SAD'].append(sad_diff)
                metrics['MSE'].append(mse_diff)
                metrics['MAD'].append(mad_diff)
                metrics['SAD_TRIMAP'].append(sad_trimap_diff)
                metrics['MSE_TRIMAP'].append(mse_trimap_diff)
                metrics['MAD_TRIMAP'].append(mad_trimap_diff)
                metrics['SAD_FG'].append(sad_fg_diff)
                metrics['SAD_BG'].append(sad_bg_diff)
                metrics['CONN'].append(conn_diff)
                metrics['GRAD'].append(grad_diff)

                if args.save_imgs:
                    lowres = np.concatenate([
                        cv2.cvtColor((lowres_seg * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                        cv2.applyColorMap(cv2.normalize(lowres_trimap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1), cv2.COLORMAP_JET),
                    ], axis=1)
                    cv2.imwrite(os.path.join(test_result_dir, img_name+'_matte.jpg'), (highres_matte * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(test_result_dir, img_name+'_lowres.jpg'), lowres)
                    ## debug
                    #cv2.imwrite('tmp/highres_matte.jpg', (highres_matte * 255).astype(np.uint8))
                    #cv2.imwrite('tmp/lowres_mask.jpg', lowres)
                    #pdb.set_trace()

    pbar.close()

    # profile results
    if args.profile_only:
        print("\n")
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1, top_level_events_only=True))
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100, top_level_events_only=False))
        print(prof.key_averages().table(sort_by="cuda_time", row_limit=3, top_level_events_only=False))     # sort by avg cuda time

        print("\n")
        print("number of parameters: {:.2f}M".format(tester.count_params() / 1.e6)) 
        print("number of total pixels: {:.2f}M; number of average pixels per image: {:.2f}M".format(np.sum(count_ops['pixels']) / 1.e6, np.mean(count_ops['pixels']) / 1.e6))
        print("number of total flops: {:.2f}M; number of average flops per image: {:.2f}M".format(np.sum(count_ops['flops']) / 1.e6, np.mean(count_ops['flops']) / 1.e6))

        prof.export_chrome_trace(os.path.join(log_dir, 'profile_trace.json'))
    else:
        metrics['img_name'].append('avg')
        for k in metrics.keys():
            if k != 'img_name':
                metrics[k].append(sum(metrics[k]) / len(metrics[k]))

        result_csv = test_result_dir + '.csv'
        if os.path.exists(result_csv):
            os.remove(result_csv)
        pd.DataFrame(data=metrics).to_csv(result_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--src_train_dir', default='', help="Directory where to write training log.")
    parser.add_argument('--src_train_folder', default='xx', help="If train_dir is not provided, this param will be used with DEFAULT_LOG_ROOT to create default train_dir. If train_dir is provided, this param will be ignored.")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--init_lowres_ckpt_file', default='', help="Init checkpoint file for low resolution network.")
    parser.add_argument('--dataset', choices=['P3M', 'PPM', 'PhotoMatte'], help="Dataset to test with.")
    parser.add_argument('--privacy_data', type=int, default=1, help="Whether to use privacy subset of the P3M data.")
    parser.add_argument('--profile_only', type=int, default=0, help="Whether to profile the inference.")
    parser.add_argument('--get_flops', type=int, default=0, help="Whether to get flops from the inference.")
    parser.add_argument('--save_imgs', type=int, default=0, help="Whether to save the output as images.")
    parser.set_defaults(verbose=False)

    main(parser.parse_args()) 