import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

import os
import cv2
import argparse
from shutil import rmtree, copytree
import numpy as np    
from tensorpack.utils.utils import logger

from P3MData.train_data import create_dataflow
from P3MData.utils import TP_Dataflow_to_PT_Dataset
from config import config as cfg
from trainer import Trainer
from utils import SummaryLogger, create_tqdm_bar, clean_folders, set_rand_seed

import traceback
import pdb

def setup_log_dir(args):
    # ============= set up logger
    if len(args.train_dir) > 0:
        train_dir = args.train_dir
    else:
        train_dir = os.path.join(cfg.TRAIN.DEFAULT_LOG_ROOT, args.train_folder)
    clean_folders([train_dir])
    log_dir = os.path.join(train_dir, 'logger')
    ## copy code
    #ignore_func = lambda d, files: [f for f in files if os.path.islink(os.path.join(d, f)) or (os.path.isfile(os.path.join(d, f)) and os.path.basename(f).split('.')[-1] not in ['py', 'so', 'cc', 'cpp', 'c', 'txt', 'json', 'sh'])]
    #copytree('.', os.path.join(train_dir, 'code'), ignore=ignore_func)

    return log_dir

def main(rank, args):
    # init process group
    if args.world_size > 1:
        dist.init_process_group(                                   
            backend='nccl',                                         
            world_size=args.world_size,                              
            rank=rank                                               
        )                   
        assert dist.is_available() and dist.is_initialized()
        logger.info("Distributed training initialized")
        #raise NotImplementedError("Something wrong with distributed training. Even if we initialize the group here, dist.is_initialized() returns False in data_loader.py")

    model_cfg = getattr(cfg, args.model_version.upper())
    train_cfg = getattr(cfg, 'TRAIN')

    # update config
    cfg.freeze(False)
    if len(args.config) > 0:
        cfg.update_args(args.config.split(';'))
    model_cfg.MODEL_VERSION = args.model_version.lower()
    cfg.freeze()
    if args.debug_run:
        cfg.freeze(False)
        train_cfg.MAX_EPOCH = 10
        train_cfg.STEPS_PER_EPOCH = 100
        train_cfg.SAVE_PER_K_EPOCHS = 1
        train_cfg.SUMMARY_PERIOD = 5
        train_cfg.LR_SCHEDULES = [120, 240]
        cfg.freeze()

    device0 = 'cuda:{}'.format(rank)

    # fix random seed
    set_rand_seed(train_cfg.SEED + rank)

    # ============== setup data loader
    #ds = VoxCeleb2Dataset('/data/yazhong/Datasets/VoxCeleb2/dev', train_cfg.IMG_SIZE, num_frames=2)
    if len(args.ipc_addr) > 0:
        assert args.world_size == 1, "Doesn't support shared dataflow from ZMQ for distributed multi gpu trainining."
        df = receive_dataflow(args.ipc_addr, train_cfg.IMG_SIZE, device0)
        ds = TP_Dataflow_to_PT_Dataset(df)
        # multi process data loading is incorporated in remote data generator, no need to use multi workers here
        data_loader = torch.utils.data.DataLoader(ds, batch_size=train_cfg.BATCH_SIZE, num_workers=0)   
    else:
        data_sources = [
            'P3M-10K',
            ]
        df = create_dataflow(data_sources, train_cfg.IMG_SIZE, train_cfg.IMG_PROC_VERSION, device0, train_cfg.BG_AUG)
        ds = TP_Dataflow_to_PT_Dataset(df)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=train_cfg.BATCH_SIZE, num_workers=4 if args.world_size == 1 else 0)

    # ============== setup model
    trainer = Trainer(train_cfg, model_cfg, rank, args.init_ckpt_file, args.init_lowres_ckpt_file)

    if args.reset_global_step >= 0:
        trainer.global_step = args.reset_global_step
    if args.reset_learning_rate is not None:
        g_lr, d_lr = args.reset_learning_rate
        trainer.set_g_lr(g_lr)
        trainer.set_d_lr(d_lr)                
    
    if rank == 0:
        logger.info("Config: ------------------------------------------\n" + str(cfg) + '\n')
        summary = SummaryLogger(args.log_dir) 
        pbar = create_tqdm_bar(total=train_cfg.STEPS_PER_EPOCH, desc='epoch 1')
    
    local_step = 0
    try:
        for data in data_loader:
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device0)

            with torch.autograd.set_detect_anomaly(args.debug_run):
                trainer.step(data)

            global_step = trainer.global_step
            epoch = global_step // train_cfg.STEPS_PER_EPOCH
            local_step = global_step % train_cfg.STEPS_PER_EPOCH
            epoch_done = (local_step == 0)

            # update learning rate
            if (global_step + 1) in train_cfg.LR_SCHEDULES:
                trainer.set_g_lr(trainer.get_g_lr() * 0.1) 
                trainer.set_d_lr(trainer.get_d_lr() * 0.1) 

            if rank == 0:
                pbar.update()

                # save summaries to tensorboard
                if (global_step + 1) % train_cfg.SUMMARY_PERIOD == 0:
                    trainer.dump_summaries()
                    summary.write_summaries(trainer.summary_collector, trainer.global_step, 5)

                if epoch_done:     #  current epoch done. next step will be new epoch
                    # save ckpt
                    if epoch % train_cfg.SAVE_PER_K_EPOCHS == 0:
                        ckpt_file = os.path.join(args.log_dir, '{:010d}.pth'.format(global_step))
                        trainer.save_ckpt(ckpt_file)

                    pbar.close()
                    pbar = create_tqdm_bar(total=train_cfg.STEPS_PER_EPOCH, desc='epoch {}'.format(epoch + 1))

                # save final ckpt before termination
                if epoch_done and epoch + 1 >= train_cfg.MAX_EPOCH:
                    ckpt_file = os.path.join(args.log_dir, '{:010d}.pth'.format(global_step))
                    trainer.save_ckpt(ckpt_file)
                    pbar.close()
                    logger.info("Max #epochs reached. Training finished.")
                    break
            
            # this is used to avoid memory explosion for multi worker data loading in DataLoader()
            torch.cuda.empty_cache()
    
    except RuntimeError as e:
        msg = str(e)
        if 'CUDA out of memory' in msg:
            logger.error('OOM error with gpu rank={}: '.format(rank) + str(e))
            logger.info("Don't interrupt. Waiting for model to be saved...")
            global_step = trainer.global_step
            ckpt_file = os.path.join(args.log_dir, '{:010d}-force_terminated.pth'.format(global_step))
            trainer.save_ckpt(ckpt_file)
        else:
            raise e

    except KeyboardInterrupt:
        if rank == 0:
            # always save ckpt upon key interruption
            logger.info("Don't interrupt. Waiting for model to be saved...")
            global_step = trainer.global_step
            ckpt_file = os.path.join(args.log_dir, '{:010d}-force_terminated.pth'.format(global_step))
            trainer.save_ckpt(ckpt_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version', default='model', help="Model version to use.")
    parser.add_argument('--ipc_addr', default='', help="ipc source address")
    parser.add_argument('--debug_run', default=False, type=bool, help="Whether to run the training for debug purposes.")
    parser.add_argument('--gpus', nargs='+', type=int, help="GPUs to use")
    parser.add_argument('--train_dir', default='', help="Directory where to write training log.")
    parser.add_argument('--train_folder', default='xx', help="If train_dir is not provided, this param will be used with DEFAULT_LOG_ROOT to create default train_dir. If train_dir is provided, this param will be ignored.")
    parser.add_argument('--config', default='', help="Config params. Each config param is separated by ';'.")
    parser.add_argument('--init_ckpt_file', default='', help="Init checkpoint file.")
    parser.add_argument('--init_lowres_ckpt_file', default='', help="Init checkpoint file for low resolution network.")
    parser.add_argument('--reset_global_step', default=-1, type=int, help="Whether to reset global step.")
    parser.add_argument('--reset_learning_rate', nargs=2, type=float, help="Whether to reset learning rate. First being g_lr, second being d_lr.")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    args.log_dir = setup_log_dir(args)

    # ============== set up for distributed trainining
    mp.set_start_method('spawn')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpus])
    args.world_size = len(args.gpus)
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'             
        os.environ['MASTER_PORT'] = '12345'      
        mp.spawn(main, nprocs=args.world_size, args=(args,))   
    else:
        main(0, args)