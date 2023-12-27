import torch
import torch.distributed as dist

import numpy as np
import lz4.frame
import cv2
import math
import os
import struct
from collections import OrderedDict, defaultdict, deque
import random
import glob
import json
from skimage.transform import estimate_transform, AffineTransform

from tensorpack.dataflow.base import DataFlow

import pdb

def distributed_partitioner(dataset):
    if dist.is_available() and dist.is_initialized():
        replica_id = dist.get_rank()
        num_replicas = dist.get_world_size()
    else:
        return 

    overall_start = dataset.start
    overall_end = dataset.end
    per_replica = int(math.ceil((overall_end - overall_start) / float(num_replicas)))
    dataset.start = overall_start + replica_id * per_replica
    dataset.end = min(dataset.start + per_replica, overall_end)

    print("{} has been partitioned by {} distributed training replicas. This instance is from {} to {}.".format(dataset.source, num_replicas, dataset.start, dataset.end))

def multi_process_load_partitioner(dataset):
    data_worker_info = torch.utils.data.get_worker_info()
    if data_worker_info is None:
        return

    overall_start = dataset.start
    overall_end = dataset.end

    # handle multi process data loading
    data_worker_id = data_worker_info.id
    num_data_workers = data_worker_info.num_workers

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(num_data_workers)))
    dataset.start = overall_start + data_worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

    print("{} has been partitioned by {} data loading processes. This instance is from {} to {}.".format(dataset.source, num_data_workers, dataset.start, dataset.end))

class BG20K(DataFlow):
    source = 'BG20K'

    def __init__(self, root):
        self.root = root

        self.img_files = glob.glob(os.path.join(root, '*.jpg'))
        
        self.start = 0
        self.end = len(self.img_files)

        print("Found {} images for BG20K.".format(len(self.img_files)))

        # distributed partitioner must be called in initializer, otherwise dist.is_initialized() always returns False even if distributed training is enabled
        distributed_partitioner(self)

    def __len__(self):
        raise NotImplementedError("This dataflow is indefinite and does not have a __len__() method.")

    def __iter__(self):
        multi_process_load_partitioner(self)
        sampling_source = self.img_files
        worker_sampling_source = sampling_source[self.start:self.end]

        while 1:
            random.shuffle(worker_sampling_source)
            for img_file in worker_sampling_source:
                img = cv2.imread(img_file)
                yield {'bg_img': img}

class P3M10KTrain(DataFlow):
    source = 'P3M-10K'

    def __init__(self, root, subset):
        self.root = root
        self.subset = subset
        self.img_files = glob.glob(os.path.join(root, subset, '**/blurred_image/*.jpg'), recursive=True)

        self.start = 0
        self.end = len(self.img_files)

        print("Found {} images for P3M-10K {}.".format(len(self.img_files), subset))

        # distributed partitioner must be called in initializer, otherwise dist.is_initialized() always returns False even if distributed training is enabled
        distributed_partitioner(self)
    
    def __len__(self):
        raise NotImplementedError("This dataflow is indefinite and does not have a __len__() method.")

    def __iter__(self):
        multi_process_load_partitioner(self)
        sampling_source = self.img_files
        worker_sampling_source = sampling_source[self.start:self.end]

        while 1:
            random.shuffle(worker_sampling_source)
            for img_file in worker_sampling_source:
                img = cv2.imread(img_file)
                mask_file = img_file.replace('original_image', 'mask').replace('blurred_image', 'mask').replace('.jpg', '.png')
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                fg_file = img_file.replace('original_image', 'fg').replace('blurred_image', 'fg')
                fg = cv2.imread(fg_file)
                bg_file = img_file.replace('original_image', 'bg').replace('blurred_image', 'bg')
                bg = cv2.imread(bg_file)
                yield {'img': img, 'fg': fg, 'bg': bg, 'matte': mask, 'source': self.source}

class P3M10KEval(DataFlow):
    def __init__(self, root, subset, use_privacy):
        self.root = root
        self.subset = subset
        if subset == 'train':
            self.img_files = glob.glob(os.path.join(root, subset, '**/blurred_image/*.jpg'), recursive=True)
        elif subset == 'validation':
            privacy = 'P' if use_privacy else 'NP'
            self.img_files = glob.glob(os.path.join(root, subset, f'P3M-500-{privacy}', '**/*.jpg'), recursive=True)

        print("Found {} images for P3M-10K {}.".format(len(self.img_files), subset))
    
    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        for img_file in sorted(self.img_files):
            img = cv2.imread(img_file)
            mask_file = img_file.replace('original_image', 'mask').replace('blurred_image', 'mask').replace('.jpg', '.png')
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            img_name = os.path.basename(img_file).replace('.jpg', '')
            res = {'img_name': img_name, 'img': img, 'matte': mask}
            if self.subset == 'validation':
                trimap_file = mask_file.replace('mask', 'trimap')
                trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)
                res['trimap'] = trimap

                #cv2.imwrite('tmp/trimap.jpg', res['trimap'])
                #pdb.set_trace()

            yield res

class VideoStreamLoader(DataFlow):
    def __init__(self, src_video):
        assert os.path.isfile(src_video) and src_video[-4:] in ['.mp4', '.avi'] or isinstance(src_video, int)
        self.src_video = src_video

        cap = cv2.VideoCapture(src_video)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

    def __len__(self):
        return self.num_frames
    
    def __iter__(self):
        cap = cv2.VideoCapture(self.src_video)
        assert cap.isOpened()
        while 1:
            ret, frame = cap.read()
            if ret is None or ret is False or frame is None:
                break
            yield {'img': frame}
        cap.release()

def debug():
    #gen = VideoMatte240K('/data/yazhong/Datasets/VideoMatte240K/Images', 'train')
    #gen = FaceSynthetics('/data/yazhong/Datasets/Errol_FaceSynthetics/dataset_1000000')
    #gen = TeamsFullRes('/data/yazhong/Datasets/fullres_Teams')
    gen = FullResBg('/data/yazhong/Datasets/FullResBackground')
    for dp in gen:
        img = dp['img']
        print('image shape', img.shape)
        fgr = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        cv2.imshow('img', img)
        if 'matte' in dp.keys():
            matte = dp['matte']
            matte = cv2.resize(matte, (0, 0), fx=0.25, fy=0.25)
            cv2.imshow('matte', matte)
        cv2.waitKey(0)

if __name__ == '__main__':
    debug()
