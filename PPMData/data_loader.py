import numpy as np
import cv2
import os
import glob
import tqdm

from tensorpack.dataflow.base import DataFlow
from tensorpack.utils.utils import logger

import pdb

def gen_trimap_with_dilation(alpha, kernel_size):	
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_and_unknown = (alpha != 0).astype(np.uint8)
    fg = (alpha == 255).astype(np.uint8)
    dilate =  cv2.dilate(fg_and_unknown, kernel)
    erode = cv2.erode(fg, kernel)
    trimap = erode * 2 + (dilate - erode) * 1
    return trimap


class PPM100Eval(DataFlow):
    def __init__(self, root):
        self.root = root
        self.img_files = glob.glob(os.path.join(root, 'image/*.jpg'))

        logger.info("Found {} images for PPM-100.".format(len(self.img_files)))
    
    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        for img_file in sorted(self.img_files):
            img = cv2.imread(img_file)
            mask_file = img_file.replace('image', 'matte')
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            img_name = os.path.basename(img_file).replace('.jpg', '')

            trimap_file = mask_file.replace('matte', 'trimap').replace('.jpg', '.png')
            trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)
            #cv2.imwrite('tmp/trimap.jpg', trimap)
            #pdb.set_trace()

            res = {'img_name': img_name, 'img': img, 'matte': mask, 'trimap': trimap}
            yield res

def gen_dataset_trimaps():
    root = './Datasets/PPM-100'
    df = PPM100Eval(root)
    df.reset_state()

    trimap_dir = os.path.join(root, 'trimap')
    if not os.path.isdir(trimap_dir):
        os.makedirs(trimap_dir)

    for data in tqdm.tqdm(df):
        matte = data['matte']
        img_name = data['img_name']
        trimap = gen_trimap_with_dilation(matte, kernel_size=30)      # values in [0, 1, 2]
        trimap = np.array([0, 128, 255], dtype=np.uint8)[trimap]     # 0->0, 1->128, 2->255
        cv2.imwrite(os.path.join(trimap_dir, f'{img_name}.png'), trimap)

if __name__ == '__main__':
    gen_dataset_trimaps()
