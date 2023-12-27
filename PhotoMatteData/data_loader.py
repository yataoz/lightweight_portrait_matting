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


class PhotoMatte85Eval(DataFlow):
    def __init__(self, root):
        self.root = root
        self.img_files = sorted(glob.glob(os.path.join(root, 'rgba_image/*.png')))
        self.bg_files = sorted(glob.glob(os.path.join(root, 'background/*.jpg')))

        logger.info("Found {} images for PhotoMatte85.".format(len(self.img_files)))
    
    def __len__(self):
        return len(self.img_files)

    def __iter__(self):
        for img_file, bg_file in zip(self.img_files, self.bg_files):
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img, mask = img[..., :3], img[..., 3]
            img_name = os.path.basename(img_file).replace('.png', '')

            bg = cv2.imread(bg_file)
            # crop vertically from bg because img is in portrait mode
            im_h, im_w = img.shape[:2] 
            bg_h, bg_w = bg.shape[:2] 
            bg_w = int(bg_h * im_w / im_h)
            bg = bg[:, :bg_w]

            #bg = cv2.resize(bg, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            #bg = cv2.GaussianBlur(bg, (5, 5), -1)
            bg = cv2.resize(bg, img.shape[:2][::-1])
            img = (255 * ((img.astype(np.float32) / 255) * mask[..., np.newaxis].astype(np.float32) / 255 + (bg.astype(np.float32) / 255) * (1 - mask[..., np.newaxis].astype(np.float32) / 255))).astype(np.uint8)

            trimap_file = img_file.replace('rgba_image', 'trimap')
            trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)

            #cv2.imwrite('tmp/trimap.jpg', trimap)
            #pdb.set_trace()

            #res = {'img_name': img_name, 'img': img, 'matte': mask}
            res = {'img_name': img_name, 'img': img, 'matte': mask, 'trimap': trimap}
            yield res

def gen_dataset_trimaps():
    root = './Datasets/PhotoMatte85'
    df = PhotoMatte85Eval(root)
    df.reset_state()

    img_dir = os.path.join(root, 'image')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    matte_dir = os.path.join(root, 'matte')
    if not os.path.isdir(matte_dir):
        os.makedirs(matte_dir)

    trimap_dir = os.path.join(root, 'trimap')
    if not os.path.isdir(trimap_dir):
        os.makedirs(trimap_dir)

    for data in tqdm.tqdm(df):
        img = data['img']
        matte = data['matte']
        img_name = data['img_name']
        trimap = gen_trimap_with_dilation(matte, kernel_size=30)      # values in [0, 1, 2]
        trimap = np.array([0, 128, 255], dtype=np.uint8)[trimap]     # 0->0, 1->128, 2->255
        cv2.imwrite(os.path.join(img_dir, f'{img_name}.jpg'), img)
        cv2.imwrite(os.path.join(matte_dir, f'{img_name}.png'), matte)
        cv2.imwrite(os.path.join(trimap_dir, f'{img_name}.png'), trimap)
        #cv2.imwrite('tmp/trimap.jpg', trimap)
        #cv2.imwrite('tmp/mask.jpg', matte)
        #pdb.set_trace()

if __name__ == '__main__':
    gen_dataset_trimaps()
