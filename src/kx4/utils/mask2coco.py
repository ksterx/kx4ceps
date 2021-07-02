# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import cv2
import datetime
import numpy as np
import json
import glob
import os.path as osp
from tqdm import tqdm
from skimage import measure
from pycocotools import mask

# +
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
YELLOW = [255, 255, 0]
MAGENTA = [255, 0, 255]
CYAN = [0, 255, 255]

info = {
    "description": "HPDataset 023",
    "url": "",
    "version": "1.0",
    "year": 2020,
    "contributor": "Kosuke Ishikawa <etarho.py@gmail.com>",
    "date_created": "{}".format(datetime.date.today())
}

LICENSES = [
    {
        "url": "",
        "id": 1,
        "name": "BMPE"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'old',
        'supercategory': 'forceps',
    },
    {
        'id': 2,
        'name': 'new',
        'supercategory': 'forceps',
    }
]


# -

class Img2Mask():
    
    def __init__(self, im, threshold=100):
        """
        Convert to mask
        """
        self.img = im.copy()
        self.img[im >= threshold] = 255
        self.img[im < threshold] = 0
        
        
    def __call__(self):
        """
        
        """
        img_ = self.img.copy()
        for color in [YELLOW, MAGENTA, CYAN]:
            img_[np.where((self.img == color).all(axis=2))] = BLACK
        
        return img_

    
    def to_npmask(self):
        img_ = np.zeros([img.shape[0], img.shape[1]])
        for i, color in enumerate([RED, GREEN, BLUE]):
            img_[np.where((self.img == color).all(axis=2))] = i+1
            
        return img_


def mask2coco(npmask, img_id, cat_id, coco_dict): 
    """
    Convert ndarray masks to COCO json file
    """
    
    images = {
                "license": 1,
                "coco_url": "",
                "date_captured": "",
                "flickr_rul": "",
                "id": img_id,
                "height": npmask.shape[0],
                "width": npmask.shape[1],
                "file_name": 'movieFrame_{:06d}.png'.format(img_id)
        }
    
    coco_dict['images'].append(images)
    
    fortran_ground_truth_binary_mask = np.asfortranarray(npmask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(npmask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": cat_id,
            "id": 1000000 + cat_id * 100000 + img_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)

    coco_dict['annotations'].append(annotation)


# +
if __name__ == '__main__':
    
    coco1a = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    coco1b = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    coco2a = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    coco2b = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    coco_dict = {'1a': coco1a, '1b': coco1b, '2a': coco2a, '2b': coco2b}
    cat_dict = {'1a': 1, '1b': 1, '2a': 2, '2b': 2}

    for dir_ in coco_dict.keys():
        img_path_list = glob.glob('../data/{}/label/*.png'.format(dir_))
        for img_path in tqdm(img_path_list):
            img = cv2.imread(img_path)
            img2mask = Img2Mask(img, threshold=1)
            mask_ = img2mask()
            npmask = (mask_ / 255).astype(np.uint8)
            npmask = npmask[:, :, 0]

            img_id = int(img_path[-10:-4])
            for i in range(4):
                if dir_ == list(coco_dict.keys())[i]:
                    mask2coco(npmask=npmask, img_id=img_id,
                              cat_id=cat_dict[dir_], coco_dict=coco_dict[dir_])

#             cv2.imwrite(img_dir + '/{}/mask_{}.png'.format(phase, i+1), mask_)

        with open('../data/{}/anno.json'.format(dir_), 'w') as f:
            json.dump(coco_dict[dir_], f, indent=4)

# -

print('Finished!!')
