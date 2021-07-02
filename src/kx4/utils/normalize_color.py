import glob
from PIL import Image
import numpy as np
import tqdm


def norm_color(img_path):
    img = np.array(Image.open(img_path)) / 255
    img_cmean = img.mean(axis=(0, 1))
    img_cstd = img.std(axis=(0, 1))

    return img_cmean, img_cstd


if __name__ == '__main__':
    ds_num = int(input('Dataset: #'))

    cmean = np.zeros(3)
    cstd = np.zeros(3)

    imgpath_list = glob.glob("../Database/ds_{:03d}/*/*.jpg".format(ds_num))

    for imgpath in tqdm.tqdm(imgpath_list):
        img_cmean, img_cstd = norm_color(imgpath)
        cmean += img_cmean
        cstd += img_cstd

    cmean /= len(imgpath_list)
    cstd /= len(imgpath_list)

    print("color mean =", cmean)
    print("color std  =", cstd)


