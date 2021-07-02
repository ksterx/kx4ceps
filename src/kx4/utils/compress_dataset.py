from PIL import Image
import os, glob
from . import notifier
import argparse


def load_and_transform_image(path, size):
    img_raw = Image.open(path)
    img = img_raw.resize(size, resample=Image.BICUBIC)

    return img


if __name__=='__main__':

    resize_W = int(input('Width: '))
    resize_H = int(input('Height: '))
    aspect = (resize_W, resize_H)

    dataset_num = int(input('Dataset: #'))
    dataset_dir = './ds_{:03d}'.format(dataset_num)

    train_img_path_list = glob.glob(dataset_dir + '/train/img_*.jpg'.format(dataset_num))
    val_img_path_list = glob.glob(dataset_dir + '/val/img_*.jpg'.format(dataset_num))
    path_list_dict = {'train': train_img_path_list, 'val': val_img_path_list}

    while True:
        if not os.path.exists(dataset_dir+"/{}x{}".format(resize_W, resize_H)):
            os.makedirs(dataset_dir + "/{}x{}/train".format(resize_W, resize_H))
            os.makedirs(dataset_dir + "/{}x{}/val".format(resize_W, resize_H))
            break

        else:
            dataset_num += 1
            continue


    def save_image(size):
        for phase in ['train', 'val']:
            count = 1
            for img in path_list_dict[phase]:
                image = load_and_transform_image(img, size)
                image.save(dataset_dir + '/{}x{}/{}/img_{:05d}.jpg'.format(size[0], size[1], phase, count), quality=100)
                count += 1


    save_image((resize_W, resize_H))

    message=("\n=========================\n"
             " PROCESS COMPLETED    \n"
             "=========================\n")

    notifier.line_notify(message)
    notifier.slack_notify(message)
    print(message)
