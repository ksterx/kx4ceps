# Import Libraries
import torch
import pandas as pd
import numpy as np
from importlib import import_module
from tkinter import *
from tkinter import filedialog, ttk
import re, os, time, tqdm
import subprocess

from utils.data_transformation_euler import Image2Tensor, Scaler
from utils.reprojection import repon2plane

from PIL import Image
import cv2


class Estimator:
    def __init__(self, net, img_path, mode, is_preresized, yaml_path, resize_shape):
        """
        :param img_path:
        :param mode: ONLINE:0/ENTIRE:1
        :param is_preresized: bool
        :param yaml_path:
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        self.mode = mode
        self.img_path = img_path
        self.yaml_path = yaml_path

        input_img = Image.open(img_path)
        if not is_preresized:
            transform = Image2Tensor(is_batch_learning=False).pil_resize
            img = transform(input_img, resize_shape)
        else:
            transform = Image2Tensor(is_batch_learning=False)
            img = transform(input_img)
        img = img.to(device)
        pred = net(img).view(9).detach().cpu().numpy()  # Estimation results
        scaler = Scaler(pred, is_batch=False)
        denorm_pred = scaler.denorm_param(yaml_path, is_batch=False)

        self.x, self.y, self.z = denorm_pred[0].cpu().numpy()
        self.x_2d, self.y_2d = repon2plane(
            denorm_pred[0].cpu().numpy(), (16, 9), 70, False
        )
        self.ux, self.uy, self.uz = denorm_pred[1].cpu().numpy()
        self.phi = denorm_pred[2]
        self.gamma = calc_deg(
            torch.Tensor([pred[7] * 2 - 1]), torch.Tensor([pred[8] * 2 - 1])
        )

    def __call__(self):
        if not self.mode:
            print(
                """
        PREDICTION>>
        x: {:.2f} mm
        y: {:.2f} mm
        z: {:.2f} mm
        x_2d: {:.2f} px
        y_2d: {:.2f} px
        ux: {:.5f} 
        uy: {:.5f} 
        uz: {:.5f}
        phi: {:.2f} deg
        gamma: {:.2f} deg
        """.format(
                    self.x,
                    self.y,
                    self.z,
                    self.x_2d,
                    self.y_2d,
                    self.ux,
                    self.uy,
                    self.uz,
                    self.phi,
                    self.gamma,
                )
            )

        return (
            self.x,
            self.y,
            self.z,
            self.ux,
            self.uy,
            self.uz,
            self.phi,
            self.gamma,
            self.x_2d,
            self.y_2d,
        )

    def show_label(self, dataframe, im_num):
        #         im_num = int(re.search(r'\d+', self.img_path).group())
        label = dataframe.iloc[im_num]

        print(
            """
        LABEL>>
        x: {:.2f} mm
        y: {:.2f} mm
        z: {:.2f} mm
        x_2d: {:.2f} px
        y_2d: {:.2f} px
        ux: {:.5f}
        uy: {:.5f}
        uz: {:.5f}
        phi: {:.2f} deg
        gamma: {:.2f} deg
        """.format(
                label[0],
                label[1],
                -1 * label[2],
                label[10] * 1920,
                label[11] * 1080,
                label[3],
                label[4],
                label[5],
                label[6],
                label[7],
            )
        )


if __name__ == '__main__':
    m_num = int(input('Model: #'))
    model = import_module("models.model_{:03d}".format(m_num)).HPNet(True)
    weight = torch.load(weight_dir + '/weight.pth')
    model.load_state_dict(weight)

    is_preresized = input('Are images pre-resized? - [Yes:y/No:n] >> ')

    mode = int(input('MODE: [ONLINE:0 /ENTIRE:1] >> '))
    # Check results one by one
    if not mode:
        while True:
            img_num = int(input('Image: #'))
            im_path = ds_dir + '/val/img_{:05d}.jpg'.format(img_num)

            t1 = time.time()
            estimator(im_path, is_preresized=isPreresized, mode=mode)
            t2 = time.time()

            show_label(img_num, df)

            print('Process time: {:.3f} sec'.format(t2 - t1))
            input_img.show()

            continueOrNot = input('Again? [Yes: /No: n]:')
            if continueOrNot == 'n':
                break

        t2 = time.time()
        print('Process time: {:.3f} sec'.format(t2 - t1))
        input_img.show()

    else:
        value_list = []
        for i in tqdm.trange(len(df)):
            im_path = ds_dir + '/val/img_{:05d}.jpg'.format(i + 1)
            value = estimator(im_path, is_preresized=isPreresized, mode=mode)
            value_list.append(value)

        pred_df = pd.DataFrame(value_list)
        pred_df.to_csv('C:/Users/Lab/Desktop/pred_{:03d}.csv'.format(ds_num),
                       header=['x', 'y', 'z', 'ux', 'uy', 'uz', 'oc'], index=None)


def calc_deg(sine, cosine):
    deg = torch.rad2deg(torch.atan(sine/cosine))

    if cosine < 0:
        if sine > 0:
            deg = deg + 180
        else:
            deg = deg - 180

    return deg.item()
