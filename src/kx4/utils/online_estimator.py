# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import mlflow
import torch
import time
import cv2
from mlflow import pytorch
from utils.data_transformation_euler import Image2Tensor, Scaler
from utils.compress_dataset import load_and_transform_image
from PIL import Image

exp_id = '222ed97ff5454eaf8899fbb57553dc53'

ds_num = 21


class Estimate():
    def __init__(self, e_id, ds_num):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = pytorch.load_model('mlruns/0/' + e_id + '/artifacts/models')
        self.model.to(self.device)
        self.model.eval()
        self.transform = Image2Tensor(is_batch_learning=False)
        self.ds_num = ds_num
        
    def __call__(self, img_path):
        with torch.no_grad():
            t1 = time.time()
            img = load_and_transform_image(img_path, (224, 224))
            norm_img = self.transform(img).to(self.device)
            pred = self.model(norm_img).squeeze().cpu().numpy()
            pred = Scaler(pred, is_batch=False)
            yaml_path = 'Database/ds_{:03d}/ds_config.yaml'.format(self.ds_num)
            p = pred.denorm_param(yaml_path, is_batch=False)
            print(p)
            t2 = time.time()
            fps = 1 / (t2 - t1)
            print('\nx: {:.2f} mm\n'
                  'y: {:.2f} mm\n'
                  'z: {:.2f} mm\n'
                  'nx: {:.5f}\n'
                  'ny: {:.5f}\n'
                  'nz: {:.5f}\n'
                  'roll: {:.2f} deg\n'
                  'joint: {:.2f} deg\n'
                  '{:.1f} fps\n'
                  .format(p[0][0], p[0][1], p[0][2], p[1][0], 
                          p[1][1], p[1][2], p[3], p[2], fps))


estimator = Estimate(exp_id, ds_num=ds_num)

t3 = time.time()
for i in range(20):
    print(i+1)
    im_path = 'Database/ds_022/train/img_{:05d}.jpg'.format(i+1)
    estimator(im_path)
t4 = time.time()
print(20 / (t4 - t3))

# + active=""
# video = cv2.VideoCapture(0)
#  
# cascade_path = "haarcascade_frontalface_default.xml"
# cascade = cv2.CascadeClassifier(cascade_path)
#  
# while video.isOpened():
#     ret, frame = video.read()
#
#     if not ret: break 
#  
#     # フレームの描画
#     cv2.imshow('frame', frame)
#  
#     # qキーの押下で処理を中止
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'): break
#  
# #メモリの解放
# video.release()
# cv2.destroyAllWindows()
