import cv2
import numpy as np
import pandas as pd


im_num = 50
im_path = r'C:\Users\Lab\GoogleDrive\HPNet\pytorch\logs\ds_015\MODEL_007_BATCH_008_EPOCH_100_NORM\results\{:05d}.png'.format(im_num)
im = cv2.imread(im_path)
h = im.shape[0]  # Height
w = im.shape[1]  # Width
a_ratio = 9 / 16  # Aspect ratio
theta = 35  # Angle of view [deg]

"""
Data shape:
+-----+-----+-----+
|  x  |  y  |  z  |
+-----+-----+-----+
x: Right direction of an image
y: Upward of an image
z: 
"""

# Import data as pandas.DataFrame
lbl_path = r'C:\Users\Lab\GoogleDrive\Database\ds_015\val_015.csv'
pred_path = r'C:\Users\Lab\GoogleDrive\HPNet\pytorch\logs\ds_015\MODEL_007_BATCH_008_EPOCH_100_NORM\pred_015.csv'
df_lbl = pd.read_csv(lbl_path)
df_pred = pd.read_csv(pred_path)

x_t, y_t, z_t = df_lbl.iloc[im_num-1, :3]
z_t = - z_t
x_pred, y_pred, z_pred = df_pred.iloc[im_num-1, :3]


def img_w(x_):  # Image width
    global w
    return w * (1 + x_) / 2


def img_h(y_):  # Image height
    global h
    return h * (1 - y_) / 2


r_t = z_t * np.tan(np.deg2rad(theta))
r_pred = z_pred * np.tan(np.deg2rad(theta))

x_t = x_t / r_t
y_t = y_t / (r_t * a_ratio)
x_pred = x_pred / r_pred
y_pred = y_pred / (r_pred * a_ratio)

x_t = img_w(x_t).round().astype(int)
y_t = img_h(y_t).round().astype(int)
x_pred = img_w(x_pred).round().astype(int)
y_pred = img_h(y_pred).round().astype(int)

print(x_t, y_t, x_pred, y_pred)


cv2.drawMarker(im, (x_t, y_t), color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)
cv2.drawMarker(im, (x_pred, y_pred), color=(0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)

cv2.imshow('image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()