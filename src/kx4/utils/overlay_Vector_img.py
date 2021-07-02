"""Generating a sim video"""

import bpy
import numpy as np
import pandas as pd
# from estimator_vector import *

unit2mm = 1 / 1000

file_format = '.png'   # jpg/png

df = pd.read_csv('C:/Users/Lab/Desktop/pred.csv')

arrow = bpy.data.objects['Arrow']
target = bpy.data.objects['Target']


def overlay(target_vec, arrow_orgn):
    target_loc = target.location
    target_loc.x, target_loc.y, target_loc.z = target_vec[0], target_vec[1], -target_vec[2]
    arrow_loc = arrow.location
    arrow_loc.x, arrow_loc.y, arrow_loc.z = arrow_orgn[0], arrow_orgn[1], -arrow_orgn[2]
    target_loc.x, target_loc.y, target_loc.z = target_loc + arrow_loc


# for i in range(len(df)):
for i in range(len(df)):
    t_vec = df.iloc[i, 3:]
    a_orgn = df.iloc[i, :3] * unit2mm
    overlay(t_vec, a_orgn)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=r'C:\Users\Lab\Desktop\overlay_imgs\overlay_img_{:05d}{}'
                                                 .format(i, file_format))
