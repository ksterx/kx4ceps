"""Generating a sim video"""

import bpy
import numpy as np
import pandas as pd
# from estimator_vector import *

ds_num = 18
df = pd.read_csv('C:/Users/Lab/Desktop/pred_{:03d}.csv'.format(ds_num))
forceps_df = pd.read_csv(r'C:\Users\Lab\GoogleDrive\Database\ds_{0:03d}\val_{0:03d}.csv'.format(ds_num))

unit2mm = 1 / 1000

file_format = '.png'   # jpg/png

arrow = bpy.data.objects['Arrow']
target = bpy.data.objects['Target']
forceps = bpy.data.objects['forceps']
free_edge = bpy.data.objects['Free_Edge']


def overlay(target_vec, arrow_orgn, frcps_loc, frcps_rot, edge_angle):
    target_loc = target.location
    target_loc.x, target_loc.y, target_loc.z = target_vec[0], target_vec[1], -target_vec[2]
    arrow_loc = arrow.location
    arrow_loc.x, arrow_loc.y, arrow_loc.z = arrow_orgn[0], arrow_orgn[1], -arrow_orgn[2]
    target_loc.x, target_loc.y, target_loc.z = target_loc + arrow_loc
    forceps_loc = forceps.location
    forceps_rot = forceps.rotation_euler
    forceps_loc.x, forceps_loc.y, forceps_loc.z = frcps_loc
    forceps_rot.x, forceps_rot.y, forceps_rot.z = frcps_rot
    fe_rot = free_edge.rotation_euler
    fe_rot.x = edge_angle


for i in range(100):
    t_vec = df.iloc[i, 3:]
    a_orgn = df.iloc[i, :3] * unit2mm
    f_loc = forceps_df.iloc[i, :3] * unit2mm
    f_rot = forceps_df.iloc[i, 3:6]
    e_angle = - (forceps_df.iloc[i, 9] + np.pi/2)
    overlay(t_vec, a_orgn, f_loc, f_rot, e_angle)

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=r'C:\Users\Lab\Desktop\results\{:05d}{}'.format(i+1, file_format))
