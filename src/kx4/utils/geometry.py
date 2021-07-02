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

import torch
import cv2
import numpy as np
import pandas as pd


def sc2deg(sine: torch.tensor, cosine: torch.tensor):
    deg = torch.rad2deg(torch.atan2(sine, cosine))
    return deg


def scale180(a: torch.tensor):
    a = torch.where(a > 180, 360-a, a)
    a = torch.where(a < -180, 360+a, a)
    return a


def project_onto_plane(p, a_ratio: float, fov, is_batch):
    '''
    Args:
        p: position (x, y, z)
        a_ratio: aspect ratio (height, width)
        fov: field of view [deg]

    O +------> x_2d
      |
      |
      v
     y_2d

    Returns:
        tuple: (x_2d [0, 1], y_2d [0, 1])
    '''
    assert 0 < a_ratio <= 1
    
    if not is_batch:

        if p[2] < 0:
            p *= -1
        x_2d = p[0] / (p[2] * np.tan(fov/2))
        y_2d = (a_ratio * p[1]) / (p[2] * np.tan(fov/2))
        x_2d = (1 + x_2d) / 2
        y_2d = (1 - y_2d) / 2
        ret = torch.tensor([x_2d, y_2d])

    else:
        if p[1, 2] < 0:
            p[:, 2] *= -1
        x_2d = p[:, 0] / (p[:, 2] * np.tan(fov/2))
        y_2d = (a_ratio * p[:, 1]) / (a_ratio * p[:, 2] * np.tan(fov/2))
        x_2d = x_2d.view(-1, 1)
        y_2d = y_2d.view(-1, 1)
        x_2d = (1 + x_2d) / 2
        y_2d = (1 - y_2d) / 2
        ret = torch.cat([x_2d, y_2d], dim=1)

    return ret

