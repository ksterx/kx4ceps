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

import torch
import yaml
import numpy as np
from PIL import Image

class Image2Tensor:
    def __init__(self, is_batch_learning=True):
        self.is_batch_learning = is_batch_learning

    def __call__(self, input_img):
        img =  np.asarray(input_img).astype('f') / 255

        if self.is_batch_learning:
            img = torch.from_numpy(img)
            img = img.numpy().transpose((2, 0, 1))
            img = torch.from_numpy(img)

        else:
            img = torch.from_numpy(img).unsqueeze_(0)
            img = img.numpy().transpose((0, 3, 1, 2))
            img = torch.from_numpy(img)

        return img

    def pil_resize(self, input_img, resize_shape):
        """
        If an image is not pre-resized...
        :param input_img:
        :param resize_shape:
        :return:
        """
        im = input_img.resize(resize_shape, Image.BICUBIC)
        im = np.asarray(im).astype('f') / 255

        if self.is_batch_learning:
            img = torch.from_numpy(im)
            img = img.numpy().transpose((2, 0, 1))
            img = torch.from_numpy(img)

        else:
            img = torch.from_numpy(im).unsqueeze_(0)
            img = img.numpy().transpose((0, 3, 1, 2))
            img = torch.from_numpy(img)

        return img


# ### Parameter range
# $$
# -1\le u_x\le 1, -1\le u_y\le 1, -1\le u_z\le -0.5
# $$

class Scaler:
    def __init__(self, value_list, is_batch):
        if is_batch:
            self.x, self.y, self.z = torch.chunk(value_list[0], 3, dim=1)
            self.ux, self.uy, self.uz = torch.chunk(value_list[1], 3, dim=1)
            self.phi = value_list[2]
            self.gamma = value_list[3]

        else:
            self.x, self.y, self.z = value_list[:3]
            self.ux, self.uy, self.uz = value_list[3:6]
            self.phi = value_list[6]
            self.gamma = value_list[7]

    def norm_param(self, yaml_path):
        """
        Normalize parmeters (0-1)
        """
        x_range, y_range, z_min, z_max, phi_max = read_conf(yaml_path)

        # Position
        x = (self.x + x_range) / (2 * x_range)
        y = (self.y + y_range) / (2 * y_range)
        z = (self.z + z_min) / (z_min - z_max)

        # Orientation
        ux = (self.ux + 1) / 2
        uy = (self.uy + 1) / 2
        uz = -(2 * self.uz + 1)
        
        # Joint angle
        phi = self.phi / phi_max
        
        # Rotation
        gamma = (self.gamma + 180) / 360

        return torch.Tensor([x, y, z]), torch.Tensor([ux, uy, uz]), torch.Tensor([phi]), torch.Tensor([gamma])

    def denorm_error(self, yaml_path=None):
        """
        Denormalize error (0-1) to error (unit) such as mm
        """
        x_range, y_range, z_min, z_max, phi_max = read_conf(yaml_path)

        # Position
        x = 2 * self.x * x_range
        y = 2 * self.y * y_range
        z = self.z * (z_max - z_min)

        # Orientation
        ux = 2 * self.ux
        uy = 2 * self.uy
        uz = self.uz / 2
        
        # Rotation
        gamma = self.gamma * 360
        
        # Joint angle
        phi = self.phi * phi_max

        return x, y, z, ux, uy, uz, phi, gamma

    def denorm_param(self, yaml_path, is_batch):
        x_range, y_range, z_min, z_max, phi_max = read_conf(yaml_path)

        # Position
        x = 2 * (self.x - 0.5) * x_range
        y = 2 * (self.y - 0.5) * y_range
        z = self.z * (z_max - z_min) + z_min
  
        # Orientation
        ux = 2 * self.ux - 1
        uy = 2 * self.uy - 1
        uz = -(self.uz + 1) / 2

        # Rotation
        gamma = self.gamma * 360 - 180

        # Joint angle
        phi = self.phi * phi_max

        if is_batch:
            trans = torch.cat([x, y, z], dim=1)
            orient = torch.cat([ux, uy, uz], dim=1)
        else:
            trans = torch.tensor([x, y, z])
            orient = torch.tensor([ux, uy, uz])

        return trans, orient, phi, gamma

def read_conf(yamlpath):
    with open(yamlpath) as f:
        ds_conf = yaml.load(f, Loader=yaml.SafeLoader)
        x_range = ds_conf['translation']['x_range']
        y_range = ds_conf['translation']['y_range']
        z_min = ds_conf['translation']['z_min']
        z_max = ds_conf['translation']['z_max']
        phi_max = ds_conf['articulation']['phi_max']
    return x_range, y_range, z_min, z_max, phi_max


# ## Cosine Similarity

def vec_similarity(a, b):
    cosine = (a * b).sum(dim=1, keepdim=True)\
             / (torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True))
    angle = cosine.acos() * 180 / np.pi

    return cosine, angle
