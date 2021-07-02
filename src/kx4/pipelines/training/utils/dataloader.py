import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import yaml
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from .transformer import ScaleTransformer


class LaparoDataset(data.Dataset):
    def __init__(self, ds_num, phase, transform):
        #         self.ds_dir = os.getcwd() + '/Database/ds_{:03d}'.format(ds_num)
        self.ds_dir = "~/workspace/Database/ds_{:03d}".format(ds_num)
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(self.ds_dir + "/{}.csv".format(phase))
        df["z"] *= -1
        df["nz"] *= -1
        self.dataframe = df
        self.PARAMS = ["x", "y", "z", "nx", "ny", "nz", "gamma_s", "gamma_c", "phi"]

        with open(self.ds_dir + "/ds_config.yaml") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        camera = config["camera"]
        x = camera["z_max"] * np.tan(np.radians(camera["fov"] / 2))
        y = x * camera["aspect"]

        # Range of each parameter
        X = [-x, x]
        Y = [-y, y]
        Z = [camera["z_min"], camera["z_max"]]
        N = [-1.0, 1.0]
        NZ = [0.25, 0.95]
        GAMMA = [-1.0, 1.0]
        PHI = [0.0, config["articulation"]["phi_max"]]

        RANGE = np.stack([X, Y, Z, N, N, NZ, GAMMA, GAMMA, PHI], 0)

        self.scaler = MinMaxScaler()
        self.scaler.fit(RANGE.T)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.ds_dir, self.phase, "img_{:05d}.jpg".format(idx + 1))
        )
        img = self.transform(img)

        t = np.array([self.dataframe.loc[idx, self.PARAMS]])
        target = torch.Tensor(self.scaler.transform(t)).squeeze()

        return img, target


class NpLaparoDataset(data.Dataset):
    def __init__(self, ds_num, phase, input_size):
        self.input_size = input_size
        self.ds_dir = os.getcwd() + "/Database/ds_{:03d}".format(ds_num)
        self.phase = phase
        df = pd.read_csv(self.ds_dir + "/{}.csv".format(phase))
        df["z"] *= -1
        df["nz"] *= -1
        self.dataframe = df
        self.PARAMS = ["x", "y", "z", "nx", "ny", "nz", "gamma_s", "gamma_c", "phi"]

        self.scaler = ScaleTransformer(ds_num)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = np.load(
            os.path.join(
                self.ds_dir,
                "{}x{}/numpy".format(self.input_size[0], self.input_size[1]),
                self.phase,
                "img_{:05d}.npy".format(idx + 1),
            )
        )

        img = torch.tensor(img)

        t = np.array([self.dataframe.loc[idx, self.PARAMS]])
        target = torch.Tensor(self.scaler.transform(t)).squeeze()

        return img, target
