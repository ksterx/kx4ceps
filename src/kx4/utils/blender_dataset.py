import re
import sys
from importlib import import_module
import pandas as pd
import torch.utils.data as data
from utils.data_transformation import Scaler

from PIL import Image


preresized = conf.pr
if preresized:
    w_resize = int(re.sub(r'\D', '', trainer.ds_dir)[3:6])  # Image width
    h_resize = int(re.sub(r'\D', '', trainer.ds_dir)[6:9])  # Image height
else:
    w_resize = int(input('Resize width: '))
    h_resize = int(input('Resize height: '))
resize_shape = (w_resize, w_resize)  # Resize shape if images are not resized


class BlenderDataset(data.Dataset):
    def __init__(self, phase, transforms, is_preresized):
        """
        :param transform:
        :param phase: 'train'/'val'
        """
        self.phase = phase
        self.transform = transforms
        self.is_preresized = is_preresized
        self.dataframe = pd.read_csv('../Database/ds_{1}/{0}_{1:03d}.csv'.format(phase, conf.ds_num))
        self.img_path_list = [trainer.ds_dir+'/{}x{}/{}/img_{:05d}.jpg'.format(w_resize, h_resize, phase, i + 1)
                              for i in range(len(self.dataframe))]  # img_00001.jpg~

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, img_idx):
        img = Image.open(self.img_path_list[img_idx])
        if self.is_preresized:
            img = self.transform(img)
        else:
            img = self.transform(img, resize_shape)

        data = Scaler(self.dataframe.iloc[img_idx])
        target = data.norm_param()

        return img, target




a = trainer.conf.ds_num
print(a)

