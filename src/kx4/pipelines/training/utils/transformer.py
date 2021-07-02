# %%
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

# %%
from .geometry import project_onto_plane


# %%
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


# %%
class Scale(object):
    def __init__(self, scale: list):
        self.scale = scale  # [scale_min, scale_max]

    def __call__(self, img, label):
        width = img.size[0]
        height = img.size[1]

        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        label = label.resize((scaled_w, scaled_h), Image.NEAREST)

        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h - height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left + width, top + height))
            label = label.crop((left, top, left + width, top + height))

        else:
            p_palette = label.copy().getpalette()

            img_original = img.copy()
            label_original = label.copy()

            pad_width = width - scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height - scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            label = Image.new(label.mode, (width, height), (0))
            label.paste(label_original, (pad_width_left, pad_height_top))
            label.putpalette(p_palette)

        return img, label


# %%
class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, label):
        rotate_angle = np.random.uniform(self.angle[0], self.angle[1])
        img = img.rotate(rotate_angle, Image.BILINEAR)
        label = label.rotate(rotate_angle, Image.NEAREST)

        return img, label


# %%
class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, img, label):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            label = ImageOps.mirror(label)
        return img, label


# %%
class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, label):
        img = img.resize(self.input_size, Image.BICUBIC)
        label = label.resize(self.input_size, Image.NEAREST)

        return img, label


# %%
class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, label):

        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)

        label = np.array(label)

        index = np.where(label == 255)
        label[index] = 0

        label = torch.from_numpy(label)

        return img, label


# %%
class ScaleTransformer:
    def __init__(self, ds_num):

        with open("Database/ds_{:03d}/ds_config.yaml".format(ds_num)) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        camera = config["camera"]
        x = camera["z_max"] * np.tan(np.radians(camera["fov"] / 2))
        y = x * camera["aspect"]
        self.aspect = camera["aspect"]
        self.fov = camera["fov"]

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

    def transform(self, target):
        return self.scaler.transform(target)

    def inverse_transform(self, target, size=None):

        if isinstance(target, dict):
            target_ = (
                torch.cat(
                    [
                        target["trans3d"],
                        target["orient"],
                        target["roll"],
                        target["joint"],
                    ],
                    dim=1,
                )
                .cpu()
                .detach()
                .numpy()
            )

            target_ = torch.tensor(self.scaler.inverse_transform(target_))

            trans2d = target["trans2d"].cpu() * torch.tensor([size])

            ret = torch.cat([target_[:, :3], trans2d, target_[:, 3:]], dim=1)

        elif target.shape[0] == 1:
            target_ = torch.tensor(self.scaler.inverse_transform(target)).squeeze()
            trans2d = project_onto_plane(
                target_[:3], a_ratio=self.aspect, fov=self.fov, is_batch=False
            )
            trans2d *= torch.tensor([1920, 1080])
            gamma = torch.rad2deg(torch.atan(target_[6] / target_[7])).unsqueeze(0)
            ret = torch.cat(
                [target_[:3], trans2d, target_[3:6], gamma, target_[8].unsqueeze(0)]
            ).numpy()

        else:
            target_ = torch.tensor(self.scaler.inverse_transform(target)).squeeze()
            trans2d = project_onto_plane(
                target_[:, :3], a_ratio=self.aspect, fov=self.fov, is_batch=True
            )
            trans2d *= torch.tensor([1920, 1080])
            gamma = (
                torch.rad2deg(torch.atan(target_[:, 6] / target_[:, 7])).unsqueeze(0).T
            )

            ret = torch.cat(
                [
                    target_[:, :3],
                    trans2d,
                    target_[:, 3:6],
                    gamma,
                    target_[:, 8].unsqueeze(0).T,
                ],
                dim=1,
            ).numpy()

        return ret
