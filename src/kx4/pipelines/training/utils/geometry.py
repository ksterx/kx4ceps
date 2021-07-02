# %%
import numpy as np
import torch


# %%
def sc2angle(s: torch.tensor, c: torch.tensor, rep: str) -> torch.tensor:
    """An angle is calcurated from sine and cosine parameters.
    These parameters don't have to meet "s^2 +c^2 = 1".

    Args:
        s (torch.tensor): Sine.
        c (torch.tensor): Cosine.

    Returns:
        rep (torch.tensor): Angle representation.
    """
    if rep == "deg":
        angle = torch.rad2deg(torch.atan2(s, c))

    elif rep == "rad":
        angle = torch.atan2(s, c)

    return angle


# %%
def scale180(x: torch.tensor) -> torch.tensor:
    """Rescale argument (deg)

    Args:
        x (torch.tensor): Degrees.

    Returns:
        torch.tensor: Degrees within [-180 deg, 180 deg].
    """
    x = torch.where(x > 180, -360 + x, x)
    x = torch.where(x < -180, 360 + x, x)
    return x


# %%
def scale_pi(x: torch.tensor) -> torch.tensor:
    """Rescale argument (rad)

    Args:
        x (torch.tensor): Radians.

    Returns:
        torch.tensor: Radians within [-pi, pi]
    """
    x = torch.where(x > np.pi, -np.pi + x, x)
    x = torch.where(x < -np.pi, np.pi + x, x)
    return x


# %%
def cam2image(
    position, a_ratio: float, fov: float, hand: str, is_batch: bool
) -> torch.tensor:
    """Project a 3D point of the camera coordinate system onto the image plane.

    Args:
        position (subscriptable): Neccessary to be this order: (x, y, z).
        a_ratio (float): Aspect ratio (height/width).
        fov (float): Field of view (deg)
        is_batch (bool): [description]

    O +------> x_2d
      |
      |
      v
     y_2d

    Returns:
        torch.tensor: (x_2d [0, 1], y_2d [0, 1])
    """
    assert 0 < a_ratio <= 1

    if not is_batch:

        if position[2] < 0:
            position *= -1
        x_2d = position[0] / (position[2] * np.tan(fov / 2))
        y_2d = (a_ratio * position[1]) / (position[2] * np.tan(fov / 2))
        x_2d = (1 + x_2d) / 2
        y_2d = (1 - y_2d) / 2
        ret = torch.tensor([x_2d, y_2d])

    else:
        if position[1, 2] < 0:
            position[:, 2] *= -1
        x_2d = position[:, 0] / (position[:, 2] * np.tan(fov / 2))
        y_2d = (a_ratio * position[:, 1]) / (a_ratio * position[:, 2] * np.tan(fov / 2))
        x_2d = x_2d.view(-1, 1)
        y_2d = y_2d.view(-1, 1)
        x_2d = (1 + x_2d) / 2
        y_2d = (1 - y_2d) / 2
        ret = torch.cat([x_2d, y_2d], dim=1)

    return ret
