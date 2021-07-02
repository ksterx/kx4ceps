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

# +
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

# import seaborn as sns
# from data_transformation_euler import vec_similarity
# -

def analyzer(pred_df, target_df, savepath, num, size):
    e_trans3d = (
        torch.tensor(pred_df.loc[:, ["e_x", "e_y", "e_z"]].values)
        .norm(dim=1, keepdim=True)
        .numpy()
    )
    a = torch.tensor(pred_df.loc[:, ["e_x_2d", "e_y_2d"]].values)
    a[:, 0], a[:, 1] = a[:, 0] * size[0], a[:, 1] * size[1]
    e_trans2d = a.norm(dim=1, keepdim=True).numpy()

    n_pred = torch.tensor(pred_df.loc[:, ["nx", "ny", "nz"]].values)
    n_target = torch.tensor(target_df.loc[:, ["nx", "ny", "nz"]].values)
    _, e_orient = vec_similarity(n_pred, n_target)

    pred_df = pred_df.join(
        pd.DataFrame(
            np.concatenate([e_trans3d, e_trans2d, e_orient.numpy()], 1),
            columns=["e_trans3d", "e_trans2d", "e_orient"],
        )
    )

    # The roll error must be in the range; [-180, 180]
    for i in range(len(pred_df)):
        if pred_df.at[i, "e_gamma"] > 180:
            pred_df.at[i, "e_gamma"] = pred_df.at[i, "e_gamma"] - 360

        elif pred_df.at[i, "e_gamma"] < -180:
            pred_df.at[i, "e_gamma"] = 360 + pred_df.at[i, "e_gamma"]

    pred_df.to_csv(savepath + "/pred_{:03d}.csv".format(num), index=False)

    report = """
    Error            MAE       SD
    ===================================
    3D [mm]:        %.2f     %.2f
    2D [px]:        %.2f     %.2f
    Orient [deg]:   %.2f     %.2f
    Joint  [deg]:   %.2f     %.2f
    Rotate [deg]:   %.2f     %.2f
    """ % (
        pred_df["e_trans3d"].mean(),
        pred_df["e_trans3d"].std(),
        pred_df["e_trans2d"].mean(),
        pred_df["e_trans2d"].std(),
        abs(pred_df["e_orient"]).mean(),
        pred_df["e_orient"].std(),
        abs(pred_df["e_phi"]).mean(),
        pred_df["e_phi"].std(),
        abs(pred_df["e_gamma"]).mean(),
        pred_df["e_gamma"].std(),
    )

    with open(savepath + "/report.md", mode="w") as f:
        f.write(report)

    print(report)


def vec_similarity(a, b):
    cosine = (a * b).sum(dim=1, keepdim=True) / (
        torch.norm(a, dim=1, keepdim=True) * torch.norm(b, dim=1, keepdim=True)
    )
    angle = cosine.acos() * 180 / np.pi

    return cosine, angle


if __name__ == "__main__":
    p_path = input("pred_xxx.csv path: ")
    t_path = input("val_xxx.csv path: ")
    s_path = input("artifact path: ")
    ds_num = int(input("Dataset: #"))
    p_df = pd.read_csv(p_path)
    t_df = pd.read_csv(t_path)
    analyzer(p_df, t_df, s_path, ds_num, (224, 224))


