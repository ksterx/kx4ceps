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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns

# %matplotlib inline
# -

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"

df = pd.read_csv("./7dc2f19acda94b3ea030e83a3a76c705/artifacts/pred_roll_021a.csv")

fig, ax = plt.subplots(figsize=(7, 4), facecolor="w", dpi=300)

m = ax.scatter(
    df["gamma_c"],
    df["gamma_s"],
    marker="o",
    s=10,
    c=df["e_gamma"],
    cmap="RdYlBu",
    vmin=0,
    vmax=180,
)

fig.colorbar(mappable=m, label="Roll Error [deg]")

ax.set_aspect("equal")

ax.set_xlabel("$\gamma_c$")
ax.set_ylabel("$\gamma_s$")

circle = patches.Circle(xy=(0, 0), radius=1.0, ec="k", fill=False)
ax.add_patch(circle)


fig.savefig("scatter180")
