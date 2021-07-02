import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tkinter import filedialog

w_resize = int(input('Width: '))
h_resize = int(input('Height: '))

log_dir = filedialog.askdirectory(initialdir='C:/Users/Lab/GoogleDrive/HPNet/pytorch/logs')
df = pd.read_csv(log_dir + '/loss_{}x{}.csv'.format(w_resize, h_resize))
new_df = pd.DataFrame({"Training": df.iloc[:, 1], "Validation": df.iloc[:, 2]})
plt.figure()
new_df.plot(title='Loss')
plt.savefig(log_dir + '/loss_{}x{}.png'.format(w_resize, h_resize), dpi=300)

print('PROCESS COMPLETED!!')