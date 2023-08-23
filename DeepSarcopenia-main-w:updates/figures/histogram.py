import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/papers'

df = pd.read_csv(proj_dir + '/results/test.csv')
data = df['delta'].values * df['xy_spacing'].values
print('median value:', np.median(data))
#data = np.absolute(df['delta'].values)

fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(1, 1, 1)
#ax.set_aspect('equal')

sns.color_palette('deep')
#sns.histplot(data=data, bins='auto', stat='frequency', binwidth=1, color='royalblue', kde=True, line_kws={'color': 'navy', 'lw': 4})
sns.histplot(data=data, bins='auto', stat='frequency', binwidth=0.5, color='green', kde=True, line_kws={'color': 'red', 'lw': 4})
plt.ylabel('Count', fontweight='bold', fontsize=30)
plt.xlabel('$\Delta$h (mm)', fontweight='bold', fontsize=30)
#plt.ylim([0, 50])
#plt.xlim([0, 12])
#plt.xticks([0, 2, 4, 6, 8, 10, 12], fontsize=25, fontweight='bold')
#plt.yticks([0, 10, 20, 30, 40, 50], fontsize=25, fontweight='bold')
plt.ylim([0, 30])
plt.xlim([-2, 4])
plt.xticks([-2, -1, 0, 1, 2, 3, 4], fontsize=25, fontweight='bold') 
plt.yticks([0, 5, 10, 15, 20, 25, 30], fontsize=25, fontweight='bold')

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(4)  # change width
    ax.spines[axis].set_color('black')    # change color
    ax.tick_params(width=4, length=8) 

plt.grid(True)
plt.savefig(proj_dir + '/figures/histogram_test.png', format='png', dpi=150, bbox_inches='tight')