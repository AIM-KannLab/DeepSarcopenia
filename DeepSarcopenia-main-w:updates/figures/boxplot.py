import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/papers'

df = pd.read_csv(proj_dir + '/results/dice.csv')

fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(1, 1, 1)
#ax.set_aspect('equal')

sns.color_palette('deep')
sns.violinplot(data=df, color='0.8', orient='v', linewidth=4)
sns.stripplot(data=df, jitter=True, zorder=1, orient='v', size=10, linewidth=2, edgecolor='black')
#sns.swarmplot(data=df, orient='v')
plt.ylabel('DCS', fontweight='bold', fontsize=30)
plt.ylim([0, 1.1])

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(4)  # change width
    ax.spines[axis].set_color('black')    # change color
    ax.tick_params(width=4, length=8)
plt.xticks(['Validation', 'Test'], fontsize=30, fontweight='bold')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=25, fontweight='bold')
#plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
plt.grid(True)
plt.savefig(proj_dir + '/figures/box_plot.png', format='png', dpi=150, bbox_inches='tight')


