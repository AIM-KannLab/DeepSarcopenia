import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from scipy.stats import linregress


proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/papers'

df = pd.read_csv(proj_dir + '/results/test.csv') 

# pearson correlation
x = linregress(df['seg_csa'].values, df['pred_csa'].values)
print(x)
# add regression line with regplot()
fig = plt.figure(figsize=(10, 10))
ax  = fig.add_subplot(1, 1, 1)
#ax.set_facecolor('gray')
#ax.set_aspect('equal')
sns.color_palette('deep')
#sns.regplot(x='seg_csa', y='pred_csa', ci=95, data=df, color='turquoise', scatter_kws={'s': 200}, line_kws={'color': 'tomato', 'lw': 5})
sns.regplot(x=df['seg_csa'], y=df['pred_csa'], ci=95, color='turquoise', scatter_kws={'s': 200}, line_kws={'color': 'tomato', 'lw': 5})
#sns.lmplot(x='seg_csa', y='pred_csa', data=df)
plt.xlabel('Ground Truth C3 CSA ($\mathregular{cm^{2}}$)', fontweight='bold', fontsize=30)
plt.ylabel('Predicted C3 CSA ($\mathregular{cm^{2}}$)', fontweight='bold', fontsize=30)
plt.xlim([10, 70])
plt.ylim([10, 70])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(4)  # change width
    ax.spines[axis].set_color('black')    # change color
    ax.tick_params(width=4, length=8)

plt.xticks([10, 20, 30, 40, 50, 60, 70], fontsize=25, fontweight='bold')
plt.yticks([10, 20, 30, 40, 50, 60, 70], fontsize=25, fontweight='bold')
#plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
plt.grid(True)
#plt.show()
plt.savefig(proj_dir + '/figures/scatter_plot.png', format='png', dpi=150, bbox_inches='tight')
plt.close()

