import pandas as pd
import numpy as np
import pingouin as pg

proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/papers/results'
df = pd.read_csv(proj_dir + '/val.csv', index_col=False)

df1 = df[['patient_id', 'seg_csa']]
rater1 = ['seg'] * df.shape[0]
df1['rater'] = rater1
df1.columns = [['pat_id', 'csa', 'rater']]

df2 = df[['patient_id', 'pred_csa']]
rater2 = ['pred'] * df.shape[0]
df2['rater'] = rater2
df2.columns = [['pat_id', 'csa', 'rater']]

df3 = df1.append(df2)
df3.reset_index(drop=True, inplace=True)
print(df3)
df3.to_csv(proj_dir + '/xxx.csv')
icc = pg.intraclass_corr(data=df3, targets='pat_id', raters='rater', ratings='csa')
icc.set_index('Type')

