import os
import numpy as np
import pandas as pd
import glob
from scipy.stats import bootstrap

proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/papers/results'
df = pd.read_csv(proj_dir + '/test.csv')
for metric in ['dice', 'precision', 'recall']:
    data = (df[metric].values, )
    #print(data)
    res = bootstrap(data, np.median, n_resamples=10000, confidence_level=0.95, random_state=42, method='percentile')
    print(metric)
    print(res.confidence_interval)
    print('median:', np.median(data))