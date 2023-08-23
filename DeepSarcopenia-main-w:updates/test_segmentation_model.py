import argparse
import warnings
import os 
import pandas as pd
import SimpleITK as sitk
import numpy as np
from opts import parse_opts, get_args
from src.infer_segmentation import test_segmentation
from src.infer_slice_selection import test_slice_selection


def segmentation(pre_process_dir, seg_model, slice_csv_path, out_dir):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('--- C3 segmentation ---')
    
    test_segmentation(
        img_dir=pre_process_dir,
        model_weight_path=model_path,
        slice_csv_path=slice_csv_path,
        output_dir=out_dir)

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)

    segmentation(dict1["pre_process_dir"],dict1["seg_model"],dict1["slice_csv_path"],dict1["out_dir"])












