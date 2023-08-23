import argparse
import warnings
import os 
import pandas as pd
import SimpleITK as sitk
import numpy as np
from opts import parse_opts, get_args

from src.infer_slice_selection import test_slice_selection

# Test the slice selection model that predicts C3 slice for each raw_scan given as input
# Please set the directories path in the following folder before executing the script.

def slice_model(img_dir,slice_model,slice_csv_path):
    """
    Test the Slice Selection Model
    Args:
        Input Scans -- nrrd files
        Model -- C3_Top_Selection_Model_Weight.hdf5 
        Output -- C3_Top_Slice_Prediction.csv' 
    """
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    #raw_img_dir = proj_dir + '/data/train_data/test_img'
    #raw_seg_dir = proj_dir + '/data/train_data/test_seg'
    #model_path = proj_dir + '/model/test/C3_Top_Selection_Model_Weight.hdf5'
    #slice_csv_path = proj_dir + '/data/slice_selection/C3_slice_sum.csv'
    #output_csv_path = proj_dir + '/internal_test/slice_model/slice_sum.csv'
    #test_slice_selection(
    #    image_dir=raw_img_dir, 
    #    model_weight_path=model_path, 
    #    csv_write_path=output_csv_path)


    print('--- slice selection ---')
    test_slice_selection(
        image_dir=img_dir, 
        model_weight_path=slice_selection_model, 
        csv_write_path=slice_csv_path)


    
def combine_csv():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    df0 = pd.read_csv(proj_dir + '/internal_test/slice_model/C3_slice_sum.csv', index_col=0)
    df0 = df0[['ID', 'C3_Manual_slice', 'Datatype']]
    print(df0)
    df1 = pd.read_csv(proj_dir + '/internal_test/slice_model/slice_sum.csv', index_col=0)
    df1 = df1[['patient_id', 'C3_Predict_slice', 'Z_spacing', 'XY_spacing']]
    df1.columns = ['ID', 'C3_predict_slice', 'z_spacing', 'xy_spacing']
    print(df1)
    df = df1.merge(df0, how='left', on='ID')
    df = df[['ID', 'C3_predict_slice', 'C3_Manual_slice', 'z_spacing', 'xy_spacing', 'Datatype']]
    df.to_csv(proj_dir + '/internal_test/slice_model/slice_tot.csv', index=False)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')

    opt = parse_opts()
    dict1 = get_args(opt)
    
    
    slice_model(dict1["img_dir"],dict1["slice_model"],dict1["slice_csv_path"])
    #combine_csv()












