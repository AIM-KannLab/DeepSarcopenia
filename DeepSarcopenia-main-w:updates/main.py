import argparse
import warnings
import os 
import pandas as pd
import SimpleITK as sitk
import numpy as np
import glob as glob
from opts import parse_opts, get_args
from src.infer_segmentation import test_segmentation
from src.infer_slice_selection import test_slice_selection
from src.segmentation_preprocess import preprocess2


def slice_selection(img_dir,slice_model,slice_csv_path):
    """
    Test the Slice Selction Model
    Args:
        Input Scans -- nrrd files
        Model -- C3_Top_Selection_Model_Weight.hdf5 
        Output -- C3_Top_Slice_Prediction.csv' 
    """
       
    
    print('--- slice selection ---')
    test_slice_selection(
        image_dir=img_dir, 
        model_weight_path=slice_model, 
        csv_write_path=slice_csv_path)


def segmentation(pre_process_dir, seg_model, slice_csv_path, out_dir):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('--- C3 segmentation ---')
    test_segmentation(
        img_dir=pre_process_dir,
        model_weight_path=seg_model,
        slice_csv_path=slice_csv_path,
        output_dir=out_dir
        )

    
def preprocess(img_dir, pre_process_dir):
    
    img_dirs = [i for i in glob.glob(img_dir + '/*nrrd')]
   
    if not os.path.exists(pre_process_dir):
        os.makedirs(pre_process_dir)
    
    preprocess2(img_dirs, pre_process_dir)
#def main() :
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#    warnings.filterwarnings('ignore')
#    opt = parse_opts()
#    dict1 = get_args(opt)
#    if (opt.test) :
        # Run Slice Slection
#        slice_selection(dict1["img_dir"], dict1["slice_model"],dict1["slice_csv_path"])
        # Run Processing steps on raw images for preprocessed files needed for segmenation
        #preprocess(dict1["img_dir"], dict1["pre_process_dir"])
        # Run the Segmentation which will generate output segmentations
        #segmentation(dict1["pre_process_dir"], dict1["seg_model"], dict1["slice_csv_path"], dict1["out_dir"])



    
if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)
    if (opt.test) :
        
        if (opt.STEP == 'ALL') :
        
            # Run Slice Slection, pre-processing and then segmentation 
            slice_selection(dict1["img_dir"], dict1["slice_model"],dict1["slice_csv_path"])
            preprocess(dict1["img_dir"], dict1["pre_process_dir"])
            segmentation(dict1["pre_process_dir"], dict1["seg_model"], dict1["slice_csv_path"], dict1["out_dir"])
        elif (opt.STEP == 'SLICE') :
            slice_selection(dict1["img_dir"], dict1["slice_model"],dict1["slice_csv_path"])
        elif (opt.STEP == 'PREPROCESS') :
            preprocess(dict1["img_dir"], dict1["pre_process_dir"])
        elif (opt.STEP == 'SEGMENT') :
            segmentation(dict1["pre_process_dir"], dict1["seg_model"], dict1["slice_csv_path"], dict1["out_dir"])
            
            
        










