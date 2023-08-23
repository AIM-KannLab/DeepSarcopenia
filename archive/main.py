import argparse
import warnings
import os 
from scripts.infer_selection import test_slice_selection
import pandas as pd
from scripts.image_processing.image_window import get_image_path_by_id, apply_window
from scripts.image_processing.slice_array_from_nifty import get_C3_seg_array_by_id
import SimpleITK as sitk
import numpy as np
from pprint import pprint
from scripts.infer_segmentation import test_segmentation
from scripts.preprocess import preprocess
from scripts.image_processing.slice_area_density import get_c3_slice_area, get_c3_slice_density


def slice_selection(proj_dir, dataset):
    """
    Test the Slice Selction Model
    Args:
        Input Scans -- nrrd files
        Model -- C3_Top_Selection_Model_Weight.hdf5 
        Output -- C3_Top_Slice_Prediction.csv' 
    """
    if dataset == 'OPC':
        folder = 'BWH'
    elif dataset == 'NonOPC':
        folder = 'NonOPC'
    raw_img_dir = proj_dir + '/HeadNeck/data/' + folder + '/raw_img'
    slice_model = 'C3_Top_Selection_Model_Weight.hdf5'
    slice_model_path = proj_dir + '/c3_segmentation/model/test/' + slice_model
    slice_csv = dataset + '_C3_top_slice_pred.csv'
    slice_csv_path = proj_dir + '/c3_segmentation/output/' + slice_csv 
    print('--- slice selection ---')
    test_slice_selection(
        image_dir=raw_img_dir, 
        model_weight_path=slice_model_path, 
        csv_write_path=slice_csv_path)


def preprocess_data(proj_dir):
    if dataset == 'OPC':
        folder = 'BWH'
    elif dataset == 'NonOPC':
        folder = 'NonOPC'
    OPC_img_dir = proj_dir + '/HeadNeck/data/BWH/raw_img'
    NonOPC_img_dir = proj_dir + '/HeadNeck/data/NonOPC/raw_img'
    crop_img_dir = proj_dir + '/c3_segmentation/data/segmentation/crop_img'
    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)
    print('--- C3 segmentation ---')
    for raw_img_dir in [OPC_img_dir, NonOPC_img_dir]:
        preprocess(raw_img_dir, crop_img_dir)

def combine_csv():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/output/'
    df_OPC = pd.read_csv(proj_dir + 'OPC_C3_top_slice_pred.csv')
    df_NonOPC = pd.read_csv(proj_dir + 'NonOPC_C3_top_slice_pred.csv')
    df = pd.concat([df_OPC, df_NonOPC])
    df.to_csv(proj_dir + 'C3_top_slice_pred.csv', index=False)


def segmentation():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'    
    img_dir = proj_dir + '/data/segmentation/img'
    seg_model = 'C3_Top_Segmentation_Model_Weight.hdf5'
    seg_model_path = proj_dir + '/model/test/' + seg_model
    slice_csv_path = proj_dir + '/output/C3_top_slice_pred.csv'
    output_dir = proj_dir + '/output/pred'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('--- C3 segmentation ---')
    test_segmentation(
        image_dir=img_dir,
        model_weight_path=seg_model_path,
        l3_slice_csv_path=slice_csv_path,
        output_dir=output_dir)


def get_area(proj_dir, dataset):
    """get C3 muscle area and density
    """
    slice_csv = dataset + '_C3_top_slice_pred.csv'
    area_csv = dataset + '_C3_body_comp_area_density.csv'
    if dataset == 'OPC':
        folder = 'BWH'
    elif dataset == 'NonOPC':
        folder = 'NonOPC'n
    crop_img_dir = proj_dir + '/HeadNeck/data/' + folder + '/crop_img'
    slice_csv_path = proj_dir + '/c3_segmentation/output/' + slice_csv
    area_csv_path = proj_dir + '/c3_segmentation/output/' + area_csv
    output_seg_dir = proj_dir + '/c3_segmentation/output/' + dataset
    print('--- get C3 muscle cross sectional area ---')
    df_infer = pd.read_csv(slice_csv_path)
    df_init = pd.DataFrame()
    IDs = []
    for idx in range(df_infer.shape[0]):
        try:
            ID = str(df_infer.iloc[idx, 1])
            c3_slice_auto = df_infer.iloc[idx, 2]
            muscle_area, sfat_area, vfat_area = get_c3_slice_area(
                patient_id=ID, 
                c3_slice=c3_slice_auto, 
                seg_dir=output_seg_dir)
            muscle_density, sfat_density, vfat_density = get_c3_slice_density(
                patient_id=ID, 
                c3_slice=c3_slice_auto, 
                seg_dir=output_seg_dir, 
                img_dir=crop_img_dir)
            #Data Frame rows for writing into a CSV File
            df_inter1 = pd.DataFrame({
                'patient_id': ID,
                'muscle_auto_segmentation_area': round(muscle_area, 2),
                'muscle_auto_edensity': round(muscle_density, 2)
                }, index=[0])
            df_init = df_init.append(df_inter1)
            df_init.to_csv(area_csv_path)
            print(idx,'th', ID, 'writen to', area_csv_path)
        except Exception as e:
            print(ID, e)
            IDs.append(ID)
    print('bad data:', IDs)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    proj_dir = '/mnt/kannlab_rfa/Zezhong'
    dataset = 'NonOPC'
    
    #combine_csv()
    preprocess_data(proj_dir)
    #segmentation()
    #get_area(proj_dir, dataset)
    #save_img_slice(proj_dir, dataset)













