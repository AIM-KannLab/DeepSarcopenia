import numpy as np
import os
import pandas as pd
import glob
from PIL import Image, ImageOps
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2
import skimage
import shutil


def draw_contour():
    dataset = 'inference'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    if dataset == 'val':
        csv_path = proj_dir + '/validation/clinical/sum.csv'
        img_dir = proj_dir + '/validation/yash_img'
        seg_dir = proj_dir + '/validation/yash_seg'
        pred_dir = proj_dir + '/validation/yash_pred'
        save_dir = proj_dir + '/papers/val_contour'
    elif dataset == 'test':
        csv_path = proj_dir + '/internal_test/clinical_data/sum_yash.csv'
        img_dir = proj_dir + '/internal_test/yash_img'
        seg_dir = proj_dir + '/internal_test/yash_seg'
        pred_dir = proj_dir + '/internal_test/yash_pred'
        save_dir = proj_dir + '/papers/test_contour'
    elif dataset == 'inference':
        csv_path = proj_dir + '/inference/clinical_files/C3_top_slice_pred.csv'
        img_dir = proj_dir + '/inference/crop_resize_img'
        seg_dir = proj_dir + '/inference/pred_new'
        save_dir = proj_dir + '/papers/inference_contour'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    df = pd.read_csv(csv_path)
    #print(df)
    count = 0
    #for ID, pred_slice, seg_slice in zip(df['ID'], df['pred_slice'], df['seg_slice']):
    for ID, pred_slice in zip(df['ID'], df['pred_slice']):
        #if ID in ['10091115443', '10029404836', '10023848707']:
        if ID in ['10034201979']:
            count += 1
            print(count, ID)
            ID = str(ID)
            try:
                img_nrrd = sitk.ReadImage(img_dir + '/' + ID + '.nrrd')
                img_arr = sitk.GetArrayFromImage(img_nrrd)
                img_slice = img_arr[seg_slice, :, :]
                img = np.uint8(img_slice*255)                
                main = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if dataset in ['val', 'test']:
                    print(dataset)
                    seg_nrrd = sitk.ReadImage(seg_dir + '/' + ID + '.nrrd')
                    seg_arr = sitk.GetArrayFromImage(seg_nrrd)
                    seg_slice = seg_arr[seg_slice, :, :]
                    seg = np.uint8(seg_slice*255)
                    seg_contour, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(
                        image=main,
                        #contours=[seg_contour, pred_contour],
                        contours=seg_contour,
                        contourIdx=-1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=16)
                    pass

                pred_nrrd = sitk.ReadImage(pred_dir + '/' + ID + '.nrrd')
                pred_arr = sitk.GetArrayFromImage(pred_nrrd)
                pred_slice = pred_arr[pred_slice, :, :]     
                pred = np.uint8(pred_slice*255)
                # remove small blobs in segmentation
                pred_slice = skimage.morphology.remove_small_objects(pred_slice.astype(bool), min_size=100)
                pred_slice = pred_slice.astype(pred_slice.dtype)    
                pred_contour, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(
                    image=main,
                    #contours=[seg_contour, pred_contour],
                    contours=pred_contour,
                    contourIdx=-1,
                    color=(255, 0, 0),
                    thickness=2,
                    lineType=16)
                cv2.imwrite(save_dir + '/' + ID + '.png', main)
            except Exception as e:
                print(ID, e) 

if __name__ == '__main__':

    draw_contour()
