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
from opts import parse_opts, get_args



def draw_contour(img_dir, csv_path, seg_dir, save_dir,visual_csv_file):
    #csv_path = proj_dir + '/inference/clinical_files/C3_top_slice_pred.csv'
    #img_dir = proj_dir + '/inference/img'
    #seg_dir = proj_dir + '/inference/pred'
    #img_dir = proj_dir + '/inference/crop_resize_img'
    #seg_dir = proj_dir + '/inference/pred_new'
    #save_dir = proj_dir + '/visualize/C3_contour'
    #save_dir = proj_dir + '/papers/inference_contour'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.read_csv(csv_path)
    count = 0
    bad_data = []
    IDs = []
    errors = []
    for ID, Slice in zip(df['patient_id'], df['C3_Predict_slice']):
        #if ID in ['10091115443', '10029404836', '10023848707']:
        if str(ID) in ['10034201979']:
            count += 1
            print(count, ID)
            IDs.append(ID)
            try:
                ID = str(ID)
                img_path = img_dir + '/' + ID + '.nrrd'
                seg_path = seg_dir + '/' + ID + '.nrrd'
                save_path = save_dir + '/' + ID + '.png'
                img_nrrd = sitk.ReadImage(img_path)
                img_arr = sitk.GetArrayFromImage(img_nrrd)
                seg_nrrd = sitk.ReadImage(seg_path)
                seg_arr = sitk.GetArrayFromImage(seg_nrrd)
                img_slice = img_arr[Slice, :, :]
                seg_slice = seg_arr[Slice, :, :]
                # remove small blobs in segmentation
                seg_slice = skimage.morphology.remove_small_objects(seg_slice.astype(bool), min_size=100)
                seg_slice = seg_slice.astype(seg_slice.dtype)
                # generate contour with CV2
                img = np.uint8(img_slice*255)
                seg = np.uint8(seg_slice*255)
                #contour, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                main = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.drawContours(
                    image=main,
                    contours=contour,
                    contourIdx=-1,
                    color=(255, 0, 0),
                    thickness=2,
                    lineType=16)
                cv2.imwrite(save_path, main)
            except Exception as e:
                print(ID, e)
                bad_data.append(ID)
                errors.append(e)
    print('bad_data:', bad_data)
    print('errors:', errors)
    df = pd.DataFrame({'ID': IDs})
    #df.to_csv(proj_dir + '/visualize/patient_list.csv', index=False)
    df.to_csv(visual_csv_file, index=False)


def save_slice(img_dir, csv_path,seg_dir,visual_save_dir ):
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    #csv_path = proj_dir + '/output/C3_top_slice_pred.csv'
    #img_dir = proj_dir + '/data/segmentation/img'
    #seg_dir = proj_dir + '/output/pred'
    #save_img_dir = proj_dir + '/visualize/img'
    #save_seg_dir = proj_dir + '/visualize/seg'
    save_img_dir = visual_save_dir + '/img'
    save_seg_dir = visual_Save_dir + '/seg'
    
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)
    df = pd.read_csv(csv_path)
    count = 0
    bad_data = []
    for ID, Slice in zip(df['patient_id'], df['C3_Predict_slice']):
        count += 1
        print(count, ID)
        for data_dir, save_dir in zip([img_dir, seg_dir], [save_img_dir, save_seg_dir]):
            try:
                data_path = data_dir + '/' + ID + '.nrrd'
                save_path = save_dir + '/' + ID + '.nrrd'
                if os.path.exists(save_path):
                    print('data exists')
                else:
                    nrrd = sitk.ReadImage(data_path)
                    arr = sitk.GetArrayFromImage(nrrd)
                    arr_slice = arr[Slice, :, :]
                    img_sitk = sitk.GetImageFromArray(arr_slice)
                    img_sitk.SetSpacing(nrrd.GetSpacing())
                    img_sitk.SetOrigin(nrrd.GetOrigin())
                    writer = sitk.ImageFileWriter()
                    writer.SetFileName(save_path)
                    writer.SetUseCompression(True)
                    writer.Execute(img_sitk)
            except Exception as e:
                print(ID)
                bad_data.append(ID)
    print('bad_data:', bad_data)


def get_subset(proj_dir,meta_save_path):
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    save_dir = proj_dir + '/visualize/C3_contour_review'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #df = pd.read_csv(proj_dir + '/clinical/meta.csv', encoding='unicode_escape', low_memory=False)
    df = pd.read_csv(meta_save_path, encoding='unicode_escape', low_memory=False)
    
    # ---exclude surgery and induction cases---
    df = df[~df['Pre-RT Neck Dissection'].isin(['Yes'])]
    df = df[~df['Pre-RT Primary Resection'].isin(['Yes'])]
    df = df[~df['Pre-RT Surgery'].isin(['Yes'])]
    df = df[~df['Radiation adjuvant to surgery'].isin(['Yes'])]
    #df = df[~df['Induction Chemotherapy'].isin(['Yes'])]
    # ---include only height info available patient---
    # weight in lbs, need to convert to kg (1 lbs = 0.454 kg)
    df['Weight'] = df['Pre-treatment Weight in Pounds'].values * 0.454
    df['BMI'] = df['Pre-treatment BMI'].to_list()
    df['Height'] = np.sqrt(df['Weight'] / df['BMI'])
    print(df['Height'].count())
    df = df[df['Height'].notna()]
    print('case number:', df.shape[0])
    df.to_csv(proj_dir + '/visualize/clinical_meta.csv')

    count = 0
    IDs = []
    pmrns = []
    #print(df['PMRN'].to_list())
    for img_path in glob.glob(proj_dir + '/visualize/C3_contour/*png'):
        pmrn = img_path.split('/')[-1].split('.')[0]
        #print(ID)
        if pmrn.split('_'):
            pmrn = int(pmrn.split('_')[0])
        else:
            pmrn = int(pmrn)
        if pmrn in df['PMRN'].to_list():
            count += 1
            print(count, pmrn)
            ID = str(count).zfill(3) + '_' + str(pmrn)
            IDs.append(ID)
            pmrns.append(pmrn)
            shutil.copyfile(img_path, save_dir + '/' + ID + '.png')
    df = pd.DataFrame({'PMRN': pmrns, 'ID': IDs})
    #df.drop_duplicates('ID', keep='first', inplace=True)
    df.to_csv(proj_dir + '/visualize/patient_list.csv', index=False)


def rename_cases(proj_dir):
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/visualize'
    save_dir = proj_dir + '/C3_contour_review2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count = 0
    id0s = []
    id1s = []
    for img_path in sorted(glob.glob(proj_dir + '/C3_contour_review/*png')):
        id0 = img_path.split('/')[-1].split('.')[0]
        id1 = str(count).zfill(3)
        print(id0, id1)
        count += 1
        id0s.append(id0)
        id1s.append(id1)
        shutil.copyfile(img_path, save_dir + '/' + id1 + '.png')
    df = pd.DataFrame({'PMRN': id0s, 'ID': id1s})
    #df.to_csv(proj_dir + '/patient_list.csv', index=False)
    #df.to_csv(proj_dir + '/patient_list.csv', index=False)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)
    
    #save_img_slice(dict1["pre_process_path"],dict1["slice_csv_path"], dict1["out_dir"],dict1["visual_save_dir"])
    draw_contour(dict1["pre_process_path"],dict1["slice_csv_path"], dict1["out_dir"],dict1["visual_save_dir"],dict1["visual_csv_file"])
    #get_subset(dict1["visual_save_dir"], dict1["seg_meta_path")
    #rename_cases(dict1["visual_save_dir"],dict1["visual_csv_file"])



