#from google.colab import drive
#drive.mount('/content/drive')
import glob, os, functools
import numpy as np
import pandas as pd
import SimpleITK as sitk
import operator
from scipy import ndimage
from SimpleITK.extra import GetArrayFromImage
from scipy import ndimage
import shutil
import matplotlib as plt
from interpolate import interpolate
from crop_img import crop_top, crop_top_img_only, crop_full_body


def save_data(proj_dir, dataset):
    if dataset == 'train':
        raw_img_dir = proj_dir + '/MDACC_Frontiers_Scans/img'
        raw_seg_dir = proj_dir + '/MDACC_Frontiers_Scans/seg'
        train_img_dir = proj_dir + '/train/img'
        train_seg_dir = proj_dir + '/train/seg'
        if not os.path.exists(train_img_dir):
            os.makedirs(train_img_dir)
        if not os.path.exists(train_seg_dir):
            os.makedirs(train_seg_dir)
        bad_data = []
        for i, img_dir in enumerate(sorted(glob.glob(raw_img_dir + '/*nrrd'))):
            ID = img_dir.split('/')[-1].split('_')[1].split('-')[2]
            print(i, ID)
            save_img_path = train_img_dir + '/MDA' + ID + '.nrrd'
            shutil.move(img_dir, save_img_path)
            try:
                seg_dir = raw_seg_dir + '/HNSCC-01-' + ID + '/muscle.nii.gz'
                save_seg_path = train_seg_dir + '/MDA' + ID + '.nii.gz'
                shutil.move(seg_dir, save_seg_path)
            except Exception as e:
                print(ID, e)
            bad_data.append(ID)
    elif dataset == 'test':
        raw_img_dir = proj_dir + '/HN_C3_Scans_and_Manual_Segmentations'
        test_img_dir = proj_dir + '/test/img'
        test_seg_dir = proj_dir + '/test/seg'
        if not os.path.exists(test_img_dir):
            os.makedirs(test_img_dir)
        if not os.path.exists(test_seg_dir):
            os.makedirs(test_seg_dir)
        bad_data = []
        i = 0
        for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
            data_type = img_dir.split('/')[-1].split('.')[1]
            if data_type == 'seg':
                ID = img_dir.split('/')[-1].split('_')[1].split('-')[2]
                i += 1
                print(i, ID)
                save_seg_path = test_seg_dir + '/MDA' + ID + '.nrrd'
                shutil.move(img_dir, save_seg_path)
            else:
                ID = img_dir.split('/')[-1].split('_')[1].split('-')[2]
                print(ID)
                save_img_path = test_img_dir + '/MDA' + ID + '.nrrd'
                shutil.move(img_dir, save_img_path)


def respace(proj_dir, dataset, new_spacing=(1, 1, 1), crop_shape=(224, 224, 64), 
            img_format='nrrd'):
    """respace data to (1, 1, 3)
    """
    raw_img_dir = proj_dir + '/' + dataset + '/img'
    raw_seg_dir = proj_dir + '/' + dataset + '/seg'
    respace_img_dir = proj_dir + '/' + dataset + '/respace_img'
    respace_seg_dir = proj_dir + '/' + dataset + '/respace_seg'
    crop_img_dir = proj_dir + '/' + dataset + '/crop_img'
    crop_seg_dir = proj_dir + '/' + dataset + '/crop_seg'
    if not os.path.exists(respace_img_dir):
        os.makedirs(respace_img_dir)
    if not os.path.exists(respace_seg_dir):
        os.makedirs(respace_seg_dir)
    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)
    if not os.path.exists(crop_seg_dir):
        os.makedirs(crop_seg_dir)
    count = 0
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        if dataset == 'train':
            seg_dir = raw_seg_dir + '/' + img_id + '.nii.gz'
        elif dataset == 'test':
            seg_dir = raw_seg_dir + '/' + img_id + '.nrrd'
        if os.path.exists(seg_dir):
            count += 1
            print(count, img_id)
            # load img and seg
            img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
            img_interp = interpolate(
                patient_id=img_id, 
                img=img, 
                interpolation_type='linear', #"linear" for image
                new_spacing=new_spacing, 
                return_type='sitk_obj', 
                output_dir=respace_img_dir,
                image_format=img_format)
            # interpolate segs
            seg_interp = interpolate(
                patient_id=img_id, 
                img=seg, 
                interpolation_type='nearest_neighbor', # nearest neighbor for label
                new_spacing=new_spacing, 
                return_type='sitk_obj', 
                output_dir=respace_seg_dir,
                image_format=img_format)


def cropping(proj_dir, dataset, crop_shape=(224, 224, 192), 
             img_format='nrrd'):
    
    respace_img_dir = proj_dir + '/' + dataset + '/respace_img'
    respace_seg_dir = proj_dir + '/' + dataset + '/respace_seg'
    crop_img_dir = proj_dir + '/' + dataset + '/crop_img'
    crop_seg_dir = proj_dir + '/' + dataset + '/crop_seg'
    bad_ids = []
    bad_scans = []
    count = 0
    # get register template
    for img_dir in sorted(glob.glob(respace_img_dir + '/*nrrd')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        seg_dir = respace_seg_dir + '/' + img_id + '.nrrd'
        count += 1
        print(count, img_id)
        # load img and seg
        img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
        z_img = img.GetSize()[2]
        z_seg = seg.GetSize()[2]
        if z_img < 50:
            print('This is an incomplete scan!')
            bad_scans.append(img_id)
        else:
            # crop full body scan
            if z_img > 300:
                img = crop_full_body(img, int(z_img * 0.65))
                seg = crop_full_body(seg, int(z_seg * 0.65))
            else:
                img = img
                seg = seg
            #try:
            print('cropping')
            crop_top(
                patient_id=img_id,
                img=img,
                seg=seg,
                crop_shape=crop_shape,
                return_type='sitk_object',
                output_img_dir=crop_img_dir,
                output_seg_dir=crop_seg_dir,
                image_format=img_format)
            #except Exception as e:
            #    bad_ids.append(img_id)
            #    print(img_id, e)


def main():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/data/train_data'
    step = 'respace'
    if step == 'save_data':
        save_data(dataset, proj_dir)
    elif step == 'respace':
        for dataset in ['train', 'test']:
            respace(proj_dir, dataset)
    elif step == 'crop':
        for dataset in ['train', 'test']:
            cropping(proj_dir, dataset)


if __name__ == '__main__':
    main()





