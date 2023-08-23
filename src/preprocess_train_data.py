import glob, os, functools
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from SimpleITK.extra import GetArrayFromImage
import shutil
from src.interpolate import interpolate
from src.crop_img import crop_top, crop_top_img_only, crop_full_body


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

def get_slice_number(proj_dir):

    raw_img_dir = proj_dir + '/raw_data/img'
    count = 0
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        count += 1
        img_id = img_dir.split('/')[-1].split('.')[0]
        img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        z_spacing = round(img.GetSpacing()[2], 1)
        z_shape = img.GetSize()[2]
        print(count, img_id, z_spacing, z_shape)


def respace_crop(proj_dir):
    """respace data to (1, 1, 3)
    """
    raw_img_dir = proj_dir + '/raw_data/img'
    raw_seg_dir = proj_dir + '/raw_data/seg'
    crop_img_dir = proj_dir + '/raw_data/crop_img'
    crop_seg_dir = proj_dir + '/raw_data/crop_seg'
    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)
    if not os.path.exists(crop_seg_dir):
        os.makedirs(crop_seg_dir)
    count = 0
    bad_data = []
    es = []
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        ID = float(img_id[3:])
        if ID < 100:
            #print(ID)
            seg_dir = raw_seg_dir + '/' + img_id + '.nrrd'
        else:
            #print(ID)
            seg_dir = raw_seg_dir + '/' + img_id + '.nii.gz'
        if img_id in ['MDA0352', 'MDA0415', 'MDA0488', 'MDA0502', 'MDA0513']:
            if os.path.exists(seg_dir):
                count += 1
                print(count, img_id)
                # load img and seg
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
                z_spacing = img.GetSpacing()[2]
                z_shape = img.GetSize()[2]
                respace_img = interpolate(
                    patient_id=img_id, 
                    img=img, 
                    interpolation_type='linear', #"linear" for image
                    new_spacing=(1, 1, z_spacing), 
                    return_type='sitk_obj', 
                    output_dir='')
                # interpolate segs
                respace_seg = interpolate(
                    patient_id=img_id, 
                    img=seg, 
                    interpolation_type='nearest_neighbor', # nearest neighbor for label
                    new_spacing=(1, 1, z_spacing), 
                    return_type='sitk_obj', 
                    output_dir='')
                try:
                    crop_top(
                        patient_id=img_id,
                        raw_img=img,
                        img=respace_img,
                        seg=respace_seg,
                        return_type='sitk_object',
                        output_img_dir=crop_img_dir,
                        output_seg_dir=crop_seg_dir)
                except Exception as e:
                    print(img_id, e)
                    bad_data.append(img_id)
                    es.append(e)
    print(bad_data)
    print(es)


def cropping(proj_dir):
    """respace data to (1, 1, 3)
    """
    respace_img_dir = proj_dir + '/raw_data/respace_img'
    respace_seg_dir = proj_dir + '/raw_data/respace_seg'
    crop_img_dir = proj_dir + '/raw_data/crop_img'
    crop_seg_dir = proj_dir + '/raw_data/crop_seg'
    if not os.path.exists(crop_img_dir):
        os.makedirs(crop_img_dir)
    if not os.path.exists(crop_seg_dir):
        os.makedirs(crop_seg_dir)
    count = 0
    bad_data = []
    for img_dir in sorted(glob.glob(raw_img_dir + '/*nrrd')):
        img_id = img_dir.split('/')[-1].split('.')[0]
        seg_dir = raw_seg_dir + '/' + img_id + '.nrrd'
        if os.path.exists(seg_dir):
            count += 1
            print(count, img_id)
            # load img and seg
            img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            seg = sitk.ReadImage(seg_dir, sitk.sitkFloat32)
            try:
                crop_top(
                    patient_id=img_id,
                    img=img,
                    seg=seg,
                    return_type='sitk_object',
                    output_img_dir=crop_img_dir,
                    output_seg_dir=crop_seg_dir)
            except Exception as e:
                print(img_id, e)
                bad_data.append(img_id)
    print(bad_data)


def main():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/data/train_data'
    step = 'respace_crop'
    if step == 'save_data':
        save_data(dataset, proj_dir)
    elif step == 'get_slice_number':
        get_slice_number(proj_dir)
    elif step == 'respace_crop':
        respace_crop(proj_dir)
    elif step == 'cropping':
        cropping(proj_dir)

if __name__ == '__main__':
    main()





