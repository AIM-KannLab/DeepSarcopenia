import glob
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from src.crop_resize import crop_resize
from src.resize_3d import resize_3d


def preprocess(img_dirs, output_dir):
    """preprocess for test data: respacing and cropping
    """
    bad_data = []
    errors = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        count += 1
        print(count, img_id)
        img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        z_size = img.GetSize()[2]
        crop_shape = (256, 256, z_size)
        try:
            crop_resize(
                patient_id=img_id,
                img=img,
                crop_shape=crop_shape,
                output_dir=output_dir)
        except Exception as e:
            bad_data.append(img_id)
            errors.append(e)
            print(img_id, e)
    print(bad_data, errors)


def preprocess2(img_dirs, output_dir):
    """preprocess for test data: respacing and cropping
    """
    IDs = []
    for img_path in glob.glob(output_dir + '/*nrrd'):
        ID = img_path.split('/')[-1].split('.')[0]
        IDs.append(ID)
    count = 0
    bad_data = []
    errors = []
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        if img_id.split('_'):
            pmrn = img_id.split('_')[0]
            if pmrn not in IDs:
                count += 1
                print(count, img_id)
                img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                z_size = img.GetSize()[2]
                crop_shape = (256, 256, z_size)
                try:
                    crop_resize(
                        patient_id=img_id,
                        img=img,
                        crop_shape=crop_shape,
                        output_dir=output_dir)
                except Exception as e:
                    bad_data.append(img_id)
                    errors.append(e)
                    print(img_id, e)
    print(bad_data, errors)


def main():
    proj_dir = '/mnt/kannlab_rfa/Zezhong'
    OPC_img_dir = proj_dir + '/HeadNeck/data/BWH_TOT/raw_img'
    NonOPC_img_dir = proj_dir + '/HeadNeck/data/NonOPC/raw_img'
    OPC_img_dirs = [i for i in glob.glob(OPC_img_dir + '/*nrrd')]
    NonOPC_img_dirs = [i for i in glob.glob(NonOPC_img_dir + '/*nrrd')]
    img_dirs = OPC_img_dirs + NonOPC_img_dirs

    output_dir = proj_dir + '/c3_segmentation/inference/crop_resize_img'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    preprocess2(img_dirs, output_dir)

def main2():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/internal_test'
    img_dir = proj_dir + '/test_img'
    seg_dir = proj_dir + '/test_seg'

if __name__ == '__main__':

    main()












