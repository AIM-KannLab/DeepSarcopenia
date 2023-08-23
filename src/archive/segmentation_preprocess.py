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
#import cv2
import matplotlib as plt

#### NOTE: ensure the correct lirabry path
from scripts.interpolate import interpolate
from scripts.data_util import get_arr_from_nrrd, get_bbox, generate_sitk_obj_from_npy_array
from scripts.crop_img import crop_top_img_only, crop_full_body
from scripts.resize_3d import resize_3d


def preprocess(raw_img_dir, crop_img_dir, new_spacing=(1, 1, 3), crop_shape=(224, 224, 48), 
               interp_type='linear', img_format='nrrd'):
    """preprocess for test data: respacing and cropping
    """
    img_dirs = [i for i in sorted(glob.glob(raw_img_dir + '/*nrrd'))]
    img_ids = []
    bad_ids = []
    bad_scans = []
    count = 0
    for img_dir in img_dirs:
        img_id = img_dir.split('/')[-1].split('.')[0]
        img_ids.append(img_id)
        count += 1
        print(count, img_id)
        # load img and seg
        img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        # --- crop full body scan ---
        z_img = img.GetSize()[2]
        if z_img < 105:
            print('This is an incomplete scan!')
            bad_scans.append(img_id)
        else:
            if z_img > 200:
                img = crop_full_body(img, int(z_img * 0.65))
            try:
                # --- interpolation for image and seg to 1x1x3 ---
                # interpolate images
                print('interpolate')
                img_interp = interpolate(
                    patient_id=img_id,
                    path_to_nrrd=img_dir,
                    interpolation_type=interp_type, #"linear" for image
                    new_spacing=new_spacing,
                    return_type='sitk_obj',
                    output_dir='',
                    image_format=img_format)
                print('cropping')
                crop_top_img_only(
                    patient_id=img_id,
                    img=img_interp,
                    crop_shape=crop_shape,
                    return_type='sitk_object',
                    output_dir=crop_img_dir,
                    image_format=img_format)
            except Exception as e:
                bad_ids.append(img_id)
                print(img_id, e)
    print('bad ids:', bad_ids)
    print('incomplete scans:', bad_scans)


if __name__ == '__main__':

    prepross(proj_dir)



