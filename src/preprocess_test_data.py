import glob, os, functools
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from SimpleITK.extra import GetArrayFromImage
import shutil
from interpolate import interpolate
from src.crop_img import crop_top, crop_top_img_only, crop_full_body
from src.resize_3d import resize_3d

### Set the proj_dir, img_dir, and seg_dir folder paths in the main function
### The script preprocess the raw scans, the steps involve respacing the files to 1x1, cropping by 256x256, and resizing to 512x512 along XY planes

def respace_crop_resize(img_dir, seg_dir, save_img_dir, save_seg_dir):
    count = 0
    bad_data = []
    es = []
    for img_path in sorted(glob.glob(img_dir + '/*nrrd')):
        ID = img_path.split('/')[-1].split('.')[0]
        seg_path = seg_dir + '/' + ID + '.nrrd'
        if os.path.exists(seg_path):
            count += 1
            print(count, ID)
            img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)
            z_spacing = img.GetSpacing()[2]
            z_shape = img.GetSize()[2]
            crop_shape = (256, 256, z_shape)
            respace_img = interpolate(
                patient_id=ID, 
                img=img, 
                interpolation_type='linear', #"linear" for image
                new_spacing=(1, 1, z_spacing), 
                return_type='sitk_obj', 
                output_dir='')
            respace_seg = interpolate(
                patient_id=ID, 
                img=seg, 
                interpolation_type='nearest_neighbor', # nearest neighbor for label
                new_spacing=(1, 1, z_spacing), 
                return_type='sitk_obj', 
                output_dir='')
            try:
                crop_resize(
                    patient_id=ID,
                    img=respace_img,
                    seg=respace_seg,
                    crop_shape=crop_shape,
                    save_img_dir=save_img_dir,
                    save_seg_dir=save_seg_dir)
            except Exception as e:
                print(ID, e)
                bad_data.append(ID)
                es.append(e)
    print(bad_data)
    print(es)


def crop_resize(patient_id, img, seg, crop_shape, save_img_dir, save_seg_dir):
    """
    Will center the image and crop top of image after it has been registered.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)
    c, y, x = img_arr.shape
    ## Get center of mass to center the crop in Y plane
    mask_arr = np.copy(img_arr) 
    mask_arr[mask_arr > -500] = 1
    mask_arr[mask_arr <= -500] = 0
    mask_arr[mask_arr >= -500] = 1 
    centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
    cpoint = c - crop_shape[2]//2
    centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
    startx = int(centermass[0] - crop_shape[0]//2)
    starty = int(centermass[1] - crop_shape[1]//2)      
    #startx = x//2 - crop_shape[0]//2       
    #starty = y//2 - crop_shape[1]//2
    startz = int(c - crop_shape[2])
    #print("start X, Y, Z: ", startx, starty, startz)
    
    #image_arr = image_arr[30:, :, :]
    norm_type = 'np_clip'
    if norm_type == 'np_interp':
        img_arr = np.interp(img_arr, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        #img_arr = np.clip(img_arr, a_min=-200, a_max=200)
        img_arr = np.clip(img_arr, a_min=-175, a_max=275)
        MAX, MIN = img_arr.max(), img_arr.min()
        img_arr = (img_arr - MIN) / (MAX - MIN)

    if startz < 0:
        img_arr = np.pad(img_arr, ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 'constant', constant_values=-1024)
        seg_arr = np.pad(seg_arr, ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 'constant', constant_values=-1024)
        img_arr_crop = img_arr[0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
        seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
    elif startx < 0 :
        img_arr_crop = img_arr[0:crop_shape[2], 0:y, 0:x]
        seg_arr_crop = seg_arr[0:crop_shape[2], 0:y, 0:x]
    else:
        img_arr_crop = img_arr[0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
        seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
    print('img shape after cropping:', img_arr_crop.shape)
    print('seg shape after cropping:', seg_arr_crop.shape)
    
    # crop and pad array
    # if startx < 0:
    #     img_arr = np.pad(img_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
    #         'constant', constant_values=-1024)
    #     seg_arr = np.pad(seg_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
    #         'constant', constant_values=0)
    # if starty < 0:
    #     img_arr = np.pad(img_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
    #         'constant', constant_values=-1024)
    #     seg_arr = np.pad(seg_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
    #         'constant', constant_values=0)
    # img_arr_crop = img_arr[0:crop_shape[2], starty-50:starty+crop_shape[1]-50, startx:startx+crop_shape[0]]
    # seg_arr_crop = seg_arr[0:crop_shape[2], starty-50:starty+crop_shape[1]-50, startx:startx+crop_shape[0]]
         
    # get stik img for cropped img
    crop_img = sitk.GetImageFromArray(img_arr_crop)
    crop_img.SetSpacing(img.GetSpacing())
    crop_img.SetOrigin(img.GetOrigin())
    crop_seg = sitk.GetImageFromArray(seg_arr_crop)
    crop_seg.SetSpacing(seg.GetSpacing())
    crop_seg.SetOrigin(seg.GetOrigin())

    # resize img back to 512x512 for segmentation model
    resize_shape = (512, 512, crop_img.GetSize()[2])
    resize_img = resize_3d(img_nrrd=crop_img, interp_type='linear', output_size=resize_shape)
    resize_seg = resize_3d(img_nrrd=crop_seg, interp_type='nearest_neighbor', output_size=resize_shape)

    # save img
    save_img_path = save_img_dir + '/' + patient_id + '.nrrd'
    print('new spacing:', resize_img.GetSpacing())
    resize_img.SetSpacing(resize_img.GetSpacing())
    resize_img.SetOrigin(resize_img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(save_img_path)
    writer.SetUseCompression(True)
    writer.Execute(resize_img)
    # save seg
    save_seg_path = save_seg_dir + '/' + patient_id + '.nrrd'
    resize_seg.SetSpacing(resize_seg.GetSpacing())
    resize_seg.SetOrigin(resize_seg.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(save_seg_path)
    writer.SetUseCompression(True)
    writer.Execute(resize_seg)


def main():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/internal_test'
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/data/mdacc_data'
    img_dir = proj_dir + '/raw_img'
    seg_dir = proj_dir + '/raw_seg'
    save_img_dir = proj_dir + '/prepro_img'
    save_seg_dir = proj_dir + '/prepro_seg'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

    respace_crop_resize(img_dir, seg_dir, save_img_dir, save_seg_dir)

if __name__ == '__main__':
    main()
