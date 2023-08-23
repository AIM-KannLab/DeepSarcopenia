import os
import operator
import numpy as np
import SimpleITK as sitk
from src.data_util import get_arr_from_nrrd, get_bbox, generate_sitk_obj_from_npy_array
#from scipy.ndimage import sobel, generic_gradient_magnitude
from scipy import ndimage
from SimpleITK.extra import GetArrayFromImage
from scipy import ndimage
###import cv2
import matplotlib as plt


def crop_top(patient_id, raw_img, img, seg, return_type, output_img_dir, output_seg_dir):
    """
    Will crop around the center of bbox of label.
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
    # get image, arr, and spacing
    img_arr = sitk.GetArrayFromImage(img)
    img_origin = img.GetOrigin()
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_origin = seg.GetOrigin()
    #assert img_arr.shape == seg_arr.shape, 'img and seg shape do not match!'
    #print('max seg value:', np.max(seg_arr))    
    # get center. considers all blobs
    #bbox = get_bbox(label_arr)
    # returns center point of the label array bounding box
    #Z, Y, X = int(bbox[9]), int(bbox[10]), int(bbox[11]) 
    #print('Original Centroid: ', X, Y, Z)
    
    #find origin translation from label to image
    #print('image origin: ', image_origin, 'label origin: ', label_origin)
    #origin_dif = tuple(np.subtract(seg_origin, img_origin).astype(int))
    #print('origin difference: ', origin_dif)
    #X_shift, Y_shift, Z_shift = tuple(np.add((X, Y, Z), np.divide(origin_dif, (1, 1, 3)).astype(int)))
    #print('Centroid shifted:', X_shift, Y_shift, Z_shift) 
    
    #---cut bottom slices---
    raw_img_spacing = raw_img.GetSpacing()
    print('spacing:', raw_img_spacing)
    print('original shape:', img_arr.shape)
    z, y, x = img_arr.shape

    # for 0.45x0.45 spacing scans
    if raw_img_spacing[0] < 0.7:
        img_arr = img_arr[int(0.4*z):int(0.8*z), :, :]
        seg_arr = seg_arr[int(0.4*z):int(0.8*z), :, :]
        z, y, x = img_arr.shape
        #print(img_arr.shape)
        crop_shape = (256, 256, z)
        if y < 256:
            print('padding img')
            if y % 2 == 0:
                pad = (256-y)//2
                img_arr_crop = np.pad(img_arr, ((0, 0), (pad, pad), (pad, pad)),
                    'constant', constant_values=-1024)
                seg_arr_crop = np.pad(seg_arr, ((0, 0), (pad, pad), (pad, pad)),
                    'constant', constant_values=0)
            else: 
                pad0 = (256-y)//2
                pad1 = (256-y)//2+1
                img_arr_crop = np.pad(img_arr, ((0, 0), (pad0, pad1), (pad0, pad1)),
                    'constant', constant_values=-1024)
                seg_arr_crop = np.pad(seg_arr, ((0, 0), (pad0, pad1), (pad0, pad1)),
                    'constant', constant_values=0)
            print('shape after padding:', img_arr_crop.shape)
        else:
            ## Get center of mass to center the crop in Y plane
            mask_arr = np.copy(img_arr)
            mask_arr[mask_arr > -500] = 1
            mask_arr[mask_arr <= -500] = 0
            mask_arr[mask_arr >= -500] = 1
            #print('mask_arr min and max:', np.amin(mask_arr), np.amax(mask_arr))
            centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
            cpoint = z - crop_shape[2]//2
            #print('cpoint, ', cpoint)
            centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])
            #print('center of mass: ', centermass)
            startx = int(centermass[0] - crop_shape[0]//2)
            starty = int(centermass[1] - crop_shape[1]//2)
            #startx = x//2 - crop_shape[0]//2       
            #starty = y//2 - crop_shape[1]//2
            #startz = int(z - crop_shape[2])
            #print('start X, Y, Z: ', startx, starty, startz)
            # crop and pad array
            if startx < 0:
                img_arr = np.pad(img_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
                    'constant', constant_values=-1024)
                seg_arr = np.pad(seg_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
                    'constant', constant_values=0)
                #img_arr_crop = img_arr[0:crop_shape[2], starty:starty+crop_shape[1], :]
                #seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty+crop_shape[1], :]
            if starty < 0:
                img_arr = np.pad(img_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
                    'constant', constant_values=-1024)
                seg_arr = np.pad(seg_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
                    'constant', constant_values=0)
            img_arr_crop = img_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
            seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty+crop_shape[1], startx:startx+crop_shape[0]]
            print('shape after cropping:', img_arr_crop.shape)
    
    # for 0.95x0.95 spacing patient
    else:
        if z > 250:
            img_arr = img_arr[int(0.65*z):int(0.85*z), :, :]
            seg_arr = seg_arr[int(0.65*z):int(0.85*z), :, :]
        else:
            img_arr = img_arr[int(0.4*z):int(0.8*z), :, :]
            seg_arr = seg_arr[int(0.4*z):int(0.8*z), :, :]
        z, y, x = img_arr.shape
        crop_shape = (256, 256, z)
        if y < 256:
            print('padding img')
            if y % 2 == 0:
                pad = (256-y)//2
                img_arr_crop = np.pad(img_arr, ((0, 0), (pad, pad), (pad, pad)),
                    'constant', constant_values=-1024)
                seg_arr_crop = np.pad(seg_arr, ((0, 0), (pad, pad), (pad, pad)),
                    'constant', constant_values=0)
            else:
                pad0 = (256-y)//2
                pad1 = (256-y)//2+1
                img_arr_crop = np.pad(img_arr, ((0, 0), (pad0, pad1), (pad0, pad1)),
                    'constant', constant_values=-1024)
                seg_arr_crop = np.pad(seg_arr, ((0, 0), (pad0, pad1), (pad0, pad1)),
                    'constant', constant_values=0)
            print('shape after padding:', img_arr_crop.shape)
        else:
            ## Get center of mass to center the crop in Y plane
            mask_arr = np.copy(img_arr) 
            mask_arr[mask_arr > -500] = 1
            mask_arr[mask_arr <= -500] = 0
            mask_arr[mask_arr >= -500] = 1 
            #print('mask_arr min and max:', np.amin(mask_arr), np.amax(mask_arr))
            centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
            cpoint = z - crop_shape[2]//2
            #print('cpoint, ', cpoint)
            centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
            #print('center of mass: ', centermass)
            startx = int(centermass[0] - crop_shape[0]//2)
            starty = int(centermass[1] - crop_shape[1]//2)      
            #startx = x//2 - crop_shape[0]//2       
            #starty = y//2 - crop_shape[1]//2
            #startz = int(z - crop_shape[2])
            #print('start X, Y, Z: ', startx, starty, startz)
            
            # crop and pad array
            if startx < 0:
                img_arr = np.pad(img_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
                    'constant', constant_values=-1024)
                seg_arr = np.pad(seg_arr, ((0, 0), (abs(startx)//2, abs(startx)//2), (0, 0)),
                    'constant', constant_values=0)
                #img_arr_crop = img_arr[0:crop_shape[2], starty:starty+crop_shape[1], :]
                #seg_arr_crop = seg_arr[0:crop_shape[2], starty:starty+crop_shape[1], :]
            if starty < 0:
                img_arr = np.pad(img_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
                    'constant', constant_values=-1024)
                seg_arr = np.pad(seg_arr, ((0, 0), (0, 0), (abs(starty)//2, abs(starty)//2)),
                    'constant', constant_values=0)
            img_arr_crop = img_arr[0:crop_shape[2], starty-50:starty+crop_shape[1]-50, startx:startx+crop_shape[0]]
            seg_arr_crop = seg_arr[0:crop_shape[2], starty-50:starty+crop_shape[1]-50, startx:startx+crop_shape[0]]
            print('shape after cropping:', img_arr_crop.shape)

    #-----normalize CT data signals-------
    #image_arr[image_arr <= -1024] = -1024
    ## strip skull, skull UHI = ~700
    #image_arr[image_arr > 700] = 0
    ## normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    img_arr_crop = np.clip(img_arr_crop, a_min=-175, a_max=275)
    MAX, MIN = img_arr_crop.max(), img_arr_crop.min()
    img_arr_crop = (img_arr_crop - MIN) / (MAX - MIN)

    #print(seg_arr_crop.shape)
    output_img_path = output_img_dir + '/' + patient_id + '.nrrd'
    output_seg_path = output_seg_dir + '/' + patient_id + '.nrrd'
    # save image
    img_sitk = sitk.GetImageFromArray(img_arr_crop)
    img_sitk.SetSpacing(img.GetSpacing())
    img_sitk.SetOrigin(img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_img_path)
    writer.SetUseCompression(True)
    writer.Execute(img_sitk)
    # save label
    seg_sitk = sitk.GetImageFromArray(seg_arr_crop)
    seg_sitk.SetSpacing(seg.GetSpacing())
    seg_sitk.SetOrigin(seg.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_seg_path)
    writer.SetUseCompression(True)
    writer.Execute(seg_sitk)

    
def crop_top_img_only(patient_id, img, crop_shape, return_type, output_dir, image_format):
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
    # get image, arr, and spacing
    image_arr = sitk.GetArrayFromImage(img)
    ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
    #print("image_arr shape: ", image_arr.shape)
    c, y, x = image_arr.shape
    ## Get center of mass to center the crop in Y plane
    mask_arr = np.copy(image_arr) 
    mask_arr[mask_arr > -500] = 1
    mask_arr[mask_arr <= -500] = 0
    mask_arr[mask_arr >= -500] = 1 
    #print("mask_arr min and max:", np.amin(mask_arr), np.amax(mask_arr))
    centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
    cpoint = c - crop_shape[2]//2
    #print("cpoint, ", cpoint)
    centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
    #print("center of mass: ", centermass)
    startx = int(centermass[0] - crop_shape[0]//2)
    starty = int(centermass[1] - crop_shape[1]//2)      
    #startx = x//2 - crop_shape[0]//2       
    #starty = y//2 - crop_shape[1]//2
    startz = int(c - crop_shape[2])
    #print("start X, Y, Z: ", startx, starty, startz)
    
    # cut bottom slices
    image_arr = image_arr[30:, :, :]
    #-----normalize CT data signals-------
    norm_type = 'np_clip'
    #image_arr[image_arr <= -1024] = -1024
    ### strip skull, skull UHI = ~700
    #image_arr[image_arr > 700] = 0
    ## normalize UHI to 0 - 1, all signlas outside of [0, 1] will be 0;
    if norm_type == 'np_interp':
        image_arr = np.interp(image_arr, [-200, 200], [0, 1])
    elif norm_type == 'np_clip':
        #image_arr = np.clip(image_arr, a_min=-200, a_max=200)
        image_arr = np.clip(image_arr, a_min=-175, a_max=275)
        MAX, MIN = image_arr.max(), image_arr.min()
        image_arr = (image_arr - MIN) / (MAX - MIN)

    if startz < 0:
        image_arr = np.pad(
            image_arr,
            ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
            'constant', 
            constant_values=-1024)
        image_arr_crop = image_arr[
            0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
    else:
        image_arr_crop = image_arr[
            0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
    if image_arr_crop.shape[0] < crop_shape[2]:
        print("initial cropped image shape too small:", image_arr_crop.shape)
        print(crop_shape[2], image_arr_crop.shape[0])
        image_arr_crop = np.pad(
            image_arr_crop,
            ((int(crop_shape[2] - image_arr_crop.shape[0]), 0), (0,0), (0,0)),
            'constant',
            constant_values=-1024)
        print("padded size: ", image_arr_crop.shape)
    #print('Returning bottom rows')
    save_dir = output_dir + '/' + patient_id + '.' + image_format
    new_sitk_object = sitk.GetImageFromArray(image_arr_crop)
    new_sitk_object.SetSpacing(img.GetSpacing())
    new_sitk_object.SetOrigin(img.GetOrigin())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(save_dir)
    writer.SetUseCompression(True)
    writer.Execute(new_sitk_object)


def crop_full_body(img, z):
    
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr[z:img.GetSize()[2], :, :]
    new_img = sitk.GetImageFromArray(img_arr)
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetOrigin(img.GetOrigin())
    
    return new_img


