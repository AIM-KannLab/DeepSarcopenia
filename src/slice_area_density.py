import numpy as np
import SimpleITK as sitk
from src.image_processing.image_window import get_image_path_by_id


def get_c3_slice_area(patient_id, c3_slice, img_dir, seg_dir):
       
    img_path = img_dir + '/' + patient_id + '.nrrd'
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)[c3_slice, :, :]
    #img_arr = np.clip(img_arr, a_min=-175, a_max=275)
    #max, min = img_arr.max(), img_arr.min()
    #img_arr = (img_arr - min) / (max - min)

    seg_path = seg_dir + '/' + patient_id + '.nrrd'
    seg_sitk =  sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg_sitk)[c3_slice, :, :]
    
    area_per_pixel = seg_sitk.GetSpacing()[0] * seg_sitk.GetSpacing()[1]
    muscle_seg = (seg_arr==1) * 1.0  
    muscle_area = np.sum(muscle_seg) * area_per_pixel/100
    tot_voxel = np.sum(muscle_seg)
    #print('total voxel:', np.sum(muscle_seg), 'spacing:', seg_sitk.GetSpacing())

    muscle_hu = np.sum(muscle_seg*img_arr)/np.sum(muscle_seg)
   
    return muscle_area, muscle_hu, tot_voxel


