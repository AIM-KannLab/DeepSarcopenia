# The z-offset for a slice should represent its offset above or below the C3 slice in mm 
# Slices below the chosen C3 slice should be given negative offsets.
import matplotlib.pyplot as plt
import functools
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import glob



def nrrd_to_npy(proj_dir, dataset):
    """save 2d slice npy arrays from 3d nrrd scans
    """
    img_dir = proj_dir + '/train_data/raw_data/crop_img'
    tr_np_dir = proj_dir + '/train_data/raw_data/train_npy'
    vl_np_dir = proj_dir + '/train_data/raw_data/val_npy'
    ts_np_dir = proj_dir + '/train_data/raw_data/test_npy'
    if not os.path.exists(tr_np_dir):
        os.makedirs(tr_np_dir)
    if not os.path.exists(vl_np_dir):
        os.makedirs(vl_np_dir)
    if not os.path.exists(ts_np_dir):
        os.makedirs(ts_np_dir)
    tr_df_path = proj_dir + '/slice_selection/train.csv'
    vl_df_path = proj_dir + '/slice_selection/val.csv'
    ts_df_path = proj_dir + '/slice_selection/test.csv'
    
    if dataset == 'train':
        np_dir = tr_np_dir
        df_save_path = tr_df_path
    if dataset == 'val':
        np_dir = vl_np_dir
        df_save_path = vl_df_path
    if dataset == 'test':
        np_dir = ts_np_dir
        df_save_path = ts_df_path
        
    df_all = pd.read_csv(proj_dir + '/slice_selection/C3_top_slice_annotation_4.csv')
    print(df_all.shape)

    df_data = df_all[df_all.Datatype.isin([dataset])]
    df_save = pd.DataFrame()
    j = 0
    count = 0
    IDs = []
    exceptions = []    
    for pat_id, c3_slice in zip(df_data['patient_id'], df_data['C3_Manual_slice']):
        ID = 'MDA' + pat_id.split('_')[1].split('-')[2]
        count += 1
        print(count, ID)
        img_path = img_dir + '/' + ID + '.nrrd'
        try:
            img_sitk = sitk.ReadImage(img_path)
            img_arr = sitk.GetArrayFromImage(img_sitk)
            spacing = img_sitk.GetSpacing()[2]
        #    image_array = apply_window(image_array)
            windowed_images = img_arr
            #print(j, pat_id, img_arr.shape, np.max(img_arr), spacing)
            resize_func = functools.partial(
                resize, output_shape=[256, 256], preserve_range=True, 
                anti_aliasing=True, mode='constant')
            series = np.dstack([resize_func(im) for im in windowed_images])
            series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
        #     print(type(series),series.shape)
            for slice_idx in range(img_arr.shape[0]):
                offset = spacing*(slice_idx - c3_slice)
                offset = round(offset, 5)
                np_name = ID + '_' + str(j).zfill(6) + '.npy'
                np_path = np_dir + '/' + np_name
                img_slice = series[slice_idx, :, :, :].astype(np.uint8)
                np.save(np_path, img_slice)
        #         print(npy_path ,im_array.shape,'spacing',spacing)
                df_offset = pd.DataFrame({'NPY_name': np_name, 'ZOffset': offset}, index=[0])
                df_save = pd.concat([df_save, df_offset])
                #df_save = df_save.append(df_offset)
                df_save.to_csv(df_save_path)   
                j = j + 1
        except Exception as e:
            print(e, ID)
            IDs.append(ID)
            exceptions.append(e)
    print(IDs)
    print(exceptions)


def np_stack(proj_dir, dataset):
    tr_np_dir = proj_dir + '/train_data/raw_data/train_npy'
    vl_np_dir = proj_dir + '/train_data/raw_data/val_npy'
    ts_np_dir = proj_dir + '/train_data/raw_data/test_npy'
    if dataset == 'train':
        np_dir = tr_np_dir
        save_path = proj_dir + '/train_data/raw_data/train_data.npy'
    elif dataset == 'val':
        np_dir = vl_np_dir
        save_path = proj_dir + '/train_data/raw_data/val_data.npy'
    elif dataset == 'test':
        np_dir = ts_np_dir
        save_path = proj_dir + '/train_data/raw_data/test_data.npy'

    arr = np.empty([256, 256, 0])
    count = 0
    for np_path in sorted(glob.glob(np_dir + '/*npy')):
        count += 1
        print(count)
        data = np.load(np_path)
        #arr1 = np.reshape(arr1, (256, 256))
        print(data.shape)
        arr = np.concatenate([arr, data], axis=2)
    print('arr shape:', arr.shape)
    np.save(save_path, arr)


if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/data'
    dataset = 'train'

    #nrrd_to_npy(proj_dir, dataset)
    np_stack(proj_dir, dataset)





