from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np


def get_csa():
    """get C3 muscle area and density
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    df = pd.read_csv(proj_dir + '/internal_test/clinical_data/sum.csv')

    for x in [1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6]:
        if x == 0:
            save_folder = 'pred0'
        if x == 1:
            save_folder = 'pred1' 
        elif x == -1:
            save_folder = 'pred_1'       
        if x == 2:
            save_folder = 'pred2' 
        elif x == -2:
            save_folder = 'pred_2'  
        if x == 3:
            save_folder = 'pred3' 
        elif x == -3:
            save_folder = 'pred_3'  
        if x == 4:
            save_folder = 'pred4' 
        elif x == -4:
            save_folder = 'pred_4'       
        if x == 5:
            save_folder = 'pred5' 
        elif x == -5:
            save_folder = 'pred_5'  
        if x == 6:
            save_folder = 'pred6' 
        elif x == -6:
            save_folder = 'pred_6'  
        pred_dir = proj_dir + '/papers/pred_test/' + save_folder
        csas = []
        count = 0
        for ID, seg_slice in zip(df['ID'], df['seg_slice']):
            count += 1
            ID = str(ID)
            pred_sitk = sitk.ReadImage(pred_dir + '/' + ID + '.nrrd')
            pred_arr = sitk.GetArrayFromImage(pred_sitk)[seg_slice + x, :, :]
            area_per_pixel = pred_sitk.GetSpacing()[0] * pred_sitk.GetSpacing()[1]
            muscle_seg = (pred_arr==1) * 1.0  
            csa = np.sum(muscle_seg) * area_per_pixel/100 # mm2
            csa = round(csa, 2)
            csas.append(csa)
            print(count, ID, csa)
        name = 'pred_csa' + str(x)
        df[name] = csas
    save_path = proj_dir + '/papers/pred_test/test_csa.csv'
    df.to_csv(save_path, index=False)


def get_mae():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    save_path = proj_dir + '/papers/pred_test/test_csa.csv'
    df = pd.read_csv(save_path)
    names = []
    maes = []
    for x in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
        name = 'pred_csa' + str(x)
        mae = mean_absolute_error(df['seg_csa'].values, df[name].values)
        maes.append(mae)
        names.append(name)
    print(maes)
    mean_mae_2 = np.mean(maes[:4])
    mean_mae_4 = np.mean(maes[:8])
    mean_mae_6 = np.mean(maes[:12])
    print('mean_mae_2:', round(mean_mae_2, 2)) 
    print('mean_mae_4:', round(mean_mae_4, 2))
    print('mean_mae_6:', round(mean_mae_6, 2))


def get_delta():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    save_path = proj_dir + '/papers/pred_test/test_csa.csv'
    df = pd.read_csv(save_path)
    deltass = []
    for x in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
        name = 'pred_csa' + str(x)
        deltas = []
        for seg_csa, pred_csa in zip(df['seg_csa'].values, df[name].values):
            delta = (pred_csa - seg_csa) / seg_csa
            deltas.append(delta)
            mean_delta = np.mean(deltas)
        deltass.append(mean_delta)
    print(deltass)

    mean_delta_2 = np.mean(deltass[:4])
    mean_delta_4 = np.mean(deltass[:8])
    mean_delta_6 = np.mean(deltass[:12])
    print('mean_2:', round(mean_delta_2, 2)) 
    print('mean_4:', round(mean_delta_4, 2))
    print('mean_6:', round(mean_delta_6, 2))


if __name__ == '__main__':

    get_mae()
    get_delta()