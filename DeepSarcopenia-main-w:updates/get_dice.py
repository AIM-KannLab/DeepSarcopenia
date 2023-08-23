import numpy as np
import os
import pandas as pd
import glob
import SimpleITK as sitk
from opts import parse_opts, get_args
from src.slice_area_density import get_c3_slice_area


def get_dice(proj_dir, img_dir,seg_dir,out_dir,test_csv_path, dice_csv):
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    #img_dir = proj_dir + '/internal_test/prepro_img'
    #seg_dir = proj_dir + '/internal_test/prepro_seg'
    #pred_dir = proj_dir + '/internal_test/pred_seg'
    #img_dir = proj_dir + '/internal_test/yash_img'
    #seg_dir = proj_dir + '/internal_test/yash_seg'
    #pred_dir = proj_dir + '/internal_test/yash_pred'
    pred_dir = out_dir
    
   # slice_csv_path = proj_dir + '/internal_test/slice_model/slice_tot.csv'
    slice_csv_path = test_csv_path
 
    inters = []
    sums = []
    unions = []
    dices = []
    df = pd.read_csv(slice_csv_path)
    for ID, pred_slice, seg_slice in zip(df['ID'], df['pred_slice'], df['seg_slice']):
        pred_path = pred_dir + '/' + ID + '.nrrd'
        seg_path = seg_dir + '/' + ID + '.nrrd'
        pred = sitk.ReadImage(pred_path)
        pred = sitk.GetArrayFromImage(pred)
        pred = pred[pred_slice, :, :]
        seg = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(seg)
        seg = seg[seg_slice, :, :]
        #pred = np.where(pred != 0.0, 1, 0)
        #seg = np.where(seg != 0.0, 1, 0)
        dice = (2*np.sum(seg*pred)) / (np.sum(seg) + np.sum(pred))
        print(round(dice, 3))
        #pred = np.where(pred != 0.0, 1, 0)
        #seg = np.where(seg != 0.0, 1, 0)
        assert pred.shape == seg.shape, print('different shape')
        pred = pred.astype(bool)
        seg = seg.astype(bool)
        volume_sum = seg.sum() + pred.sum()
        volume_intersect = (seg & pred).sum()
        volume_union = volume_sum - volume_intersect
        #dice = 2*volume_intersect / volume_sum
        #dice = round(dice, 3)
        #print(ID, dice)
        inters.append(volume_intersect)
        sums.append(volume_sum)
        unions.append(volume_union)
        dices.append(dice)
    print('dice:', dices)
    print('median dice:', np.nanmedian(dices))
    print('mean dice:', round(np.nanmean(dices), 3))
    dsc_agg = round(2*sum(inters)/sum(sums), 3)
    jaccard_agg = round(sum(inters)/sum(unions), 3)
    print('aggregated dice score:', dsc_agg)
    print('aggregated jaccard score:', jaccard_agg)
    df['dice'] = dices
  #  df.to_csv(proj_dir + '/internal_test/slice_model/slice_yash.csv')
    df.to_csv(dice_csv)




if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)
    
    get_dice(dict1['proj_dir'],dict1['img_dir'],dict1['manual_seg_dir'],dict1['out_dir'],dict1['test_csv_path'],dict1['dice_csv'])
    #get_CSA()



