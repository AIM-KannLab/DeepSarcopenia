import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from opts import parse_opts, get_args
from src.slice_area_density import get_c3_slice_area, get_c3_slice_density


def get_CSA(pre_process_dir, out_dir, slice_csv_path, seg_csa_path):
    """get C3 muscle area and density
    """
    df_slice = pd.read_csv(slice_csv_path)
    
    pred_csas = []
    pred_csds = []
    seg_csas = []
    seg_csds = []
    pred_voxels = []
    seg_voxels = []
    count = 0
    for ID, pred_slice in zip(df_slice['Patient_id'], df_slice['C3_predict_slice']):
        count += 1
        ID = str(ID)
        # prediction CSA
        csa, csd, tot_voxel = get_c3_slice_area(
            patient_id=ID,
            c3_slice=pred_slice,
            img_dir=pre_process_dir,
            seg_dir=out__dir)
        pred_csa = round(csa, 2)
        pred_csd = round(csd, 2)
        pred_voxel = tot_voxel
        pred_csas.append(pred_csa)
        pred_csds.append(pred_csd)
        pred_voxels.append(pred_voxel)

        print(count, ID, pred_csa, pred_voxel, pred_csa, pred_voxel)
    #df['seg_csa'], df['pred_csa'], df['seg_csd'], df['pred_csd'] = [seg_csas, pred_csas, seg_csds, pred_csds]
    df_slice['seg_area'], df_slice['seg_csd'], df_slice['seg_voxel'] = [pred_csa, pred_csd, pred_voxel]

    df_slice.to_csv(seg_csa_path, index=False)


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    
    #proj_dir = '/mnt/kannlab_rfa/Zezhong'

    opt = parse_opts()
    dict1 = get_args(opt)
    get_CSA(dict1["pre_process_dir"],dict1["out_dir"],dict1["slice_csv_path"],dict1["seg_csa_path"])
