from sklearn.metrics import cohen_kappa_score
import os
import numpy as np
import pandas as pd
import shutil
from opts import parse_opts, get_args



def inter_rater(proj_dir, reviewer_1_file, reviewer_2_file):
    """
    Use Cohen Kappa to calculate inter rater variability
    dichotomize scores 0, 1 and 2, 3
    Kappa < 0, Less than chance agreement
    0.10 - 0.20, Slight agreement
    0.21 - 0.40, Fair agreement
    0.41 - 0.60, Moderate agreement
    0.61 - 0.80, Substantial agreement
    0.81 - 0.99, Almost perfect agreement
    """
    df1 = pd.read_csv(proj_dir + '/'+ reviewer_1_file)
    df2 = pd.read_csv(proj_dir + '/' + reviewer_2_file)
    # drop C3 score 0 as they are incomplete scans
    df0 = df1[df1['C3 Acceptability']==0]
    list0 = df0['ID'].to_list()
    print(list0)
    df1 = df1[~df1['ID'].isin(list0)]
    df2 = df2[~df2['ID'].isin(list0)]
    print(df1)
    print(df2)
    old_scores1 = df1['C3 Acceptability'].to_list()
    old_scores2 = df2['C3 Acceptability'].to_list()
    print(old_scores1)
    print(old_scores2)
    old_scoress = [old_scores1, old_scores2]
    new_scoress = []
    for old_scores in old_scoress:
        new_scores = []
        for old_score in old_scores:
            if old_score < 2:
                new_score = 0
            else:
                new_score = 1
            new_scores.append(new_score)
        new_scoress.append(new_scores)
    print(new_scoress[0])
    print(new_scoress[1])
    kappa1 = cohen_kappa_score(old_scoress[0], old_scoress[1])
    kappa2 = cohen_kappa_score(new_scoress[0], new_scoress[1])
    print('Cohen Kappa score:', round(kappa1, 3), round(kappa2, 3))


def check_scans(proj_dir, save_checked_scans_dir,reviewer_1_file,reviewer_2_file):
    #proj_dir = '/mnt/kannlab_rfa/Zezhong/C3_segmentation'
    save_dir = proj_dir + '/'+ save_checked_scans_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
  #  df1 = pd.read_csv(proj_dir + '/visualize/patient_list_Ben.csv')
  #  df2 = pd.read_csv(proj_dir + '/visualize/patient_list_Frank.csv')
    df1 = pd.read_csv(proj_dir + '/' + reviewer_1_file)
    df2 = pd.read_csv(proj_dir + '/' + reviewer_2_file)
  
    df1 = df1[df1['C3 Acceptability']==0]
    df2 = df2[df2['C3 Acceptability']==0]
    df = pd.concat([df1, df2], axis=0)
    df.drop_duplicates(subset='ID', keep='first', inplace=True)
    print(df)
    for ID in df['ID']:
        print(ID)
        ID = ID.split('_')[1]
        data_path = proj_dir + '/inference/crop_resize_img/' + ID + '.nrrd'
        save_path = save_dir + '/' + ID + '.nrrd'
        shutil.copyfile(data_path, save_path)


if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)

    inter_rater(dict1["proj_dir"], dict1["reviewer_1_file"], dict1["reviewer_2_file"] )
    #check_scans(dict1["proj_dir"], dict1["save_checked_scans_dir"], dict1["reviwer_1_file"], dict1["reviewer_2_file"])







