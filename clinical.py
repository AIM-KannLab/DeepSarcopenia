import pandas as pd
import numpy as np
from opts import parse_opts, get_args



def clinical(seg_csa_path, seg_meta_path, clinical_file_path):
    """
    Get cervical CSA, Lumbar CSA and lumbar SMI
    Args:
        proj_dir -- proj dir
    Returns:
        dataframe, csv
    """
    
    
    
    # ---get C3 segmention data---
    df_csa = pd.read_csv(seg_csa_path)
   
    # ---get clinical information---
    df0 = pd.read_csv(seg_meta_path, encoding='unicode_escape')
    df = df0.merge(df_csa, on='patient_id', how='left').reset_index()
    # exclude surgery and induction cases
    df = df[~df['Pre-RT Neck Dissection'].isin(['Yes'])]
    df = df[~df['Pre-RT Primary Resection'].isin(['Yes'])]
    df = df[~df['Pre-RT Surgery'].isin(['Yes'])]
    df = df[~df['Radiation adjuvant to surgery'].isin(['Yes'])]
    #df = df[~df['Induction Chemotherapy'].isin(['Yes'])]

    #df['Weight'] = df['Pre-treatment Weight in Kilograms'].to_list()
    df['Weight'] = df['Pre-treatment Weight in Pounds'].values * 0.454
    df['BMI'] = df['Pre-treatment BMI'].to_list()
    df = df[df['Weight'].notna()]
    df = df[df['BMI'].notna()]
    df['Height'] = np.sqrt(df['Weight'] / df['BMI'])
    print('case number:', df.shape[0])
    ages = []
    for x, y in zip(df['Radiation Therapy Start Date'], df['Date of Birth']):
        #if np.isnan(x):
        if pd.isnull(x) or pd.isnull(y):
            age = 0
        else:
            x = int(str(x).split('/')[-1])
            y = int(str(y).split('/')[-1])
            if x > 80:
                # this would be 1980s or 1990s
                age = x - y
            else:
                # this would be 2000s, 2010s or 2020s
                age = x - y + 100
        ages.append(age)
    df['Age'] = ages
    df = df[df['Age'].notna()]
    df = df.merge(df0, on='PMRN', how='left').reset_index()
    df['CSA'] = df['seg_area'].to_list()
    df['CSD'] = df['seg_csd'].to_list()
    df = df[df['CSA'].notna()]
    print('case number:', df.shape[0])
    #df.to_csv(proj_dir + '/clinical/C3_test.csv', index=False)

    # ---get C3 CSA and C3 SMI---
    C3_SMIs = []
    C3_CSAs = []
    bad_data = []
    errors = []
    for i in range(df.shape[0]):
        ID = df['PMRN'].iloc[i]
        gender = df['Gender'].iloc[i]
        CSA = df['CSA'].values[i]
        age = df['Age'].values[i]
        weight = df['Weight'].values[i]
        height = df['Height'].values[i]
        if gender == 'Female':
            sex = 1
        elif gender == 'Male':
            sex = 2
        else:
            print('input wrong gender info!')
        # CSA (cm2), age (years), weight (kg)
        L3_CSA = 27.304 + CSA*1.363 - age*0.671 + weight*0.640 + sex*26.422
        L3_SMI = L3_CSA/(height**2)
        C3_CSAs.append(L3_CSA)
        C3_SMIs.append(L3_SMI)
    df['C3_CSA'] = C3_CSAs
    df['C3_SMI'] = C3_SMIs
    df.to_csv(clinical_file_path, index=False)


def exclusion():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    df0 = pd.read_csv(proj_dir + '/clinical/HN_C3_TOT.csv', encoding='unicode_escape')
    df1 = pd.read_csv(proj_dir + '/visualize/patient_list_review_sum.csv', encoding='unicode_escape')
    excludes = []
    pmrns = []
    for ID, comment in zip(df1['ID'], df1['exclusion']):
        pmrn = int(ID.split('_')[1])
        pmrns.append(pmrn)
        if comment in ['postop', 'incomplete scan', 'artifact', 'skin fold']:
            exclude = 'yes'
        else:
            exclude = 'no'
        excludes.append(exclude)
    exclusions = df1['exclusion'].to_list()
    df2 = pd.DataFrame({'PMRN': pmrns, 'Exclude': excludes, 'Comments': exclusions})
    print(df2)
    print(excludes)
    print(exclusions)
    df = df0.merge(df2, how='left', on='PMRN')
    df.to_csv(clinical_file_path, index=False)

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    opt = parse_opts()
    dict1 = get_args(opt)
    clinical(dict1["seg_csa_path"],dict1["seg_meta_path"],dict1["clinical_file_path"] )
    #exclusion()





