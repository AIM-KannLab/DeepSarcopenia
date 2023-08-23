import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    # Key arguments to be passed to the code
    parser.add_argument('--proj_dir', default='/Users/yashravipati/Desktop/AIM/DeepSarcopenia-main', type=str, help='project path')
    parser.add_argument('--dataset', default='RADCURE', type=str, help='Instituion the dataset belongs to')
    parser.add_argument('--STEP', default='ALL', type=str, help='Process Steps: options SLICE/PREPROCESS/SEGMENT/ALL')    
    parser.add_argument('--slice_model', default='/model/test/C3_Top_Selection_Model_Weight.hdf5', type=str, help='Slice Selection Model Directory path')
    parser.add_argument('--seg_model', default='/model/test/C3_Top_Segmentation_Model_Weight.hdf5', type=str, help='Segmentation Model Directory path')
    
    # Use this argument if you are generating the clinical file with sarcopenia metrics merging with a meta file.
    parser.add_argument('--meta_file', default='HN_clinical_meta_data.csv', type=str, help='label')
    
    # Use these arguments if we want to explicitly change the directories of input files.
    parser.add_argument('--img_dir', default='/data/test/img/', type=str, help='raw images path')
    parser.add_argument('--manual_seg_dir', default='/data/test/segs/', type=str, help='Segmentation Model Directory path')

    # data preprocessing
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
          
    # OPTIONAL:  utilities arguments 
    parser.add_argument('--reviewer_1_file', default='reviewer1.csv', type=str, help='Reviewer 1 ratings')
    parser.add_argument('--reviewer_2_file', default='reviewer2.csv', type=str, help='Reviewer 2 ratings')
    parser.add_argument('--save_checked_scans_dir', default='save_checked_scans_dir', type=str, help='Directory path for saving the checed scans')
    parser.add_argument('--test_csv_path', default='test.csv', type=str, help='CSV containing the id, preidcted slide and manual slice for dice score generation')
    parser.add_argument('--dice_csv', default='dice_csv.csv', type=str, help='CSV storing final dice values')
          
    # others 
    parser.add_argument('--load_data', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(load_data=True)
    parser.add_argument('--load_model', action='store_true', help='If true, load model is performed.')
    parser.set_defaults(load_model=True)
        
    args = parser.parse_args()

    return args


def get_args(opt):
    
    proj_dir = opt.proj_dir
    dataset = opt.dataset
   
    img_dir = proj_dir + '/data/'+dataset+'/img/'
    #img_dir = opt.img_dir
    
    slice_model = proj_dir + opt.slice_model
    

    slice_csv = dataset + '_C3_top_slice_pred.csv'
    
    slice_csv_path = proj_dir + '/data/'+ dataset + '/' + slice_csv 
    
    pre_process_dir = proj_dir + '/data/' + dataset + '/preprocessed/'
    #pre_process_dir= '/Volumes/T7/radcure_nrrd/preprocessed/'
    
    seg_model = proj_dir + opt.seg_model
        
    out_dir = proj_dir + '/data/'+dataset+ '/output_segmentations'
    #out_dir = '/Volumes/T7/radcure_nrrd/segs/'
    
    seg_csa = dataset + '_C3_CSA.csv'
    
    seg_csa_path = proj_dir + '/data/'+dataset+'/' + seg_csa

    if (dataset != "test") :
        seg_meta = dataset + '_meta.csv'
        seg_meta_path = proj_dir + '/data/'+dataset+'/' + seg_meta

    clinical_file = dataset + '_meta.csv'
    
    clinical_file_path = proj_dir + '/data/'+dataset+'/' + clinical_file
    visual_save_dir = proj_dir + '/data/'+dataset+'/visual_contours/' 
    visual_csv_file = proj_dir+'/data/'+dataset+'/visual_patient_list.csv'    
    
    print("------ Print arguments -----")
    print("Project root directory :"+ proj_dir)
    print("Raw Images directory :"+ img_dir)
    print("Preprocess Image directory :"+ pre_process_dir)
    print("Output Segmentation Directory :"+ out_dir)
    print("Slice Selection Model :"+ slice_model)
    print("Segmentation Model :"+ seg_model)
    
    print("Slice Prediction File :"+ slice_csv_path)
    print("Segmentation CSA File :"+ seg_csa_path)
    print("Clinical META File :"+ seg_meta_path)
    print("Consolidated Clinical File :"+ clinical_file_path)
    
    dict1 = {"proj_dir":proj_dir, "img_dir":img_dir}
    dict1["pre_process_dir"]=pre_process_dir
    dict1["out_dir"]=out_dir
    dict1["slice_model"]=slice_model
    dict1["seg_model"]=seg_model
    dict1["slice_csv_path"]=slice_csv_path
    dict1["seg_csa_path"]=seg_csa_path
    dict1["seg_meta_path"]=seg_meta_path
    dict1["clinical_file_path"]=clinical_file_path
    dict1["visual_save_dir"]=visual_save_dir
    dict1["visual_csv_file"]=visual_csv_file
    dict1["reviewer_1_file"]=opt.reviewer_1_file
    dict1["reviewer_2_file"]=opt.reviewer_2_file
    dict1["save_checked_scans_dir"]=opt.save_checked_scans_dir
    dict1["manual_seg_dir"]=opt.manual_seg_dir
   

    return dict1
