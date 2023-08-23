import argparse
import os
import warnings
from scripts.infer_segmentation import test
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")
   
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bodycomposition is segmented and stored in NIFTI format.') 
    
    parser.add_argument('--image_dir','-i', type = str, default = '../data/test/input', 
                        help = 'File path of CT scan')
    
    parser.add_argument('--model_weight_path','-m', type = str, default = '../model/test/C3_Top_Segmentation_Model_Weight.hdf5',
                        help = 'File path of well-trained model weight')
    
    parser.add_argument('--l3_slice_csv_path','-c', type = str, default = '../data/test/output_csv/C3_Top_Slice_Prediction.csv',
                        help = 'C3 top slice number of the input CT scan')
    
    parser.add_argument('--output_dir','-o', type = str, 
                        default = '../data/test/output_segmentation/',
                        help = 'File path of well-trained model weight')    
    
    args = parser.parse_args()
    model = test(**vars(args))
    
