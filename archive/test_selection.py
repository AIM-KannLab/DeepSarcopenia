import argparse
import warnings
import os 
from scripts.infer_selection import test

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")   

if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description=' C3 top slice is selected by AI algorithm.') 
    
    parser.add_argument('--image_dir','-i', type = str, default = '../data/test/input/', 
                        help = 'Directory path of abdominal CT scan')
    parser.add_argument('--model_weight_path','-m', type = str, default = '../model/test/C3_Top_Selection_Model_Weight.h5',
                        help = 'File path of well-trained model weight')
    parser.add_argument('--csv_write_path','-c', type = str, default = '../data/test/output_csv/C3_Top_Slice_Prediction.csv',
                        help = 'File path of well-trained model weight')
    
    args = parser.parse_args()
    model = test(**vars(args))