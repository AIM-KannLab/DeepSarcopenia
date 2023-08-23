import argparse
from scripts.segmentation import train
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description  =  'Train a u-net model on multiple classes')
    parser.add_argument('--data_dir', '-d',type = str, default = '../data/train/train_segmentation/', 
                        help  =  'Directory in which the segmentation training data arrays are stored')
    parser.add_argument('--model_dir', '-m',type = str, default = '../model/train/unet_models/', 
                        help = 'Location where trained models are to be stored')
    
    parser.add_argument('--epochs', '-e', type  =  int, default  =  600, help = 'number of training epochs')
    parser.add_argument('--batch_size','-b', type  =  int, default  =  1, help = 'batch size')
    parser.add_argument('--load_weights','-w', help  =  'load weights in this file to initialise model')
    parser.add_argument('--name','-a', help  =  'trained model will be stored in a directory with this name')
    parser.add_argument('--gpus','-g', type  =  int, default  =  1, help  =  'number of gpus')
    parser.add_argument('--learning_rate','-l', type = float, default = 0.004, help = 'learning rate')
    parser.add_argument('--upsamlping_modules','-D', type = int,  default = 5,
                        help = 'downsampling/upsamlping module numbers')
    parser.add_argument('--initial_features','-F', type = int, default = 16,
                        help = 'number of feautres in first model')
    parser.add_argument('--activation','-A', default = 'relu', help = 'activation function to use')
    parser.add_argument('--num_convs','-N', type = int, default = 2, help = 'activation function to use')
    
    args = parser.parse_args()
    
    model = train(**vars(args))



    
    

