import argparse
import os
import warnings
from scripts.slice_selection import train

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
warnings.filterwarnings("ignore")     

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train a densenet regression model to predict the selection offset')
    parser.add_argument('--data_dir', '-d',type = str, default = '../data/train/train_selection/',
                        help = 'Location of the training data directory')
    parser.add_argument('--model_dir', '-m',type = str, default = '../model/', 
                        help = 'Location where trained models are to be stored')
    
    parser.add_argument('--epochs', '-e', type = int, default = 60, help = 'number of training epochs')
    parser.add_argument('--name', '-a', help = 'weights will be stored with this name')
    parser.add_argument('--batch_size', '-b', type = int, default = 16, help = 'batch size')
    parser.add_argument('--load_weights', '-w', help = 'load these weights and continue training')

    parser.add_argument('--gpus', '-g', type = int, default = 1, help = 'number of gpus')
    parser.add_argument('--learning_rate', '-l', type = float, default = 0.0001, help = 'learning rate')
    parser.add_argument('--threshold', '-t', type = float, default = 10.0,
                        help = 'soft-threshold the distance with a sigmoid function with this scale parameter')
    parser.add_argument('--nb_layers_per_block', default = 12, type = int,
                        help = "number of layers per block in densenet")
    parser.add_argument('--nb_blocks', default = 4, type = int, help = "number of layers of blocks in densenet")
    parser.add_argument('--nb_initial_filters', default = 16, type = int,
                        help = "number of initial filters in densenet")
    parser.add_argument('--growth_rate', default = 12, type = int, help = "densenet growth rate (k) parameter")
    parser.add_argument('--compression_rate', default = 0.5, type = float, help = "densenet compression rate parameter")
    parser.add_argument('--initializer', '-I', default = 'glorot_uniform', help = "initializer for weights in the network")
    parser.add_argument('--activation', '-A', default = 'relu', help = "activation for units in the network")
    parser.add_argument('--omit_batch_norm', '-B', action = 'store_false', dest = 'batch_norm',
                        help = "omit batch normalization")

    args = parser.parse_args()

    model = train(**vars(args))
