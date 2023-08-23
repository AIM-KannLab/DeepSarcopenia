import datetime
import json
import os
import numpy as np
import pandas as pd
import matplotlib as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.generators import SliceSelectionSequence
from src.densenet_regression import DenseNet


def train(proj_dir, epochs=100, name=None, batch_size=128,
          load_weights=None, gpus=1, learning_rate=0.001, threshold=10.0,
          nb_layers_per_block=4, nb_blocks=4, nb_initial_filters=16,
          growth_rate=12, compression_rate=0.5, activation='relu',
          initializer='glorot_uniform', batch_norm=True, wandb_callback=True):
    
    # check tf version and GPU
    print('\ntf version:', tf. __version__)
    tf.test.gpu_device_name()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.config.list_physical_devices('GPU')
    print('\nNum GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    args = locals()
    tr_img_dir = proj_dir + '/data/train_data/raw_data/train_npy'
    vl_img_dir = proj_dir + '/data/train_data/raw_data/val_npy'
    tr_df_path = proj_dir + '/data/slice_selection/train.csv'
    vl_df_path = proj_dir + '/data/slice_selection/val.csv'
    tr_labels = pd.read_csv(tr_df_path)['ZOffset'].values 
    vl_labels = pd.read_csv(vl_df_path)['ZOffset'].values
    print('train shape:',tr_labels.shape,'val shape:', vl_labels.shape)
    
    train_jitter = 1201  # default 1000 times of image augmentation for each epoch
    val_jitter = 387  # default 50 times of image augmentation for each epoch
    #train_jitter = 10
    #val_jitter = 10
    train_generator = SliceSelectionSequence(
        labels=tr_labels, image_dir=tr_img_dir, batch_size=batch_size, 
        batches_per_epoch=train_jitter, jitter=True, sigmoid_scale=threshold)
    val_generator = SliceSelectionSequence(
        labels=vl_labels, image_dir=vl_img_dir, batch_size=batch_size, 
        batches_per_epoch=val_jitter, sigmoid_scale=threshold)

    # Directories and files to use
    if name is None:
        name = 'DenseNet_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = proj_dir + '/model/slice_selection/' + name
    tflow_dir = output_dir + '/tensorboard_log'
    weights_path = output_dir + '/weights-{epoch:02d}-{val_loss:.4f}.hdf5'
    architecture_path = output_dir + '/architecture.json'
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=0, write_graph=False, write_images=False)
    
    model = DenseNet(
        img_dim=(256, 256, 1),
        nb_layers_per_block=nb_layers_per_block,
        nb_dense_block=nb_blocks,
        growth_rate=growth_rate,
        nb_initial_filters=nb_initial_filters,
        compression_rate=compression_rate,
        sigmoid_output_activation=True,
        activation_type=activation,
        initializer=initializer,
        output_dimension=1,
        batch_norm=batch_norm)
    if load_weights is None:
        os.makedirs(output_dir)
        os.makedirs(tflow_dir)

        args_path = os.path.join(output_dir, 'args.json')
        with open(args_path, 'w') as json_file:
            json.dump(args, json_file, indent=4)

        # Create the model
        print('Compiling model')
    # Save the architecture
        with open(architecture_path, 'w') as json_file:
            json_file.write(model.to_json())

    else:
        # Load the weights
        model.load_weights(load_weights)

    # Move to multi GPUs
    parallel_model = model
    keras_model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)

    # Set up the learning rate scheduler
    def lr_func(e):
        print("Learning Rate Update at Epoch", e)
        if e > 0.75 * epochs:
            return 0.01 * learning_rate
        elif e > 0.5 * epochs:
            return 0.1 * learning_rate
        else:
            return learning_rate

    lr_scheduler = LearningRateScheduler(lr_func)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10) 
    #model_callbacks = [keras_model_checkpoint, tensorboard, lr_scheduler, es]
    model_callbacks = [keras_model_checkpoint, tensorboard, lr_scheduler]

    # Compile multi-gpu model
    loss = 'mean_absolute_error'
    parallel_model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    print('Starting training...')
    history = parallel_model.fit(train_generator, epochs=epochs,
                                 shuffle=False, validation_data=val_generator,
                                 callbacks=model_callbacks,
                                 use_multiprocessing=False,
                                 verbose=1)
  
    # training curves
    loss_train = history.history['acc']
    loss_val = history.history['val_acc']
    epochs = range(1, 101)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(output_dir + '/train_curves.jpg')
   
    return model


if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation'
    train(proj_dir)







