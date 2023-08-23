import datetime
import json
import os
import numpy as np
import warnings
import matplotlib as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from scripts.data_generator import train_generator

from scripts.unet import get_unet_2D
from scripts.losses import dice_coef_multiclass_loss_2D
from scripts.callbacks import MultiGPUModelCheckpoint
from scripts.generators import SegmentationSequence



def train(data_dir, model_dir, name=None, epochs=100, batch_size=1, load_weights=None,
          gpus=1, learning_rate=0.0005,  num_convs=2,activation='relu', 
          upsamlping_modules=5, initial_features=16):

    args = locals()   
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    warnings.filterwarnings("ignore")
    
    compression_channels = list(2**np.arange(int(np.log2(initial_features)),
                                             1+upsamlping_modules+int(np.log2(initial_features))))
    decompression_channels=sorted(compression_channels,reverse=True)[1:]

    train_images_file = os.path.join(data_dir, 'train_images.npy')
    val_images_file = os.path.join(data_dir, 'val_images.npy')
    train_masks_file = os.path.join(data_dir, 'train_masks.npy')
    val_masks_file = os.path.join(data_dir, 'val_masks.npy')

    images_train = np.load(train_images_file)
    images_train = images_train.astype(float)
    
    images_val = np.load(val_images_file)
    images_val = images_val.astype(float)
    
    
    masks_train = np.load(train_masks_file)
    masks_train = masks_train.astype(np.uint8)
    
    
    masks_val = np.load(val_masks_file)
    masks_val = masks_val.astype(np.uint8)
    
    
    print('\n\n\nimages_train.shape,images_val.shape', images_train.shape,images_val.shape,'\n\n\n')
    
    # Directories and files to use
    if name is None:
        name = 'untitled_model_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join(model_dir, name)
    tflow_dir = os.path.join(output_dir, 'tensorboard_log')
    os.mkdir(output_dir)
    os.mkdir(tflow_dir)
    weights_path = os.path.join(output_dir, 'weights-{epoch:02d}-{val_loss:.4f}.hdf5')
    architecture_path = os.path.join(output_dir, 'architecture.json')
    tensorboard = TensorBoard(log_dir=tflow_dir, histogram_freq=0, write_graph=False, write_images=False)

    args_path = os.path.join(output_dir, 'args.json')
    with open(args_path, 'w') as json_file:
        json.dump(args, json_file, indent=4)

    print('Creating and compiling model...')
    with tf.device('/gpu:0'):
        model = get_unet_2D(
            4,
            (512, 512, 1),
            num_convs=num_convs,
            activation=activation,
            compression_channels=compression_channels,
            decompression_channels=decompression_channels
            )

    # Save the architecture
    with open(architecture_path,'w') as json_file:
        json_file.write(model.to_json())

    # Use multiple devices
    #if gpus > 1:
    #    parallel_model = multi_gpu_model(model, gpus)
    #else:
    parallel_model = model

    # Should we load existing weights?
    if load_weights is not None:
        print('Loading pre-trained weights...')
        parallel_model.load_weights(load_weights)

    val_batches = images_val.shape[0] // batch_size 
    print('\n \n  val_batches::::',val_batches, '\n')
    train_batches = images_train.shape[0] // batch_size 
    print('\n \n  train_batches::::',train_batches, '\n')

    # Set up the learning rate scheduler
    def lr_func(e):
        print("                                       Learning Rate Update at Epoch", e)
        if e > 0.75 * epochs:
            print("                                 Lr ",0.2* learning_rate)
            return 0.01 * learning_rate
        elif e > 0.5 * epochs:
            print("                             Lr ",0.5* learning_rate)
            return 0.1 * learning_rate
        else:
            print(                           "Lr",1 * learning_rate)
            return learning_rate
        
   
    lr_scheduler = LearningRateScheduler(lr_func)
    
    train_generator = SegmentationSequence(images_train, masks_train, batch_size, jitter=True)
    val_generator = SegmentationSequence(images_val, masks_val, batch_size, jitter=True)

    parallel_model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_multiclass_loss_2D)
    
    print('Fitting model...')

    keras_model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=False)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001,
        mode='min', patience=10, verbose=1,
        restore_best_weights=True)
    
    #datagen = ImageDataGenerator()
    #it = datagen.flow(train_generator)
    
    #history = parallel_model.fit_generator(it, train_batches, steps_per_epoch=313, epochs=epochs, shuffle=True, #validation_steps=val_batches, validation_data=val_generator, use_multiprocessing=False, workers=1, max_queue_size=40, callbacks=#[keras_model_checkpoint, tensorboard, lr_scheduler, early_stopping])  
    
    history = parallel_model.fit_generator(train_generator, train_batches, epochs=epochs,
                      shuffle=True, validation_steps=val_batches, validation_data=val_generator, use_multiprocessing=False,
                      workers=1,max_queue_size=40, callbacks=[keras_model_checkpoint, tensorboard, lr_scheduler])     

    # Save the template model weights
#     model.save_weights(os.path.join(output_dir, 'final_model.hdf5'))
#     run.finish()


    return model
