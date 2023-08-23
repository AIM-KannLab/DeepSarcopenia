from scipy.ndimage.filters import gaussian_filter
from scripts.densenet_regression import DenseNet
from scripts.image_processing.image_window import apply_window
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np 
import pandas as pd
import tensorflow as tf
import glob, os, functools

def get_slice_number_from_prediction(predictions):
    
    predictions = 2.0 * (predictions - 0.5)  #sigmoid.output = True
    smoothing_kernel = 2.0
    smoothed_predictions = gaussian_filter(predictions, smoothing_kernel)

    zero_crossings = []
    for s in range(len(smoothed_predictions) - 1):
        if (smoothed_predictions[s] < 0.0) != (smoothed_predictions[s + 1] < 0.0):
            if(abs(smoothed_predictions[s]) < abs(smoothed_predictions[s + 1])):
                zero_crossings.append(s)
            else:
                zero_crossings.append(s + 1)
    if len(zero_crossings) == 0:
        chosen_index = np.argmin(abs(smoothed_predictions))    
    else: 
        chosen_index = sorted(zero_crossings[:1])[0]
    return chosen_index


def test(image_dir, model_weight_path,csv_write_path):
    
    with tf.device('/gpu:0'):
        model = DenseNet(img_dim=(256, 256, 1), 
                nb_layers_per_block=12, nb_dense_block=4, growth_rate=12, nb_initial_filters=16, 
                compression_rate=0.5, sigmoid_output_activation=True, 
                activation_type='relu', initializer='glorot_uniform', output_dimension=1, batch_norm=True )
    model.load_weights(model_weight_path)
    print('\n','\n','\n','loaded:' ,model_weight_path)  
    
    images = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    df_prediction = pd.DataFrame()
    
    for idx, image_path in enumerate(images):
        image_sitk = sitk.ReadImage(image_path)    
        image_array  = sitk.GetArrayFromImage(image_sitk)
        windowed_images = apply_window(image_array)
        
        resize_func = functools.partial(resize, output_shape=model.input_shape[1:3],
                                        preserve_range=True, anti_aliasing=True, mode='constant')
        series = np.dstack([resize_func(im) for im in windowed_images])
        series = np.transpose(series[:, :, :, np.newaxis], [2, 0, 1, 3])
        predictions = model.predict(series)
        chosen_index = get_slice_number_from_prediction(predictions)
        
        df_inter = pd.DataFrame({'patient_id':image_path.split('/')[-1],
                                'L3_Predict_slice':chosen_index,
                                'Z_spacing':round(image_sitk.GetSpacing()[-1],5),
                                 'XY_spacing': round(image_sitk.GetSpacing()[0],5)},index=[0])
        df_prediction = df_prediction.append(df_inter)
        df_prediction.to_csv(csv_write_path)
           
        print(idx,' th image, path: ',image_path,'\n','L3_Predict_slice:',chosen_index,'\n')
    print('L3 slice prediction is written into:',csv_write_path,'\n')
