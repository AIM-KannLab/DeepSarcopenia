import os
import SimpleITK as sitk
import numpy as np
from src.unet import get_unet_2D
from src.image_processing.image_window import get_image_path_by_id, remove_arm_area
from src.image_processing.get_sitk_from_array import write_sitk_from_array_by_template
import matplotlib.pyplot as plt
import pandas as pd


def test_segmentation(image_dir, model_weight_path, l3_slice_csv_path, output_dir):
    
    model = get_unet_2D(2, (512, 512, 1), num_convs=2, activation='relu',
            compression_channels=[16, 32, 64, 128, 256, 512],
            decompression_channels=[256, 128, 64, 32, 16]   )
    model.load_weights(model_weight_path)
    
    df_prediction_l3 = pd.read_csv(l3_slice_csv_path, index_col=0)
    ids = []
    for idx in range(df_prediction_l3.shape[0]):    
        patient_id = str(df_prediction_l3.iloc[idx, 0])
        print('pat id:', patient_id)
        #infer_3d_path = output_dir + patient_id + '.seg.nrrd'
        infer_3d_path = output_dir + '/' + patient_id + '.nrrd'
        print(infer_3d_path)
        image_path = get_image_path_by_id(patient_id, image_dir)
        try:
            image_sitk = sitk.ReadImage(image_path)
            #image_sitk = resample_sitk_image(image_sitk1, new_spacing=[1, 1, 3],new_size= None, interpolator=sitk.sitkLinear)
            image_array_3d = sitk.GetArrayFromImage(image_sitk)
            im_xy_size = image_array_3d.shape[1]
            l3_slice_auto = int(df_prediction_l3.iloc[idx, 1])
            #print("c3 slice")
            print('slice #:', l3_slice_auto)
            image_array = sitk.GetArrayFromImage(image_sitk)[l3_slice_auto, :, :].reshape(1, 512, 512, 1) 
            image_array_2d = sitk.GetArrayFromImage(image_sitk)[l3_slice_auto, :, :]
            target_area = remove_arm_area(image_array_2d)
            infer_seg_array = model.predict(image_array)
            softmax_threshold = 0.5
            muscle_seg = (infer_seg_array[:,:,:,1] >= softmax_threshold) * 1.0 * target_area
           # sfat_seg = (infer_seg_array[:,:,:,2] >= softmax_threshold) * 2.0 * target_area
           # vfat_seg = (infer_seg_array[:,:,:,3] >= softmax_threshold) * 3.0 * target_area
            sfat_seg = 0
            vfat_seg = 0 
           
            infer_seg_array_2d = muscle_seg+sfat_seg+vfat_seg
            infer_seg_array_3d = np.zeros(image_array_3d.shape)
            infer_seg_array_3d[l3_slice_auto,:,:] = infer_seg_array_2d
        
            print(image_array_2d.shape)
            print(infer_seg_array_2d.shape)
            fig, ax = plt.subplots(1, 2, figsize=(24, 8))
            plt.subplots_adjust(wspace=0.05)
            ax[0].imshow(image_array_2d)
    #         ax[0].set_title('L3_slice Predicted: '+str(l3_slice_auto), fontsize=18)
            ax[0].set_title('C3_slice auto selected: '+str(l3_slice_auto), fontsize=18)
            infer_seg_array_2d_1 = infer_seg_array_2d[0]
           
            ax[1].imshow(infer_seg_array_2d_1)
            ax[1].set_title('Model Seg',fontsize=18)
                  
            write_sitk_from_array_by_template(infer_seg_array_3d, image_sitk, infer_3d_path )

            print(idx,'th image:',patient_id,'(C3_slice_auto:',l3_slice_auto,')  segmentation_in_NIFTI saved into')
            #print(infer_3d_path)
            #print()
        except Exception as e:
            print(patient_id, e)
            ids.append(patient_id)
    print(ids)


# SOURCE: https://github.com/iantsen/hecktor/blob/main/src/data/utils.py
def resample_sitk_image(sitk_image,
                        new_spacing=[1, 1, 1],
                        new_size=None,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        fill_value=0):
    """
    Resample a SimpleITK Image.
    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    new_spacing : list of int
        A distance between adjacent voxels in each dimension given in physical units (mm) for the output image.
    new_size : list of int or None
        A number of pixels per dimension of the output image. If None, `new_size` is computed based on the original
        input size, original spacing and new spacing.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.
    Returns
    -------
    sitk.Image
        The resampled image.
    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # provided attributes:
    if attributes:
        orig_pixelid = attributes['orig_pixelid']
        orig_origin = attributes['orig_origin']
        orig_direction = attributes['orig_direction']
        orig_spacing = attributes['orig_spacing']
        orig_size = attributes['orig_size']

    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=int)

    # new image size:
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetDefaultPixelValue(orig_pixelid)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetDefaultPixelValue(fill_value)

    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image

