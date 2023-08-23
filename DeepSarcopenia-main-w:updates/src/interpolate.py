import SimpleITK as sitk
import sys
import os


def interpolate(patient_id, img, interpolation_type, new_spacing, return_type, output_dir):
    
    """
    Interpolates a given nrrd file to a given voxel spacing.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
        path_to_nrrd (str): Path to nrrd file.
        interpolation_type (str): Either 'linear' (for images with continuous values), 'bspline' (also for images but will mess up the range of the values), or 'nearest_neighbor' (for masks with discrete values).
        new_spacing (tuple): Tuple containing 3 values for voxel spacing to interpolate to: (x,y,z).
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type').
    Raises:
        Exception if an error occurs.
    """

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    #print('{} {}'.format('original size: ', original_size))
    #print('{} {}'.format('original spacing: ', original_spacing))
    new_size = [int(round((original_size[0]*original_spacing[0])/float(new_spacing[0]))),
                int(round((original_size[1]*original_spacing[1])/float(new_spacing[1]))),
                int(round((original_size[2]*original_spacing[2])/float(new_spacing[2])))]
    #print('{} {}'.format('new size: ', new_size))

    if interpolation_type == 'linear':
        interpolation_type = sitk.sitkLinear
    elif interpolation_type == 'bspline':
        interpolation_type = sitk.sitkBSpline
    elif interpolation_type == 'nearest_neighbor':
        interpolation_type = sitk.sitkNearestNeighbor

    # interpolate
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interpolation_type)
    resample.SetDefaultPixelValue(img.GetPixelIDValue())
    resample.SetOutputPixelType(sitk.sitkFloat32)
    image = resample.Execute(img)

    if output_dir != '':
        save_fn = output_dir + '/' + patient_id + '.nrrd'
        writer = sitk.ImageFileWriter()
        writer.SetFileName(save_fn)
        writer.SetUseCompression(True)
        writer.Execute(image)

    if return_type == 'sitk_obj':
        return image
    elif return_type == 'np_array':
        return sitk.GetArrayFromImage(image)





