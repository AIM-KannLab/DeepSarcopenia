
import numpy as np
import itertools
import SimpleITK as sitk



def reduce_arr_dtype(arr, verbose=False):
    """ Change arr.dtype to a more memory-efficient dtype, without changing
    any element in arr. """

    if np.all(arr-np.asarray(arr,'uint8') == 0):
        if arr.dtype != 'uint8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint8 np.ndarray')
            arr = np.asarray(arr, dtype='uint8')
    elif np.all(arr-np.asarray(arr,'int8') == 0):
        if arr.dtype != 'int8':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int8 np.ndarray')
            arr = np.asarray(arr, dtype='int8')
    elif np.all(arr-np.asarray(arr,'uint16') == 0):
        if arr.dtype != 'uint16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to uint16 np.ndarray')
            arr = np.asarray(arr, dtype='uint16')
    elif np.all(arr-np.asarray(arr,'int16') == 0):
        if arr.dtype != 'int16':
            if verbose:
                print('Converting '+str(arr.dtype)+' to int16 np.ndarray')
            arr = np.asarray(arr, dtype='int16')

    return arr

def combine_masks(mask_list):
    if len(mask_list) >= 2:
        for a, b in itertools.combinations(mask_list, 2):
            assert a.shape == b.shape, "masks do not have the same shape"
            assert a.max() == b.max(), "masks do not have the same max value (1)"
            assert a.min() == b.min(), "masks do not have the same min value (0)"

        """
        we will ignore the fact that 2 masks at the same voxel will overlap and cause that vixel to have a value of 2. The get_bbox function doesnt really care about that - it just evaluates zero vs non-zero
        """
        combined = np.zeros((mask_list[0].shape))
        for mask in mask_list:
            if mask is not None:
                combined = combined + mask
        return combined
    elif len(mask_list) == 1:
        return mask_list[0]
    else:
        print ("No masks provided!")

def get_bbox(mask_data):
    """
    Returns min, max, length, and centers across Z, Y, and X. (12 values)
    """
    # crop maskData to only the 1's
    # http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # maskData order is z, y, x because we already rolled it
    #print(mask_data.shape)
    Z = np.any(mask_data, axis=(1, 2))
    Y = np.any(mask_data, axis=(0, 2))
    X = np.any(mask_data, axis=(0, 1))
    #print(Z)
    #print(X)
    #print(Y)
    Z_min, Z_max = np.where(Z)[0][[0, -1]]
    Y_min, Y_max = np.where(Y)[0][[0, -1]]
    X_min, X_max = np.where(X)[0][[0, -1]]
    # 1 is added to account for the final slice also including the mask
    return Z_min, Z_max, Y_min, Y_max, X_min, X_max, Z_max-Z_min+1, Y_max-Y_min+1, X_max-X_min+1, (Z_max-Z_min)/2 + Z_min, (Y_max-Y_min)/2 + Y_min, (X_max-X_min)/2 + X_min

def bbox2_3D(img):
    """
    Returns bounding box fit to the boundaries of non-zeros
    """
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    rmin=int(rmin)
    rmax=int(rmax)
    cmin=int(cmin)
    cmax=int(cmax)
    zmin=int(zmin)
    zmax=int(zmax)
    return rmin, rmax, cmin, cmax, zmin, zmax
    
def threshold(pred, thresh=0.5):
    pred[pred<thresh] = 0
    pred[pred>=thresh] = 1
    return pred

def get_spacing(sitk_obj):
    """
    flip spacing from sitk (x,y,z) to numpy (z,y,x)
    """
    spacing = sitk_obj.GetSpacing()
    return (spacing[2], spacing[1], spacing[0])

def get_arr_from_nrrd(link_to_nrrd, type):
    '''
    Used for images or labels.
    '''
    sitk_obj = sitk.ReadImage(link_to_nrrd)
    spacing = get_spacing(sitk_obj)
    origin = sitk_obj.GetOrigin()
    arr = sitk.GetArrayFromImage(sitk_obj)
    if type=="label":
        arr = threshold(arr)
        assert arr.min() == 0, "minimum value is not 0"
        #assert arr.max() == 1, "minimum value is not 1"
        #assert len(np.unique(arr)) == 2, "arr does not contain 2 unique values"
    return sitk_obj, arr, spacing, origin


def generate_sitk_obj_from_npy_array(image_sitk_obj, pred_arr, resize=True, output_dir=""):

    """
    When resize==True: Used for saving predictions where padding needs to be added to increase the size of the prediction and match that of input to model. This function matches the size of the array in image_sitk_obj with the size of pred_arr, and saves it. This is done equally on all sides as the input to model and model output have different dims to allow for shift data augmentation.
    When resize==False: the image_sitk_obj is only used as a reference for spacing and origin. The numpy array is not resized.
    image_sitk_obj: sitk object of input to model
    pred_arr: returned prediction from model - should be squeezed.
    NOTE: image_arr.shape will always be equal or larger than pred_arr.shape, but never smaller given that
    we are always cropping in data.py
    """
    if resize==True:
        # get array from sitk object
        image_arr = sitk.GetArrayFromImage(image_sitk_obj)
        # change pred_arr.shape to match image_arr.shape
        # getting amount of padding needed on each side
        z_diff = int((image_arr.shape[0] - pred_arr.shape[0]) / 2)
        y_diff = int((image_arr.shape[1] - pred_arr.shape[1]) / 2)
        x_diff = int((image_arr.shape[2] - pred_arr.shape[2]) / 2)
        # pad, defaults to 0
        pred_arr = np.pad(pred_arr, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff)), 'constant')
        assert image_arr.shape == pred_arr.shape, "oops.. The shape of the returned array does not match your requested shape."

    # save sitk obj
    new_sitk_object = sitk.GetImageFromArray(pred_arr)
    new_sitk_object.SetSpacing(image_sitk_obj.GetSpacing())
    new_sitk_object.SetOrigin(image_sitk_obj.GetOrigin())

    if output_dir != "":
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_dir)
        writer.SetUseCompression(True)
        writer.Execute(new_sitk_object)
    return new_sitk_object

def write_sitk(sitk_obj, path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.SetUseCompression(True)
    writer.Execute(sitk_obj)


