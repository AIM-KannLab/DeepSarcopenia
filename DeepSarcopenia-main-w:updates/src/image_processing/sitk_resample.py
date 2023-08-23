import SimpleITK as sitk
import numpy as np


def get_resampled_sitk(data_sitk,target_spacing):
    new_spacing = target_spacing

    orig_spacing = data_sitk.GetSpacing()
    orig_size = data_sitk.GetSize()

    new_size = [int(orig_size[0] * orig_spacing[0] / new_spacing[0]),
              int(orig_size[1] * orig_spacing[1] / new_spacing[1]),
              int(orig_size[2] * orig_spacing[2] / new_spacing[2])]

    res_filter = sitk.ResampleImageFilter()
    img_sitk = res_filter.Execute(data_sitk,
                                new_size,
                                sitk.Transform(),
                                sitk.sitkLinear,
                                data_sitk.GetOrigin(),
                                new_spacing,
                                data_sitk.GetDirection(),
                                0,
                                data_sitk.GetPixelIDValue())

    return img_sitk

    

def get_cropped_sitk(data_sitk, target_size = [512,512]):
  # 1) Crop down bigger axes

    image_array = sitk.GetArrayFromImage(data_sitk)
    air = int(np.min(image_array))
    
    oldSize = data_sitk.GetSize()
    
    newSizeDown = [max(0, int((oldSize[0] - target_size[0]) / 2)),
                 max(0, int((oldSize[1] - target_size[1]) / 2)),
                 0]
    newSizeUp = [max(0, oldSize[0] - target_size[0] - newSizeDown[0]),
               max(0, oldSize[1] - target_size[1] - newSizeDown[1]),
               0]

    cropFilter = sitk.CropImageFilter()
    cropFilter.SetUpperBoundaryCropSize(newSizeUp)
    cropFilter.SetLowerBoundaryCropSize(newSizeDown)
    data_sitk = cropFilter.Execute(data_sitk)

    # 2) Pad smaller axes
    oldSize = data_sitk.GetSize()

    newSizeDown = [max(0, int((target_size[0] - oldSize[0]) / 2)),
                 max(0, int((target_size[0] - oldSize[1]) / 2)),
                 0]
    newSizeUp = [max(0, target_size[0] - oldSize[0] - newSizeDown[0]),
               max(0, target_size[1] - oldSize[1] - newSizeDown[1]),
               0]

    padFilter = sitk.ConstantPadImageFilter()
    padFilter.SetConstant(air)
    padFilter.SetPadUpperBound(newSizeUp)
    padFilter.SetPadLowerBound(newSizeDown)
    data_sitk = padFilter.Execute(data_sitk)

    return data_sitk


def save_sitk_into_path(sitk_object,target_path):
    nrrdWriter = sitk.ImageFileWriter()
    nrrdWriter.SetFileName(target_path)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(sitk_object)
    
