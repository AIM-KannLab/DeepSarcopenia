import SimpleITK as sitk
import numpy as np


def write_sitk_from_array_by_template(array, template_sitk, sitk_output_path):
    
    output_spacing = template_sitk.GetSpacing()
    output_direction = template_sitk.GetDirection()
    output_origin = template_sitk.GetOrigin()

    sitk_output = sitk.GetImageFromArray(array)
    sitk_output.SetSpacing(output_spacing)
    sitk_output.SetDirection(output_direction)
    sitk_output.SetOrigin(output_origin)

    nrrdWriter = sitk.ImageFileWriter()
    nrrdWriter.SetFileName(sitk_output_path)
    nrrdWriter.SetUseCompression(True)
    nrrdWriter.Execute(sitk_output)
    #nrrdWriter.Execute(sitk_output)
    
    print()
