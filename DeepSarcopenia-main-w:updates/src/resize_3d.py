import SimpleITK as sitk


def resize_3d(img_nrrd, interp_type, output_size):

#    image = sitk.ReadImage(nrrd_image)
    image = img_nrrd
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (
        (input_size[0] * input_spacing[0]) / output_size[0],
        (input_size[1] * input_spacing[1]) / output_size[1],
        (input_size[2] * input_spacing[2]) / output_size[2]
        )
    #print('{} {}'.format('input spacing: ', input_spacing))
    #print('{} {}'.format('output spacing: ', output_spacing))

    ### choose interpolation algorithm
    if interp_type == 'linear':
        interp_type = sitk.sitkLinear
    elif interp_type == 'bspline':
        interp_type = sitk.sitkBSpline
    elif interp_type == 'nearest_neighbor':
        interp_type = sitk.sitkNearestNeighbor

    ### interpolate
    resample = sitk.ResampleImageFilter()
    resample.SetSize(output_size)
    resample.SetOutputSpacing(output_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interp_type)
    img_nrrd = resample.Execute(image)

    return img_nrrd
