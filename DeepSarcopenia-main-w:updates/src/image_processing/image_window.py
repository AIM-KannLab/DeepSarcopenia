import os
from skimage.measure import label
from skimage import morphology
from skimage.morphology import square
import numpy as np


# window level of 40 and a window width of 400 (-160,240) ,  rescaling to 0 to 255.
def apply_window(image, win_centre= 40, win_width= 400):
    range_bottom = win_centre - win_width / 2
    scale = 256 / win_width
    image = image - range_bottom

    image = image * scale
    image[image < 0] = 0
    image[image > 255] = 255

    return image


def get_image_path_by_id(patient_id, image_dir):
    image_order = patient_id
    file_name_list = [os.path.relpath(os.path.join(image_dir, x)) \
                    for x in os.listdir(image_dir) \
                    if os.path.isfile(os.path.join(image_dir, x)) and patient_id in x]
    #file_path_list = []
    #for x in os.listdir(image_dir):
    #    print(x)
    #    if os.path.isfile(os.path.join(image_dir, x)) and patient_id in x:
    #        path = os.path.relpath(os.path.join(image_dir, x))
    #        file_name_list.append(path)

    if len(file_name_list)>0:
        return file_name_list[0]
    else:
        return ''


def remove_arm_area(ct_array_2d):
    bw_img = ct_array_2d>-250
    labeled_img , num = label(bw_img, connectivity=2, background=0, return_num=True)
    max_label = 0
    max_num = 0
    
    for i in range(1, num+1):
        if np.sum(labeled_img==i) > max_num:
            max_num = np.sum(labeled_img ==i)
            max_label = i
    biggest_area = (labeled_img==max_label)
    
    biggest_area_closed = morphology.remove_small_holes(biggest_area, area_threshold=10000, connectivity=2)
    biggest_area_closed_dilated = morphology.dilation(biggest_area_closed, square(8))
    return biggest_area_closed_dilated









