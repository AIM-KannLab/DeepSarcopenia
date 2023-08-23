import numpy as np
import os
import pandas as pd
import glob
from PIL import Image, ImageOps
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2


def draw_contour():
    """get segmentation and image fusion in JPG format
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/test/visualize'
    if not os.path.exists(proj_dir + '/C3_contour_png'):
        os.makedirs(proj_dir + '/C3_contour_png')
    img_paths = [i for i in sorted(glob.glob(proj_dir + '/img/*nrrd'))]
    seg_paths = [i for i in sorted(glob.glob(proj_dir + '/seg/*nrrd'))]
    count = 0
    IDs = []
    for img_path, seg_path in zip(img_paths, seg_paths):
        ID = img_path.split('/')[-1].split('.')[0]
        count += 1
        print(count, ID)
        IDs.append(ID)
        nrrd_img = sitk.ReadImage(img_path)
        arr_img = sitk.GetArrayFromImage(nrrd_img)
        nrrd_seg = sitk.ReadImage(seg_path)
        arr_seg = sitk.GetArrayFromImage(nrrd_seg)
        # generate contour with CV2
        img = np.uint8(arr_img*255)
        seg = np.uint8(arr_seg*255)
        contour, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.drawContours(
            image=main, 
            contours=contour, 
            contourIdx=-1, 
            color=(0, 0, 255), 
            thickness=1,
            lineType=16)
        cv2.imwrite(proj_dir + '/C3_contour_png/' + ID + '.png', main) 
    df = pd.DataFrame({'ID': IDs})
    df.to_csv(proj_dir + '/patient_list.csv', index=False)


def save_img_slice():
    proj_dir = '/mnt/kannlab_rfa/Zezhong'
    csv_path = proj_dir + '/c3_segmentation/output/NonOPC_C3_top_slice_pred.csv'
    img_dir = proj_dir + '/c3_segmentation/test/crop_img'
    seg_dir = proj_dir + '/c3_segmentation/test/pred'
    save_img_dir = proj_dir + '/c3_segmentation/test/visualize/img'
    save_seg_dir = proj_dir + '/c3_segmentation/test/visualize/seg'
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)
    df = pd.read_csv(csv_path)
    count = 0
    for ID, Slice in zip(df['patient_id'], df['C3_Predict_slice']):
        count += 1
        print(count, ID)
        for data_dir, save_dir in zip([img_dir, seg_dir], [save_img_dir, save_seg_dir]):
            data_path = data_dir + '/' + ID + '.nrrd'
            nrrd = sitk.ReadImage(data_path)
            arr = sitk.GetArrayFromImage(nrrd)
            arr_slice = arr[Slice, :, :]
            save_path = save_dir + '/' + ID + '.nrrd'
            img_sitk = sitk.GetImageFromArray(arr_slice)
            img_sitk.SetSpacing(nrrd.GetSpacing())
            img_sitk.SetOrigin(nrrd.GetOrigin())
            writer = sitk.ImageFileWriter()
            writer.SetFileName(save_path)
            writer.SetUseCompression(True)
            writer.Execute(img_sitk)

def main0():
    """get segmentation and image fusion in JPG format
    """

    proj_dir = '/mnt/kannlab_rfa/Zezhong/c3_segmentation/visualize'
    if not os.path.exists(proj_dir + '/merge_jpg'):
        os.makedirs(proj_dir + '/merge_jpg')
    if not os.path.exists(proj_dir + '/img_jpg'):
        os.makedirs(proj_dir + '/img_jpg')
    if not os.path.exists(proj_dir + '/seg_jpg'):
        os.makedirs(proj_dir + '/seg_jpg')
    if not os.path.exists(proj_dir + '/fusion_jpg'):
        os.makedirs(proj_dir + '/fusion_jpg')
    if not os.path.exists(proj_dir + '/contour_jpg'):
        os.makedirs(proj_dir + '/contour_jpg')

    img_paths = [i for i in sorted(glob.glob(proj_dir + '/img/*nrrd'))]
    seg_paths = [i for i in sorted(glob.glob(proj_dir + '/seg/*nrrd'))]
    count = 0
    IDs = []
    for img_path, seg_path in zip(img_paths, seg_paths):
        ID = img_path.split('/')[-1].split('.')[0]
        count += 1
        print(count, ID)
        IDs.append(ID)
        nrrd_img = sitk.ReadImage(img_path)
        arr_img = sitk.GetArrayFromImage(nrrd_img)
        nrrd_seg = sitk.ReadImage(seg_path)
        arr_seg = sitk.GetArrayFromImage(nrrd_seg)
        #print(arr_img[250])
        
        # generate contour with CV2
        img = np.uint8(arr_img*255)
        _seg = np.uint8(arr_seg*255)
        #idx = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1][0]
        #points = np.transpose(idx)
        #out = np.zeros_like(seg)
        #out[tuple(points)] = 255
        #out[idx[:, 0, 0], idx[:, 0, 1]] = 255
        contours, hierarchy = cv2.findContours(_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        tmp = np.zeros_like(_seg)
#        boundary = cv2.drawContours(tmp, contours, -1, (255, 255, 255), 1)
#        boundary[boundary > 0] = 255
#        #plt.imsave(proj_dir + '/contour_jpg/' + ID + '.jpg', boundary, cmap='gray')
#
#        # np array to jpg with PIL
#        img = Image.fromarray(np.uint8(arr_img*255), 'L')
#        seg = Image.fromarray(np.uint8(arr_seg*255), 'L')
#        contour = Image.fromarray(np.uint8(boundary), 'L')
#        img = img.convert('RGBA')
#        contour = contour.convert('RGBA')
#        
#        # change contour color
#        d = contour.getdata()
#        new_img = []
#        for item in d:
#            # change all white (also shades of whites) pixels to yellow
#            if item[0] in list(range(200, 256)):
#                new_img.append((255, 0, 0, 1))
#            else:
#                #new_img.append()
#                new_img.append((0, 0, 0, 0))
#        # update image data
#        contour.putdata(new_img)
#
#        #seg = ImageOps.colorize(seg, black='black', white='red') 
#        
#        # img + seg fusion
#        #merge = Image.blend(img, seg, 0.5)
        main = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.drawContours(
            image=main, 
            contours=contours, 
            contourIdx=-1, 
            color=(0, 0, 255), 
            thickness=1,
            lineType=16)
        #fusion = Image.blend(contour, img, 0.9)
        cv2.imwrite(proj_dir + '/fusion_jpg/' + ID + '.png', main) 
        #fusion.save(proj_dir + '/fusion_jpg/' + ID + '.png')
        #contour.save(proj_dir + '/contour_jpg/' + ID + '.png')
        #img.save(proj_dir + '/img_jpg/' + ID + '.jpg')
        #seg.save(proj_dir + '/seg_jpg/' + ID + '.jpg')
#       #plt.imsave(proj_dir + '/merge/' + ID + '.jpg', arr_img, cmap='gray')
#       #cv2.imwrite(proj_dir + '/merge/' + ID + '.jpg', arr_img)

    df = pd.DataFrame({'ID': IDs})
    df.to_csv(proj_dir + '/patient_list.csv', index=False)


if __name__ == '__main__':

    main()
    #save_img_slice()




