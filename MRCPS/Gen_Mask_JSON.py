# -*- coding: utf-8 -*-

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] =  '2033120000'
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat

import argparse
from time import perf_counter
from functools import wraps
import pyvips
import json
from tqdm import tqdm

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

def numpy2vips(a):
    height, width = a.shape
    linear = a.reshape(width * height * 1)  #train mask only 0~1 not *255
    vi = pyvips.Image.new_from_memory(linear.data, width, height, 1,
                                      dtype_to_format[str(a.dtype)])
    # height, width = a.shape
    # linear = a.reshape(width * height * 1)*255
    # vi = pyvips.Image.new_from_array(a*255)
    return vi


def maskGenOrderMethod(mask, ann_info, disease_label, is_roi):
    
    label_profiles = disease_label['label_profile']
    annotations = ann_info['annotation']
    total_list = []

    for annotation in tqdm(annotations):
        #roi check
        if annotation['name'] == 'roi' and not is_roi:
            continue
        elif is_roi and annotation['name'] != 'roi':
            continue

        for label_type in label_profiles:
            if label_type['name'] == annotation['name']:
                ann_value = label_type['value']

        coordinates = np.array(annotation['coordinates'], dtype=np.int32)

        # if len(coordinates) != 0:
        #     print(f't: {len(coordinates)}')
        points = []
        mask_temp = np.zeros(mask.shape)
        for region in coordinates:
            x = float(region[0])
            y = float(region[1])
            points.append([x, y])
        if len(points):
            pts = np.asarray([points], dtype=np.int32)
            cv2.fillPoly(mask_temp, pts, color=int(ann_value))
            total_list.append([ann_value, pts, np.sum(mask_temp)])
    total_list.sort(key=lambda l:l[2], reverse=True)
 
    if len(total_list)==0:
        return False, mask
    print(mask.max())
    #check category
    for i in range(len(total_list)):
        cv2.fillPoly(mask, total_list[i][1], color= total_list[i][0])  
    return True, mask


def genMask(json_path, slide_path, mask_path, roi_path,uuid, disease_label):

    tifpth = os.path.join(slide_path, f"{uuid}.tif")
    annpth = os.path.join(json_path, f"{uuid}.json")
    save_path = os.path.join(mask_path, f"{uuid}.tif")
    save_roi_path = os.path.join(roi_path, f"{uuid}.tif")

    if not os.path.exists(tifpth):
        print(f"tifpth not match: {uuid}")
        return
    if not os.path.exists(annpth):
        print(f"xmlpth not match: {uuid}")
        return
    
    slide = pyvips.Image.new_from_file(tifpth)

    # 讀取 JSON 文件
    with open(annpth, 'r') as jsonfile:
        ann_info = json.load(jsonfile)

    print(f'Saving {uuid} : {save_path}')

    # tumor mask
    mask = np.zeros((slide.height, slide.width), dtype=np.uint8)
    print('start process')
    check_flag, mask = maskGenOrderMethod(mask, ann_info, disease_label, is_roi=False)
    if check_flag:
        vips_img = numpy2vips(mask)
        vips_img.tiffsave(save_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)

    # roi mask
    mask_roi = np.zeros((slide.height, slide.width), dtype=np.uint8)
    check_roi_flag, mask_roi = maskGenOrderMethod(mask, ann_info, disease_label, is_roi=True)
    if check_roi_flag:
        vips_img_roi = numpy2vips(mask_roi)
        vips_img_roi.tiffsave(save_roi_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="./dataset/images/")
    parser.add_argument('--json_path', default="./dataset/annotations/")
    parser.add_argument('--save_path', default="./dataset/")
    args = parser.parse_args()

    slide_path = args.slide_path
    json_path = args.json_path
    mask_path = os.path.join(args.save_path, 'mask/')
    roi_path = os.path.join(args.save_path, 'rois/')

    # slide_path = './dataset/images/'
    # json_path = './dataset/annotations/'
    # mask_path = './dataset/masks/'
    # roi_path = './dataset/rois/'
    disease_label = {
                        "label_profile": [
                            {"name": "tumor", "value": 1},
                            {"name": "normal", "value": 0}
                        ]
                    }

    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(roi_path, exist_ok=True)
    
    files = []
    for file in os.listdir(slide_path):
        if file in os.listdir(mask_path):
            continue
        # if '-' in file and file != '-f':
        #     files.append(file)
        file_type = file[-len(file.split('.')[-1]):]
        if file_type in ['tif','tiff','ndpi','svs','mrxs']:
            files.append(file[:-4])
    print(len(files))
    print(len(os.listdir(slide_path)))

    for file in files:
        filename = file[:-len(file.split('.')[-1])]
        new_output_path = os.path.join(mask_path, f"{filename}.tif")
        genMask(json_path, slide_path, mask_path, roi_path, file, disease_label)


