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

def numpy2vips(a):
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
    height, width = a.shape
    linear = a.reshape(width * height * 1)  #train mask only 0~1 not *255
    vi = pyvips.Image.new_from_memory(linear.data, width, height, 1,
                                      dtype_to_format[str(a.dtype)])
    # height, width = a.shape
    # linear = a.reshape(width * height * 1)*255
    # vi = pyvips.Image.new_from_array(a*255)
    return vi


# def maskGenOrderMethod(mask, ann_info, disease_label, is_roi):
    
#     label_profiles = disease_label['label_profile']
#     annotations = ann_info['annotation']
#     total_list = []

#     for annotation in tqdm(annotations):
#         #roi check
#         if annotation['name'] == 'roi' and not is_roi:
#             continue
#         elif is_roi and annotation['name'] != 'roi':
#             continue

#         for label_type in label_profiles:
#             if label_type['name'] == annotation['name']:
#                 ann_value = label_type['value']

#         coordinates = np.array(annotation['coordinates'], dtype=np.int32)

#         # if len(coordinates) != 0:
#         #     print(f't: {len(coordinates)}')
#         points = []
#         mask_temp = np.zeros(mask.shape)
#         for region in coordinates:
#             x = float(region[0])
#             y = float(region[1])
#             points.append([x, y])
#         if len(points):
#             pts = np.asarray([points], dtype=np.int32)
#             cv2.fillPoly(mask_temp, pts, color=int(ann_value))
#             total_list.append([ann_value, pts, np.sum(mask_temp)])
#     total_list.sort(key=lambda l:l[2], reverse=True)
 
#     if len(total_list)==0:
#         return False, mask
#     print(mask.max())
#     #check category
#     for i in range(len(total_list)):
#         cv2.fillPoly(mask, total_list[i][1], color= total_list[i][0])  
#     return True, mask

def maskGenOrderMethod(mask, ann_info, disease_label, is_roi):
    
    label_profiles = disease_label['label_profile']
    annotations = ann_info['annotation']
    total_list = []

    for annotation in tqdm(annotations):
        #roi check
        if annotation['name'].lower() == 'roi' and not is_roi:
            continue
        elif is_roi and annotation['name'].lower() != 'roi':
            continue

        
        ann_value = 0
        check_label_exist = False
        for label_type in label_profiles:
            if label_type['name'].lower() == annotation['name'].lower():
                ann_value = label_type['value']
                check_label_exist = True
        if not check_label_exist:
            print(f"check label type:{annotation['name'].lower()}")

        # alovas新舊格式判斷
        if type(annotation['coordinates'][0]) is dict and 'x' in annotation['coordinates'][0].keys():
            coordinates = np.array([[pt['x'], pt['y']] for pt in annotation['coordinates']], dtype=np.int32)
        else:
            coordinates = np.array(annotation['coordinates'], dtype=np.int32)

        if len(coordinates) == 0:
            continue
            
        # 用 cv2.contourArea 直接計算多邊形面積，無需創建 mask_temp
        if annotation['type']=="rectangle":
            area = (coordinates[1][0]-coordinates[0][0])*(coordinates[1][1]-coordinates[0][1])
        else:
            area = cv2.contourArea(coordinates.reshape(-1, 2))  # 計算實際幾何面積
        total_list.append({
            'value': ann_value,
            'coordinates': coordinates,
            'area': area,
            'type': annotation['type']
        })
    
    # 按面積降序排序
    total_list.sort(key=lambda x: x['area'], reverse=True)

    if not total_list:
        return False, mask

    # 直接在主 mask 上繪製
    for item in total_list:
        if item['type']=="rectangle":
            cv2.rectangle(mask, item['coordinates'][0], item['coordinates'][1], item['value'], -1)
        else:
            cv2.fillPoly(mask, [item['coordinates']], color=item['value'])

    return True, mask


def genMask(json_path, slide_path, mask_path, roi_path, uuid, wsi_type, disease_label):

    tifpth = os.path.join(slide_path, f"{uuid}{wsi_type}")
    annpth = os.path.join(json_path, f"{uuid}json")
    save_path = os.path.join(mask_path, f"{uuid}tif")
    save_roi_path = os.path.join(roi_path, f"{uuid}tif")

    print(tifpth)
    if not os.path.exists(tifpth):
        print(f"tifpth not match: {uuid}")
        return
    if not os.path.exists(annpth):
        print(f"xmlpth not match: {uuid}")
        return
    
    slide = pyvips.Image.new_from_file(tifpth)
    slide_height, slide_width = slide.height, slide.width

    # 讀取 JSON 文件
    with open(annpth, 'r') as jsonfile:
        ann_info = json.load(jsonfile)

    print(f'Saving {uuid} : {save_path}')

    # tumor mask
    mask = np.zeros((slide_height, slide_width), dtype=np.uint8)
    print('start process')
    check_flag, mask = maskGenOrderMethod(mask, ann_info, disease_label, is_roi=False)
    if check_flag:
        vips_img = numpy2vips(mask)
        vips_img.tiffsave(save_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)

    # roi mask
    mask_roi = np.zeros((slide_height, slide_width), dtype=np.uint8)
    check_roi_flag, mask_roi = maskGenOrderMethod(mask_roi, ann_info, disease_label, is_roi=True)
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
    mask_path = os.path.join(args.save_path, 'masks/')
    roi_path = os.path.join(args.save_path, 'rois/')

    # slide_path = './dataset/images/'
    # json_path = './dataset/annotations/'
    # mask_path = './dataset/masks/'
    # roi_path = './dataset/rois/'
    # disease_label = {
    #                     "label_profile": [
    #                         {"name": "tumor", "value": 1},
    #                         {"name": "normal", "value": 0}
    #                     ]
    #                 }
    disease_label = {
                        "label_profile": [
                            {"name": "ROI", "value": 1},
                            {"name": "Background", "value": 0},
                            {"name": "void", "value": 0},
                            {"name": "Invasive", "value": 1},
                            {"name": "In_situ", "value": 1},
                            {"name": "tumor", "value": 1},
                            {"name": "normal", "value": 0},
                            {"name": "Benign", "value": 0},
                            {"name": "necrosis", "value": 0},
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
        filetype = file[-len(file.split('.')[-1]):]
        if filetype in ['tif','tiff','ndpi','svs','mrxs']:
            files.append([file[:-len(file.split('.')[-1])], filetype])
            
    print(len(files))
    print(len(os.listdir(slide_path)))

    for filename, filetype in files:
        # filename = file[:-len(file.split('.')[-1])]
        # filetype = file[-len(file.split('.')[-1]):]
        print(filename, filetype)
        new_output_path = os.path.join(mask_path, f"{filename}.tif")
        genMask(json_path, slide_path, mask_path, roi_path, filename, filetype, disease_label)


