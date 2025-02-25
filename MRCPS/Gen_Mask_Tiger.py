# -*- coding: utf-8 -*-
'''
Author : Taki
Date : 2021/08/22
Description:
    Genrate mask from XML.
    讓ASAP能夠讀取 讚爆!    
'''

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] =  '2033120000'
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat

from time import perf_counter
from functools import wraps
import pyvips

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


def normal(mask, xml_root):
    normal_list=[
        'tumor-associated stroma',
        'healthy glands',
        'necrosis not in-situ',
        'inflamed stroma',
        'rest'
    ]

    for gp in normal_list:
        coordinates = xml_root.findall(
            f".//Annotation[@PartOfGroup='{gp}']/Coordinates")

        if coordinates != []:
            print(f'n: {len(coordinates)}')

        for regions in coordinates:
            points = []
            for region in regions:
                x = float(region.attrib['X'])
                y = float(region.attrib['Y'])
                points.append([x, y])
            if len(points):
                pts = np.asarray([points], dtype=np.int32)
                cv2.fillPoly(img=mask, pts=pts, color=0)


    return mask


def tumor(mask, xml_root):
    tumor_list=[
        'invasive tumor',
        'in-situ tumor'
    ]

    for gp in tumor_list:
        coordinates = xml_root.findall(
            f".//Annotation[@PartOfGroup='{gp}']/Coordinates")
        
        if coordinates != []:
            print(f't: {len(coordinates)}')
        
        for regions in coordinates:
            points = []
            for region in regions:
                x = float(region.attrib['X'])
                y = float(region.attrib['Y'])
                points.append([x, y])
            if len(points):
                pts = np.asarray([points], dtype=np.int32)
                cv2.fillPoly(mask, pts, color=1)

    return mask


def maskGenOrderMethod(mask, xml_root):
    type_list=[
        'invasive tumor',       # tumor
        'in-situ tumor',        # tumor
        'tumor-associated stroma',
        'healthy glands',
        'necrosis not in-situ',
        'inflamed stroma',
        'rest'
    ]

    print('function work')
    total_list = []
    for tl_i in range(len(type_list)):
        coordinates = xml_root.findall(
            f".//Annotation[@PartOfGroup='{type_list[tl_i]}']/Coordinates")
        
        if coordinates != []:
            print(f't: {len(coordinates)}')
        
        for regions in coordinates:
            mask_temp = np.zeros(mask.shape)
            points = []
            for region in regions:
                if ',' in region.attrib['X']:
                    x = float(region.attrib['X'].split(',')[0])
                    y = float(region.attrib['Y'].split(',')[0])
                else:
                    x = float(region.attrib['X'])
                    y = float(region.attrib['Y'])
                points.append([x, y])
            if len(points):
                pts = np.asarray([points], dtype=np.int32)
                cv2.fillPoly(mask_temp, pts, color=1)
                total_list.append([tl_i, pts, np.sum(mask_temp)])
    
    total_list.sort(key=lambda l:l[2], reverse=True)

    #check category
    for i in range(len(total_list)):
        print(total_list[i][0], total_list[i][2])

        if total_list[i][0] in [0,1]:
            cv2.fillPoly(mask, total_list[i][1], color=1)
        else:
            cv2.fillPoly(mask, total_list[i][1], color=0)    


        #check overlap problem
        # if total_list[i][0] in [0,1]:
        #     check_interLabel = 0
        #     for pt in total_list[i][1][0]: #pts shape= [1,2]
        #         check_interLabel+=mask[pt[1],pt[0]]
        #     if check_interLabel/len(total_list[i][1]) > 0:
        #         cv2.fillPoly(mask, total_list[i][1], color=0)
        #     else:
        #         cv2.fillPoly(mask, total_list[i][1], color=1)
        # else:
        #     cv2.fillPoly(mask, total_list[i][1], color=0)    

# def result(mask, xml_root):
#     coordinates = xml_root.findall(
#         ".//Annotation[@PartOfGroup='Result']/Coordinates")
#     if coordinates != []:
#         print(f'r: {len(coordinates)}')

#     # for regions in coordinates:
#     #     points = []
#     #     for region in regions:
#     #         x = float(region.attrib['X'])
#     #         y = float(region.attrib['Y'])
#     #         points.append([x, y])
#     #     if len(points):
#     #         pts = np.asarray([points], dtype=np.int32)
#     #         cv2.fillPoly(mask, pts, color=1)

    return mask


def roi(mask, xml_root):
    coordinates = xml_root.findall(
        ".//Annotation[@PartOfGroup='roi']/Coordinates")

    for regions in coordinates:
        points = []
        for region in regions:
            if ',' in region.attrib['X']:
                x = float(region.attrib['X'].split(',')[0])
                y = float(region.attrib['Y'].split(',')[0])
            else:
                x = float(region.attrib['X'])
                y = float(region.attrib['Y'])
            points.append([x, y])   
        if len(points):
            pts = np.asarray([points], dtype=np.int32)
            cv2.fillPoly(mask, pts, color=1)
    print(np.sum(mask))
    return mask


def genMask(anno_path,anno_bcss_path, tifs_path, mask_path, roi_path, uuid, nonlabel_record):
    tifpth = os.path.join(tifs_path, f"{uuid}.tif")
    if 'TCGA' in uuid:
        xmlpth = os.path.join(anno_bcss_path, f"{uuid}.xml")
    else:
        xmlpth = os.path.join(anno_path, f"{uuid}.xml")
    save_path = os.path.join(mask_path, f"{uuid}.tif")
    save_roi_path = os.path.join(roi_path, f"{uuid}.tif")

    if not os.path.exists(tifpth):
        print(f"tifpth not match: {uuid}")
        return
    if not os.path.exists(xmlpth):
        print(f"xmlpth not match: {uuid}")
        nonlabel_record.append(uuid)
        return
    
    slide = pyvips.Image.new_from_file(os.path.join(tifs_path, f"{uuid}.tif"))

    xml_root = ET.parse(xmlpth)
    print(f'Saving {uuid}')
    print(save_path)


    # mask = tumor(mask, xml_root)
    # mask = normal(mask, xml_root)
    # roi mask
    mask = np.zeros((slide.height, slide.width), dtype=np.uint8)
    print('start process')
    mask = maskGenOrderMethod(mask, xml_root)
    vips_img = numpy2vips(mask)
    print(save_path)
    vips_img.tiffsave(save_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)

    # roi mask
    mask_roi = np.zeros((slide.height, slide.width), dtype=np.uint8)
    mask_roi = roi(mask_roi, xml_root)
    vips_img_roi = numpy2vips(mask_roi)
    vips_img_roi.tiffsave(save_roi_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)

    return nonlabel_record
#try to load multi mask tif to make tumor mask (but tif load problem result failed)
# def genMask_tif(anno_path,anno_bcss_path, tifs_path, mask_path, roi_path, uuid):
#     tifpth = os.path.join(tifs_path, f"{uuid}.tif")
#     if 'TCGA' in uuid:
#         maskpth = os.path.join(anno_bcss_path, f"{uuid}.tif")
#     else:
#         maskpth = os.path.join(anno_path, f"{uuid}.tif")
#     save_path = os.path.join(mask_path, f"{uuid}.tif")
#     save_roi_path = os.path.join(roi_path, f"{uuid}.tif")

#     if not os.path.exists(tifpth):
#         print(f"tifpth not match: {uuid}")
#         return
#     if not os.path.exists(maskpth):
#         print(f"xmlpth not match: {uuid}")
#         return
    

#     '''紀錄: 嘗試 使用 現有的tiff mask 以編號分成tumor與normal兩類 但遇到tiff載入不成功問題'''
#     '''error message 
#         pyvips.error.Error: unable to fetch from region
#         TIFFFillTile: 0: Invalid tile byte count, tile 0
#     '''

#     ''''''
#     # Load the TIFF image 
#     print(maskpth)
#     # maskpth = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/images/TCGA-GM-A3XL-01Z-00-DX1.CCE8AA1D-9194-4E49-9546-DBF25A35847C.tif'
#     # maskpth = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/masks/TCGA-EW-A1OV-01Z-00-DX1.93698123-5B34-4163-848B-2D75A5F7B001.tif'
#     maskpth = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-bcss-masks/TCGA-A1-A0SK-01Z-00-DX1.A44D70FA-4D96-43F4-9DD7-A61535786297.tif'
#     mask_slide = pyvips.Image.openslideload(maskpth, level=0, access='sequential')
#     # print(mask_slide.size)
#     # import os
#     # import cv2
#     # cv2.MAX_IMAGE_PIXELS = 2033120000
#     # mask_slide = cv2.imread(maskpth)
#     # print(mask_slide.height)
#     # print(mask_slide.shape)
#     # mask_slide = pyvips.Image.thumbnail(maskpth, 1024)
#     # print(mask_slide.bands)
#     # Convert the image to a NumPy array
#     # numpy_array = mask_slide[0].numpy()
#     # numpy_array = np.array(mask_slide)
#     # print(numpy_array.shape)

#     numpy_array = np.ndarray(buffer=mask_slide[0].write_to_memory(),
#                             dtype=np.uint8,
#                             shape=[mask_slide.height, mask_slide.width, mask_slide.bands])
#     # data = [[i*512,j*512] for i in range(int(mask_slide.height/512)) for j in range(int(mask_slide.width/512))]
#     # img_region = pyvips.Region.new(mask_slide)
#     # patchdata = img_region.fetch(0, 0, 512, 512)
#     # patchdata = img_region.fetch(data[1][0], data[1][1], 512, 512)
#     # img = np.ndarray(buffer=mask_slide, dtype=np.uint8, shape=[
#     #                  512, 512, mask_slide.bands])

#     print(img.shape)
#     numpy_array=0
#     ''''''


#     #tumor (1,3)
#     tumor_mask = np.zeros((mask_slide.height, mask_slide.width), dtype=np.uint8)
#     tumor_list = [1,3]
#     for t in tumor_list:
#         tumor_mask += numpy_array[:,:,t]
#     tumor_mask[tumor_mask>1] = 1

#     #non tumor (2,4,5,6,7)
#     nontumor_mask = np.zeros((mask_slide.height, mask_slide.width), dtype=np.uint8)
#     nontumor_list = [2,4,5,6,7]
#     for t in nontumor_list:
#         nontumor_mask += numpy_array[:,:,t]
#     tumor_mask[nontumor_mask>1] = 0


#     vips_img = numpy2vips(tumor_mask)
#     vips_img.tiffsave(save_path, tile=True, compression='deflate', bigtiff=True, pyramid=True)


if __name__ == '__main__':

    # root = '/work/u7085556/ColorectalDataset'
    # tif_path = os.path.join(root, 'tifs')
    # anno_path = os.path.join(root, 'annotations')
    # mask_path = os.path.join(root, 'masks')

    # tif_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/images'
    # # anno_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls'
    # # anno_bcss_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls'
    # anno_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/refine_xml'
    # anno_bcss_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/refine_xml'
    # # anno_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-masks'
    # # anno_bcss_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-bcss-masks'
    # mask_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/masks_refine'
    # roi_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/masks_rois'



    tif_path = r'D:\dataset\tiger\wsirois\wsi-level-annotations\images'
    # anno_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls'
    # anno_bcss_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls'
    anno_path = r'D:\dataset\tiger\wsirois\wsi-level-annotations\images'
    anno_bcss_path = r'D:\dataset\tiger\wsirois\wsi-level-annotations\images'
    # anno_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-masks'
    # anno_bcss_path = '/work/u2676425/dataset/tiger_dataset/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-bcss-masks'
    mask_path = r'D:\dataset\tiger\wsirois\wsi-level-annotations\masks_refine'
    roi_path = r'D:\dataset\tiger\wsirois\wsi-level-annotations\masks_rois'



    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(roi_path, exist_ok=True)

    files = []
    for file in os.listdir(tif_path):
        if file in os.listdir(mask_path):
            continue
        # if '-' in file and file != '-f':
        #     files.append(file)
        file_type = file[-len(file.split('.')[-1]):]
        if file_type in ['tif','tiff','ndpi','svs','mrxs']:
            files.append(file[:-4])
    print(len(files))
    print(len(os.listdir(tif_path)))

    nonlabel_record=[]
    nonlabel_records=[]
    for file in files:
        nonlabel_record = genMask(anno_path, anno_bcss_path, tif_path, mask_path, roi_path,file, nonlabel_record)
        # genMask_tif(anno_path, anno_bcss_path, tif_path, mask_path, roi_path,file)
    # with ProcessPoolExecutor(max_workers=1) as e:
    #     nonlabel_records = e.map(genMask,
    #                         repeat(anno_path), 
    #                         repeat(anno_bcss_path), 
    #                         repeat(tif_path),
    #                         repeat(mask_path),
    #                         repeat(roi_path),
    #                         files, repeat(nonlabel_record))
    print(nonlabel_record)
