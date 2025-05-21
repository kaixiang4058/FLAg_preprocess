# coding: utf-8

import json
import os
from tqdm import tqdm
import pyvips
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import cv2
import pandas as pd
import argparse
markset = set()

def read_WSI(wsi_path, data_type, level):
    slice = None
    if data_type == 'tif':
        slice = pyvips.Image.tiffload(wsi_path, page=level)
    elif data_type in ['svs','ndpi','mrxs']:
        slice = pyvips.Image.new_from_file(wsi_path, level=level)
    else:
        print('type not include')
    return slice


def filetering(data, img_region, roi_region, mask_region, img_slide_band, mask_slide_band, patch_size, processtype):
    '''
    judge patch class
    process flow consider processtype:
        label:   in_roi(if roi and mask), all(if mask) 
        unlabel: out_roi(if roi and mask), all(if no mask)
    return: workflag, class, roi_check(err:true), mark_check(err:true)
    '''

    #check in roi range
    mark_check = False

    ##fetch roi, img, mask patch from tif and check work
    if roi_region is not None:
        patchroi = roi_region.fetch(data[1][0], data[1][1], patch_size, patch_size)
        roiarea = np.ndarray(buffer=patchroi, dtype=np.uint8,
                             shape=[patch_size, patch_size, mask_slide_band])
        if np.average(roiarea) <1:  #out roi -> label skip
            if processtype=='label':        #--------------------------------
                return False, None, True, None
        else:                       #in roi -> unlabel skip
            if processtype=='unlabel':
                return False, None, True, None
    #=> no_roi, in_roi+label, not_entire_ori+unlabel
    
    mask = None
    if processtype=='label':
        if not (mask_region is None):
            maskdata = mask_region.fetch(data[1][0], data[1][1], patch_size, patch_size)
            mask = np.ndarray(buffer=maskdata, dtype=np.uint8, 
                            shape=[patch_size, patch_size, mask_slide_band])
        else:
            return False,None,None,None
    elif processtype=='unlabel' and  not(mask_region is None):
        return False,None,True,None

    #=> no_roi+mask+label, no_roi+unlabel, in_roi label+mask+label, not_entire_ori+unlabel
    patchdata = img_region.fetch(data[1][0], data[1][1], patch_size, patch_size)
    img = np.ndarray(buffer=patchdata, dtype=np.uint8, 
                     shape=[patch_size, patch_size, img_slide_band])
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    
    ##class judgement
    np.seterr(divide='ignore', invalid='ignore')

    # check saturation -> rgb too close -> gray no color info(white/black no saturation)
    sat = np.nan_to_num((np.amax(img, axis=2) - np.amin(img, axis=2))/255)
    pix_sat_count = (sat > 0.2).sum() 
    all_pix_count = (sat > -1).sum()
    count_max = all_pix_count * 0.6 #unlabel 0.75, label 0.9, highqulity 0.5
    count_min = all_pix_count * 0.4 #unlabel 0.3, label, highqulity0.2

    # check avgcolor -> each pixel too close avg -> background 
    img_avg = np.average(img, axis=2)
    avg_divide = np.int32(img_avg)-np.int32(np.average(img_avg))
    avg_divide[avg_divide<0]=-avg_divide[avg_divide<0]
    avg_divide_count = (avg_divide<10).sum()
    all_divide_count = (avg_divide > -1).sum()
    same_color_check = True if avg_divide_count>0.95*all_divide_count else False

    # color check (H&E stain focus on red color)
    gr_div_count = (np.uint8((np.int32(img[:,:,1])-np.int32(img[:,:,0])).clip(min=0))>100).sum()
    br_div_count = (np.uint8((np.int32(img[:,:,2])-np.int32(img[:,:,0])).clip(min=0))>50).sum()

    if pix_sat_count <= (count_min):
        target = 'white_background'
    elif same_color_check:
        target = 'white_background'
    elif (gr_div_count>count_min or br_div_count>count_min):
        target = 'white_background'
        mark_check = True

    elif pix_sat_count < count_max and pix_sat_count > count_min:
        if mask is not None and 1 in mask:            
            target = 'partial_tissue_wtarget'
        else:
            target = 'partial_tissue'
    elif pix_sat_count >= count_max:
        if mask is None or 1 not in mask:
            target = 'tissue_background'   
        elif 0 in mask:
            target = 'partial_frontground' 
        else:
            target = 'whole_frontground'

    # if target=='tissue_background' and processtype=='unlabel':
    #     cv2.imshow(target,img)
    #     cv2.waitKey(0)
   
    return True, target, False, mark_check


def pruning(tifroot, maskroot, roiroot, save_path, name, datainfo, level, scale_level, processtype):
    '''
    read wsi file
    use data location list check each patch class -> filtering()
    save class dict info (contain patches' location)
    '''

    results = {
        'white_background' : [],        # non tissue background
        'tissue_background' : [],       # tissue but non labeled
        'partial_tissue' : [],          # partial non labeled tissue and white backgorund 
        'whole_frontground' : [],       # almost labeled
        'partial_frontground' : [],     # partial labeled and white backgorund 
        'partial_tissue_wtarget' : []   
    }

    ##read tif (wsi, roi, mask) 
    tifpath = os.path.join(tifroot, f'{name}'+'.'+datainfo['data_type'])
    img_slide = read_WSI(tifpath, datainfo['data_type'], scale_level)
    img_region = pyvips.Region.new(img_slide)

    roi_region = None
    if not (roiroot is None):
        roipath = os.path.join(roiroot, f'{name}'+'.'+datainfo['data_type'])
        if os.path.exists(roipath):
            roi_slide = read_WSI(roipath, datainfo['data_type'], scale_level)
            roi_region = pyvips.Region.new(roi_slide)
        else:
            roi_region = None

    mask_region, mask_bands = None, None
    if not (maskroot is None):
        maskpath = os.path.join(maskroot, f'{name}'+'.'+datainfo['data_type'])
        if os.path.exists(maskpath):
            mask_slide = read_WSI(maskpath, datainfo['data_type'], scale_level)
            mask_region = pyvips.Region.new(mask_slide)
            mask_bands = mask_slide.bands

    ##class assign
    for data in datainfo['datas']:
        workflag, target, roi_check, mark_check= filetering(
                    data = data,
                    img_region = img_region,
                    roi_region = roi_region,
                    mask_region = mask_region,
                    img_slide_band = img_slide.bands,
                    mask_slide_band = mask_bands,
                    patch_size = datainfo['patch_size'],
                    processtype = processtype
                    )
        data[1][0] = int(data[1][0]*datainfo['magnification'])
        data[1][1] = int(data[1][1]*datainfo['magnification'])
        if workflag:
            results[target].append(data)
        if mark_check:
            markset.add(name)

    ## save result
    #skip non result
    result_sum = 0
    for r_v in results.values():
        result_sum+=len(r_v)

    if result_sum>0: 
        #pkl format
        with open(os.path.join(save_path, f"{name}.pkl"), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        #transfer dict format to dataframe
        rows = []
        for key, values in results.items():
            for value in values:
                rows.append({"key": key, "label": value[0], "coordinates": value[1]})        
        results_df = pd.DataFrame(rows)
        feather_path = os.path.join(save_path, f"{name}.feather")
        results_df.to_feather(feather_path)


    stat_result = {
            "white_background" : len(results['white_background']),
            "tissue_background" : len(results['tissue_background']),
            "whole_frontground" : len(results['whole_frontground']),
            "partial_frontground" : len(results['partial_frontground']),
            "partial_tissue" : len(results['partial_tissue']),
            "partial_tissue_wtarget" : len(results['partial_tissue_wtarget']),
        }

    return name, stat_result


def getPatchDict(tifroot, datalist, save_path, patch_size, stride_size, level, scale_level):
    '''
    create dict which use key as filename and record each patch locations(left+top) in file.
    '''
    #need finish skip from already saved file
    check_dict = {'data':[],
                 'error':[]}
    datainfo_dict = {}

    for dataname in tqdm(datalist):
        data_type = dataname.split('.')[-1]

        if data_type=='svs':
            magnification = pow(4,(scale_level-level))
        elif data_type in ['tif','tiff','ndpi','mrxs']:
            magnification = pow(2,(scale_level-level))
        else:
            continue
        patch_size_drop = int(patch_size/magnification)
        stride_size_drop = int(stride_size/magnification)

        name = dataname[:-(len(data_type)+1)]
        
        try:
            slice = read_WSI(os.path.join(tifroot, dataname), data_type, scale_level)
            if slice is None:
                continue
            check_dict['data'].append(os.path.join(tifroot, dataname))
        except:
            check_dict['error'].append(os.path.join(tifroot, dataname))
            continue

        width = slice.width
        height = slice.height
        datas = []
        for sy in range(0, height-patch_size_drop-1, stride_size_drop):
            for sx in range(0, width-patch_size_drop-1, stride_size_drop):
                datas.append([name, [sx, sy], data_type])

        datainfo_dict[name] = { 'data_type':data_type,
                                'patch_size':patch_size_drop,
                                'stride_size':stride_size_drop,
                                'magnification':magnification,
                                'width':width,
                                'height':height,
                                'datas':datas
                            }

    with open(os.path.join(save_path, "check_list.json"), 'w') as f:
        json.dump(check_dict, f)

    return datainfo_dict


def execute(tifroot, maskroot, roiroot, save_path, \
            datalist, patchsize = 256, stride_size = 256, level=0, scale_level=2, processtype='label', maxworkers=1):
    #get patch dict (each patch locations in tif)
    datainfo_dict = getPatchDict(tifroot, datalist, save_path, patchsize, stride_size, level, scale_level)
    
    total_stats = {}
    
    if maxworkers>1:
        # multiple process
        with ProcessPoolExecutor(max_workers=maxworkers) as exe:
            pbar = tqdm(exe.map(pruning, repeat(tifroot), repeat(maskroot), repeat(roiroot), repeat(save_path),
                        datainfo_dict.keys(), datainfo_dict.values(), repeat(level), repeat(scale_level), repeat(processtype)
                        ))
            for name, stat_result in pbar:
                total_stats[name] = stat_result
    else:
        # single process
        for pk in tqdm(datainfo_dict.keys()):
            name, stat_result = pruning(tifroot, maskroot, roiroot, save_path, pk, datainfo_dict[pk], level, scale_level, processtype)
            total_stats[name] = stat_result

    with open(os.path.join(save_path, f"result_statistic_{processtype}.json"), 'w') as f:
        json.dump(total_stats, f)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tifroot', default="./dataset/images/")
    parser.add_argument('--maskroot', default="./dataset/masks/")
    parser.add_argument('--roiroot', default="./dataset/rois/") 
    parser.add_argument('--saveroot', default="./dataset/pkl/")
    args = parser.parse_args()
    
    tifroot = args.tifroot
    maskroot = args.maskroot
    roiroot = args.roiroot
    saveroot = args.saveroot
    
    # tifroot = './dataset/images/'
    # maskroot = './dataset/masks/'
    # roiroot = './dataset/rois/'
    # saveroot = './dataset/pkl/'
    os.makedirs(saveroot, exist_ok=True)

    level = 0                                   # based page of tif (page trainset used )
    scale_level = 2                             # filter used page of tif (1 level drop twofold, but in svs 1 level drop fourfold)

    name='MRCPS'                                #folder name
    PATCHSIZE = 512
    STRIDESIZE = [384, 512]
    datatypes = ['label','unlabel']             #process type

    MAXWORKERS = 4

    import time
    rct = time.time()
    for t_i in range(len(datatypes)):
        save_dirname = f'{name}_p{PATCHSIZE}_s{STRIDESIZE[t_i]}_{datatypes[t_i]}_level{level}'
        print(f"Start {save_dirname}")

        #save path of each type (train, test, unlabel)
        save_path = os.path.join(saveroot, save_dirname)
        os.makedirs(save_path, exist_ok=True)

        #read tifs list
        datalist = os.listdir(tifroot)

        #run with multi-process process
        execute(tifroot, maskroot, roiroot, save_path,\
                datalist, int(PATCHSIZE), int(STRIDESIZE[t_i]),
                level, scale_level, datatypes[t_i], \
                MAXWORKERS)
    print('cost time:', time.time()-rct)

