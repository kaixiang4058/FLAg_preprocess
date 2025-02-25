from scipy.sparse import lil_matrix
import numpy as np
import json
from tqdm import tqdm
import cv2
import os
import sys
import openslide
from tifffile import TiffWriter


def save_pyramid_tif(output_path, dense_mask, tile_size=512, compression="deflate"):
    levels = []
    current_image = dense_mask

    # 逐層縮小圖像，生成金字塔層級
    while min(current_image.shape[:2]) > tile_size:
        levels.append(current_image)
        current_image = cv2.resize(
            current_image, 
            (current_image.shape[1] // 2, current_image.shape[0] // 2), 
            interpolation=cv2.INTER_AREA
        )
    levels.append(current_image)  # 最後一層

    # 使用 TiffWriter 寫入多層
    with TiffWriter(output_path, bigtiff=True) as tif:
        for i, level in enumerate(levels):
            subfiletype = 0 if i == 0 else 1  # 0: full resolution, 1: reduced resolution
            tif.write(
                level,
                tile=(tile_size, tile_size),
                compression=compression,
                photometric="minisblack",
                subfiletype=subfiletype
            )
    print(f"Pyramid TIFF saved to {output_path}")

def save_as_tif(slide_path, json_path, output_path, disease_label, tile_size=512):
    # 使用 openslide 打開圖像並獲取維度
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions

    # 使用稀疏矩陣
    mask = lil_matrix((height, width), dtype=np.uint8)

    # 讀取 JSON 文件
    with open(json_path, 'r') as jsonfile:
        data = json.load(jsonfile)

    label_profiles = disease_label['label_profile']
    annotations = data['annotation']


    # 遍歷標籤和注釋
    for label_type in label_profiles:
        for annotation in tqdm(annotations, desc=f"Processing label: {label_type['name']}"):
            if label_type['name'] == annotation['name'] or label_type['name'] == annotation['partOfGroup']:
                coordinates = np.array(annotation['coordinates'], dtype=np.int32)

                # 獲取多邊形的 Bounding Box
                x_min = max(0, min(coord[0] for coord in coordinates))
                x_max = min(width, max(coord[0] for coord in coordinates))
                y_min = max(0, min(coord[1] for coord in coordinates))
                y_max = min(height, max(coord[1] for coord in coordinates))

                # roi區域將額外處理
                if label_type['name'] == 'roi':
                    continue

                # 分塊處理
                for ty in range(y_min, y_max, tile_size):
                    for tx in range(x_min, x_max, tile_size):
                        block_width = min(tile_size, x_max - tx)
                        block_height = min(tile_size, y_max - ty)

                        # 創建臨時稠密矩陣來處理當前區塊
                        temp_mask = np.zeros((block_height, block_width), dtype=np.uint8)
                        sub_coords = coordinates - [tx, ty]

                        # 檢查座標是否在當前區塊內
                        valid_coords = [coord for coord in sub_coords if 0 <= coord[0] < block_width and 0 <= coord[1] < block_height]
                        if valid_coords:
                            cv2.fillPoly(temp_mask, [np.array(valid_coords, dtype=np.int32)], color=int(label_type['value']))

                            # 更新稀疏矩陣
                            non_zero_coords = np.nonzero(temp_mask)
                            for y, x in zip(*non_zero_coords):
                                mask[ty + y, tx + x] = max(mask[ty + y, tx + x], temp_mask[y, x])

    # 轉換稀疏矩陣為稠密矩陣
    dense_mask = mask.toarray()

    # 使用 tifffile 保存多分辨率金字塔結構 TIFF
    save_pyramid_tif(
        output_path,
        dense_mask,
        tile_size = tile_size,  # 分塊存儲
        compression="deflate",  # 壓縮方式
    )
    return 'OK'


def save_as_tif_ROI(slide_path, json_path, output_path_roi, disease_label, tile_size=512):
    # 使用 openslide 打開圖像並獲取維度
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.dimensions

    # 使用稀疏矩陣
    mask = lil_matrix((height, width), dtype=np.uint8)

    # 讀取 JSON 文件
    with open(json_path, 'r') as jsonfile:
        data = json.load(jsonfile)

    label_profiles = disease_label['label_profile']
    annotations = data['annotation']

    # 遍歷標籤和注釋
    for label_type in label_profiles:
        for annotation in tqdm(annotations, desc=f"Processing label: {label_type['name']}"):
            if label_type['name'] == annotation['name']:
                coordinates = np.array(annotation['coordinates'], dtype=np.int32)

                # 獲取多邊形的 Bounding Box
                x_min = max(0, min(coord[0] for coord in coordinates))
                x_max = min(width, max(coord[0] for coord in coordinates))
                y_min = max(0, min(coord[1] for coord in coordinates))
                y_max = min(height, max(coord[1] for coord in coordinates))

                # roi區域將額外處理
                if label_type['name'] != 'roi': 
                    continue

                # 分塊處理
                for ty in range(y_min, y_max, tile_size):
                    for tx in range(x_min, x_max, tile_size):
                        block_width = min(tile_size, x_max - tx)
                        block_height = min(tile_size, y_max - ty)

                        # 創建臨時稠密矩陣來處理當前區塊
                        temp_mask = np.zeros((block_height, block_width), dtype=np.uint8)
                        sub_coords = coordinates - [tx, ty]

                        # 檢查座標是否在當前區塊內
                        valid_coords = [coord for coord in sub_coords if 0 <= coord[0] < block_width and 0 <= coord[1] < block_height]
                        if valid_coords:
                            cv2.fillPoly(temp_mask, [np.array(valid_coords, dtype=np.int32)], color=int(label_type['value']))

                            # 更新稀疏矩陣
                            non_zero_coords = np.nonzero(temp_mask)
                            for y, x in zip(*non_zero_coords):
                                mask[ty + y, tx + x] = max(mask[ty + y, tx + x], temp_mask[y, x])

    # 轉換稀疏矩陣為稠密矩陣
    dense_mask = mask.toarray()

    # 使用 tifffile 保存多分辨率金字塔結構 TIFF
    save_pyramid_tif(
        output_path_roi,
        dense_mask,
        tile_size = tile_size,  # 分塊存儲
        compression="deflate",  # 壓縮方式
    )
    return 'OK'


if __name__ == "__main__":

    # slide_path = r'F:\Users\Jimmy\Desktop\tmp\CCM 19-07 1) 103 Lu Masson - 2023-02-14 14.38.41.ndpi'
    # json_path = r'F:\Users\Jimmy\Desktop\tmp\CCM 19-07 1) 103 Lu Masson - 2023-02-14 14.38.41.json'
    # output_path = r'F:\Users\Jimmy\Desktop\tmp\maskCCM 19-07 1) 103 Lu Masson - 2023-02-14 14.38.41.tif'
    # disease_label={
    #     "label_profile": [
    #         {"name": "f1", "value": 100},
    #         {"name": "background", "value": 150},
    #         {"name": "f2", "value": 255}
    #     ]
    # }

    slide_path = './dataset/images/'
    json_path = './dataset/annotations/'
    mask_path = './dataset/masks/'
    roi_path = './dataset/rois/'
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
        save_as_tif(slide_path, json_path, new_output_path, disease_label)
        new_output_roi_path = os.path.join(roi_path, f"{filename}.tif")
        save_as_tif_ROI(slide_path, json_path, new_output_roi_path, disease_label)
