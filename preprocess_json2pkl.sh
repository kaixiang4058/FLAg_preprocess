#!/bin/bash
python preprocess_code/Gen_Mask_JSON.py --slide_path="/dataset/images/" --json_path="/dataset/annotations/" --save_path="/dataset/"
python preprocess_code/filtering_to_pkl_FL.py --tifroot="/dataset/images/" --maskroot="/dataset/masks/" --roiroot="/dataset/rois/" --saveroot="/dataset/pkl/"
python preprocess_code/Gen_datalist.py --pklroot_label="/dataset/pkl/MRCPS_p512_s384_label_level0/" --pklroot_unlabel="/dataset/pkl/MRCPS_p512_s512_unlabel_level0/" --saveroot="/dataset/"
