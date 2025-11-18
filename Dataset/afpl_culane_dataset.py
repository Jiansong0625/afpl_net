"""
CULane dataset for AFPL-Net training and testing
Uses AFPL-specific ground truth generation
"""

import torch 
import numpy as np
import os
import cv2
from .afpl_base_dataset import AFPLBaseTrSet, AFPLBaseTsSet


class AFPLCULaneTrSet(AFPLBaseTrSet):
    """CULane training dataset for AFPL-Net"""
    
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms)
        self.data_root = cfg.data_root
        self.img_path_list, self.label_path_list = self.get_data_list()
        self.cut_height = cfg.cut_height
    
    def get_sample(self, index):
        img_path, label_path = self.img_path_list[index], self.label_path_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lanes = self.get_label(label_path)
        img, lanes = self.cut_img(img, lanes)
        return img, lanes
    
    def get_data_list(self):
        list_path = os.path.join(self.data_root, 'list/train_gt_new.txt')
        with open(list_path, 'r') as f:
            path_list = [line.strip(' \n').split(' ')[0][1:] for line in f.readlines()]

        img_path_list = [os.path.join(self.data_root, path) for path in path_list]
        label_path_list = [os.path.join(self.data_root, path.replace('.jpg', '.lines.txt')) for path in path_list]
        return img_path_list, label_path_list

    def get_label(self, label_path):
        with open(label_path, 'r') as f:
            lane_strs = f.readlines()
        lane_arrays = []
        for lane_str in lane_strs:
            lane_array = np.array(lane_str.strip(' \n').split(' ')).astype(np.float32)
            lane_array_size = int(len(lane_array)/2)
            lane_array = lane_array.reshape(lane_array_size, 2)[::-1, :]
            ind = np.where((lane_array[:, 0] >=0)&(lane_array[:, 1] >=0))
            lane_array = lane_array[ind]
            if lane_array.shape[0]>2:
                lane_arrays.append(lane_array)
        return lane_arrays
    
    def cut_img(self, img, lanes):
        img = img[self.cut_height:]
        for lane in lanes:
            lane[:, 1] = lane[:, 1] - self.cut_height
        return img, lanes


class AFPLCULaneTsSet(AFPLBaseTsSet):
    """CULane test dataset for AFPL-Net"""
    
    def __init__(self, cfg=None, transforms=None):
        super().__init__(cfg=cfg, transforms=transforms) 
        if self.is_val:
            self.txt_path = os.path.join(self.data_root, 'list/val.txt')
        else:
            self.txt_path = os.path.join(self.data_root, 'list/test.txt')
        with open(self.txt_path, 'r') as f:
            for line in f.readlines():
                self.file_name_list.append(line.strip('\n')[1:])
        self.img_path_list = [os.path.join(self.data_root, file_name) for file_name in self.file_name_list]