from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import numpy as np
import pandas as pd
from PIL import Image
import cv2

class Dataset(data.Dataset):

    def __init__(self, data_json, with_label, transform=None):
        self.data_df = pd.read_json(data_json)
        self.with_label = with_label
        self.transform = transform
        
        # concat bands into (N,2,75,75) images
        band_1 = np.concatenate([img for img in self.data_df['band_1']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([img for img in self.data_df['band_2']]).reshape(-1, 75, 75)
        # self.full_img = np.stack([band_1, band_2], axis=1)
        band_3 = (band_1 + band_2) / 2
        
        # color composition
        # band_1 = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        # band_2 = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        # band_3 = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))
        
        # band_1 = ((band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min()))) * 255).astype(np.uint8)
        # band_2 = ((band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min()))) * 255).astype(np.uint8)
        # band_3 = ((band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min()))) * 255).astype(np.uint8)
        
        self.full_img = np.stack([band_1, band_2, band_3], axis=1)
        # self.full_img = {'band_1': band_1, 'band_2': band_2, 'band_3': band_3}
        
        self.Ids = [Id for Id in self.data_df['id']]
        
        if self.with_label:
            self.labels = np.array([label for label in self.data_df['is_iceberg']])

    def __getitem__(self, idx):
        img_1 = Image.fromarray(self.full_img[idx][0])
        img_2 = Image.fromarray(self.full_img[idx][1])
        img_3 = Image.fromarray(self.full_img[idx][2])
        
        # img_1 = self.full_img[idx][0]
        # img_2 = self.full_img[idx][1]
        # img_3 = self.full_img[idx][2]
        
        # img = cv2.imread(img)
        Id = self.Ids[idx]
        
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
        
        img_1 = np.array(img_1).astype(np.float32)
        img_2 = np.array(img_2).astype(np.float32)
        img_3 = np.array(img_3).astype(np.float32)
        
        img = np.stack([img_1, img_2, img_3], axis=0)
            
        target = -1
        if self.with_label:
            target = self.labels[idx]

        return img, target, Id

    def __len__(self):
        return len(self.data_df)
