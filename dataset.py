from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import RepeatedKFold

class Dataset(data.Dataset):

    def __init__(self, data, targets=None, Ids=None, transform=None):
        self.data = data
        self.targets = targets
        self.Ids = Ids
        self.transform = transform
        
        # color composition
        # band_1 = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        # band_2 = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        # band_3 = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))
        
        # band_1 = ((band_1 - band_1.min()) / (band_1.max() - band_1.min()) * 255).astype(np.uint8)
        # band_2 = ((band_2 - band_2.min()) / (band_2.max() - band_2.min()) * 255).astype(np.uint8)
        # band_3 = ((band_3 - band_3.min()) / (band_3.max() - band_3.min()) * 255).astype(np.uint8)
        
        # band_1 = ((band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min()))) * 255).astype(np.uint8)
        # band_2 = ((band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min()))) * 255).astype(np.uint8)
        # band_3 = ((band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min()))) * 255).astype(np.uint8)

    def __getitem__(self, idx):
        img = self.data[idx]
        
        if self.transform:
            img = self.transform(img)
        # img = img.transpose(1, 2, 0)
        # img = cv2.resize(img, (75, 75))
        # img = img.transpose(2, 0, 1).astype(np.float32)
    
        # img = Image.fromarray(self.full_img[idx], mode='RGB')
        
        # img_1 = Image.fromarray(self.full_img[idx][0])
        # img_2 = Image.fromarray(self.full_img[idx][1])
        # img_3 = Image.fromarray(self.full_img[idx][2])
        
        # img_1 = self.full_img[idx][0]
        # img_2 = self.full_img[idx][1]
        # img_3 = self.full_img[idx][2]
        
        # img = cv2.imread(img)
        
            # img_1 = self.transform(img_1)
            # img_2 = self.transform(img_2)
            # img_3 = self.transform(img_3)
        
        # img_1 = np.array(img_1).astype(np.float32)
        # img_2 = np.array(img_2).astype(np.float32)
        # img_3 = np.array(img_3).astype(np.float32)
        
        # img = np.stack([img_1, img_2, img_3], axis=0)
            
        Id = None
        target = None
        
        if self.targets is not None:
            target = self.targets[idx]
            
        if self.Ids is not None:
            Id = self.Ids[idx]

        return img, target, Id

    def __len__(self):
        return len(self.data_df)
        
def repeated_kfold_train_val(data_json, n_splits, n_repeats, transform=None):
    # read json
    data_df = pd.read_json(data_json)
    
    # reshape to 75x75
    data_df['band_1'] = data_df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    data_df['band_2'] = data_df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

    # concat bands into (N,2,75,75) images
    band_1 = np.concatenate([img for img in data_df['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([img for img in data_df['band_2']]).reshape(-1, 75, 75)
    band_3 = (band_1 + band_2) / 2
    
    # get targets
    targets = np.array([target for target in data_df['is_iceberg']])
    
    # get Ids
    Ids = np.array([Id for Id in data_df['id']])
    
    data = np.stack([band_1, band_2, band_3], axis=1)
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    train_val_datasets = []
    
    for train_idx, val_idx in rkf.split(data):
        train_data, train_targets, train_Ids = data[train_idx], targets[train_idx], Ids[train_idx]
        print(train_data.shape)
        print(train_targets.shape)
        print(train_Ids.shape)
        train_dataset = Dataset(train_data, train_targets, train_Ids, transform=transform)
        
        val_data, val_targets, val_Ids = data[val_idx], targets[val_idx], Ids[val_idx]
        val_dataset = Dataset(val_data, val_targets, val_Ids, transform=transform)
        print(val_data.shape)
        print(val_targets.shape)
        print(val_Ids.shape)
        
        train_val_datasets.append([train_dataset, val_dataset])
    
    return train_val_datasets
