from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from PIL import Image

data_folder = '/media/swk/data/iceberg/data/'

# read json
full_train_df = pd.read_json(data_folder+'train.json')
test_df = pd.read_json(data_folder+'test.json')

# reshape to 75x75
full_train_df['band_1'] = full_train_df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
full_train_df['band_2'] = full_train_df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

# concat bands into (N,2,75,75) images
band_1 = np.concatenate([img for img in full_train_df['band_1']]).reshape(-1, 75, 75)
band_2 = np.concatenate([img for img in full_train_df['band_2']]).reshape(-1, 75, 75)
band_3 = (band_1 + band_2) / 2
        
# color composition
band_1 = ((band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min()))) * 255).astype(np.uint8)
band_2 = ((band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min()))) * 255).astype(np.uint8)
band_3 = ((band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min()))) * 255).astype(np.uint8)
        
full_imgs = np.stack([band_1, band_2, band_3], axis=-1)

for idx, img_arr in enumerate(full_imgs):
    # print(img_arr[:,:,0])
    # print(img_arr[:,:,1])
    # print(img_arr[:,:,2])
    # for row in img_arr[:,:,2]:
    #     print(row)
    img = Image.fromarray(img_arr, mode='RGB')
    filename = str(idx)+'.png'
    img.save(data_folder+'train/'+filename)
    # break
    
exit()
test_df['band_1'] = test_df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test_df['band_2'] = test_df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

# convert inc_angle to number
full_train_df['inc_angle'] = pd.to_numeric(full_train_df['inc_angle'], errors='coerce')
test_df['inc_angle'] = pd.to_numeric(test_df['inc_angle'], errors='coerce')

# split train and val set
train_df = full_train_df.sample(frac=0.8)
val_df = full_train_df[~full_train_df.isin(train_df)].dropna()

print(train_df.info())

train_df.to_json(data_folder+'train_1.json')
val_df.to_json(data_folder+'val_1.json')
