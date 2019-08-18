#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this script prepare a new csv file with 

@author: leyan
"""

# import skimage.io as io
# from skimage.transform import rescale, resize, downscale_local_mean
# import PIL.Image
import numpy as np
import pandas as pd
import os
import warnings
import time

data_path = '/data1/lyan/CellularImage/20190721/RecursionCellClass' # 'aydin'  # 
data_new_path = '/data1/lyan/CellularImage/20190721/processed'

df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_control = pd.read_csv(os.path.join(data_path, 'train_controls.csv'))  # useful to train feature extractors
df_tcontrol = pd.read_csv(os.path.join(data_path, 'test_controls.csv'))
# df_control = pd.concat([df_control, df_tcontrol], ignore_index=True)

df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

count = 0
t0 = time.time()
# path stores the path to the image file
df  = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
# r=root, d=directories, f = files
for index, row in df_train.iterrows():
    plate = row['plate']
    well = row['well']
    for s in [1,2]:
        for i in range(9):
            if count>0 and count%1000==0:
                df = pd.concat([df, df0], ignore_index=True)
                df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
                print(count, time.time()-t0)
                t0 = time.time()

            path = os.path.join(data_new_path, 'train', row['experiment'], f'Plate{plate}', f'{well}_s{s}_0{i}.npy')
            if os.path.isfile(path):
                df1 = pd.DataFrame({'id_code':[row['id_code']], 'experiment':[row['experiment']], 'plate':[plate], 'well':[well], 
                                    'site':[s], 'patch':[i], 'path':[path], 
                                    'sirna':[row['sirna']]})
                df0 = df0.append(df1)
                count += 1
            else:
                print('Warning! lose path: '+path) 

df = pd.concat([df, df0], ignore_index=True)

df.to_csv(os.path.join(data_new_path, 'train.csv'), index=False)

df_new = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
# r=root, d=directories, f = files
for index, row in df_control.iterrows():
    plate = row['plate']
    well = row['well']
    for s in [1,2]:
        for i in range(9):
            if count>0 and count%1000==0:
                df_new = pd.concat([df_new, df0], ignore_index=True)
                df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
                print(count, time.time()-t0)
                t0 = time.time()

            path = os.path.join(data_new_path, 'train', row['experiment'], f'Plate{plate}', f'{well}_s{s}_0{i}.npy')
            if os.path.isfile(path):
                df1 = pd.DataFrame({'id_code':[row['id_code']], 'experiment':[row['experiment']], 'plate':[plate], 'well':[well], 
                                    'site':[s], 'patch':[i], 'path':[path], 
                                    'sirna':[row['sirna']]})
                df0 = df0.append(df1)
                count += 1
            else:
                print('Warning! lose path: '+path) 

df_new = pd.concat([df_new, df0], ignore_index=True)

df = pd.concat([df, df_new], ignore_index=True)

df_new = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
# r=root, d=directories, f = files
for index, row in df_tcontrol.iterrows():
    plate = row['plate']
    well = row['well']
    for s in [1,2]:
        for i in range(9):
            if count>0 and count%1000==0:
                df_new = pd.concat([df_new, df0], ignore_index=True)
                df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})
                print(count, time.time()-t0)
                t0 = time.time()

            path = os.path.join(data_new_path, 'test', row['experiment'], f'Plate{plate}', f'{well}_s{s}_0{i}.npy')
            if os.path.isfile(path):
                df1 = pd.DataFrame({'id_code':[row['id_code']], 'experiment':[row['experiment']], 'plate':[plate], 'well':[well], 
                                    'site':[s], 'patch':[i], 'path':[path], 
                                    'sirna':[row['sirna']]})
                df0 = df0.append(df1)
                count += 1
            else:
                print('Warning! lose path: '+path) 

df_new = pd.concat([df_new, df0], ignore_index=True)

df = pd.concat([df, df_new], ignore_index=True)
df.to_csv(os.path.join(data_new_path, 'train_withControls.csv'), index=False)


count = 0
t0 = time.time()
df  = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[]})
df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[]})
# r=root, d=directories, f = files
for index, row in df_test.iterrows():
    plate = row['plate']
    well = row['well']
    for s in [1,2]:
        for i in range(9):
            if count>0 and count%1000==0:
                df = pd.concat([df, df0], ignore_index=True)
                df0 = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[]})
                print(count, time.time()-t0)
                t0 = time.time()

            path = os.path.join(data_new_path, 'test', row['experiment'], f'Plate{plate}', f'{well}_s{s}_0{i}.npy')
            if os.path.isfile(path):
                df1 = pd.DataFrame({'id_code':[row['id_code']], 'experiment':[row['experiment']], 'plate':[plate], 'well':[well], 
                                    'site':[s], 'patch':[i], 'path':[path]})
                df0 = df0.append(df1)
                count += 1
            else:
                print('Warning! lose path: '+path) 

df = pd.concat([df, df0], ignore_index=True)
df.to_csv(os.path.join(data_new_path, 'test.csv'), index=False)

# load speed test
# from PIL import Image
# from time import time
# import numpy as np
# import torch
# from torchvision import transforms as T

# filename = '/data1/lyan/CellularImage/20190721/RecursionCellClass/train/HEPG2-01/Plate1/B05_s1_w1.png'

# npyname = '/data1/lyan/CellularImage/20190721/processed/train/HEPG2-01/Plate1/B05_s1_00.npy'

# t0 = time()
# img = T.ToTensor()(Image.open(filename)).to('cuda')
# t1 = time()
# npf = torch.from_numpy(np.load(npyname)).float().to('cuda')
# print(t1-t0, time()-t1)