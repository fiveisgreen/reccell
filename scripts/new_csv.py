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

#data_path = '/data1/lyan/CellularImage/20190721/RecursionCellClass' # 'aydin'  # 
#data_new_path = '/data1/lyan/CellularImage/20190721/processed'
data_path = '../data/kaggle/reccell/recursion-cellular-image-classification' # 'aydin'  # 
data_new_path = '../data/kaggle/reccell/data'


exps = ['HEPG2-07', 'HUVEC-15', 'HUVEC-16', 'RPE-07', 'U2OS-03']  # exps for validation 

df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
df_control = pd.read_csv(os.path.join(data_path, 'train_controls.csv'))  # useful to train feature extractors
df_tcontrol = pd.read_csv(os.path.join(data_path, 'test_controls.csv'))
# df_control = pd.concat([df_control, df_tcontrol], ignore_index=True)

df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

def createpath(row):
    return os.path.join(data_new_path, 'train', row['experiment'], 'Plate{}'.format(row['plate']), '{}_s{}_0{}.pt'.format(row['well'], row['site'], row['patch']))

# df['path'] = df.apply(createpath, axis=1)

# df  = pd.DataFrame({'id_code':[], 'experiment':[], 'plate':[], 'well':[], 'site':[], 'patch':[], 'path':[], 'sirna':[]})

def newcsv(df):
    df_new = pd.DataFrame()
    for s in [1, 2]:
        for i in range(9):
            df0 = df.copy()
            df0['site'] = s
            df0['patch'] = i
            df0['path'] = df0.apply(createpath, axis=1)
            df_new = pd.concat([df_new, df0], ignore_index=True)
    return df_new


t0 = time.time()

# separate validation from train
df_val = pd.DataFrame(columns=df_train.columns)
for expr in exps:
    df_val = df_val.append(df_train.loc[df_train['experiment'] == expr])
    df_train = df_train[df_train['experiment'] != expr]

df_train = newcsv(df_train)
df_val = newcsv(df_val)

df_train.to_csv(os.path.join(data_new_path, 'train.csv'), index=False)
df_val.to_csv(os.path.join(data_new_path, 'validation.csv'), index=False)

t1 = time.time()

# add controls
df_val_control = pd.DataFrame(columns=df_train.columns)
for expr in exps:
    df_val_control = df_val_control.append(df_control.loc[df_control['experiment'] == expr])
    df_control = df_control[df_control['experiment'] != expr]

df_control = newcsv(df_control)
df_tcontrol = newcsv(df_tcontrol)  # all controls from test goes to training
df_val_control = newcsv(df_val_control)

df_train = pd.concat([df_train, df_control, df_tcontrol], sort=False, ignore_index=True)
df_val = pd.concat([df_val, df_val_control], sort=False, ignore_index=True)

df_train.to_csv(os.path.join(data_new_path, 'train_withControls.csv'), index=False)
df_val.to_csv(os.path.join(data_new_path, 'validation_withControls.csv'), index=False)

# make for test
df_test = newcsv(df_test)
df_test.to_csv(os.path.join(data_new_path, 'test.csv'), index=False)

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
