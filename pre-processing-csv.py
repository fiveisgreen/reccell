#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:22:14 2019

@author: zcgu
"""

import pandas as pd
import os.path as op
import argparse
from sklearn.model_selection import train_test_split

'''
def preproc_csv_block(df, mode, data_path_root, num_sample = 9):
    # File path
    df['path'] = "{}/{}/{}/Plate{}.pt".format(data_path_root, mode, df['experiment'],df['plate'])
    
    # idx in path
    gb = df_train.get_group(['experiment','plate'])
    for i in num_sample:
       # df['npy_idx'] = 
'''



def preproc_csv(df, mode, data_path_root, num_sample, cell_split = False):
    df_list = []
    sites = 2
    
    for site in range(1, (sites+1)):
        df['site'] = site
        df_list.append(df.copy())
    
    df = pd.concat(df_list,ignore_index=True)
    
    df_list = []
    for sample in range(0, num_sample):
        df['patch'] = "{:02d}".format(sample)
        df_list.append(df.copy())
        
    df = pd.concat(df_list,ignore_index=True)
    
    def _createpath(row):
        return "{}/{}/{}/Plate{}/{}_s{}_{}.pt".format(data_path_root, mode, row['experiment'],row['plate'],row['well'],row['site'],row['patch'])
    
    def _get_cell(row):
        return row['experiment'].split("-")[0]
    
    df['path'] = df.apply(_createpath,axis =1)
    
    df = df.sort_values(['id_code','site','patch'],ascending = [True,True,True])
    
    df['cell'] = df.apply(_get_cell,axis =1)
    
    
    if cell_split:
        gb = df.groupby(['cell'])
        
        for exp, df_cell in gb:
            dpath = op.join(data_path_root ,'{}_{}.csv'.format(mode,exp))
            df_cell.to_csv(dpath, index = False)
    else:
        dpath = op.join(data_path_root , '{}.csv'.format(mode))
        df.to_csv(dpath, index = False)


if __name__ == "__main__":
    
       
    parser = argparse.ArgumentParser(description='path, data_path_root, split_cell = True/False')
    parser.add_argument("--path", type=str, default="/home/zcgu/workspace/data/kaggle/reccell/recursion-cellular-image-classification/",
                        help="path to downloaded data")
    parser.add_argument("--data_path_root", type=str, default="/home/zcgu/workspace/data/kaggle/reccell/data",
                        help="path to processed data")
    parser.add_argument("--n_subsample", type=int, default=16,
                        help="# of subsamples")
    parser.add_argument("--cell_type_split", type=bool, default=True,
                    help="whether split train and test by cell_type")
    parser.add_argument("--include_control", type=bool, default=False,
                    help="whether include control")
    args = parser.parse_args()
    
    path = args.path
    data_path_root = args.data_path_root
    
    
    ftrain = path + "train.csv"
    ftest = path + "test.csv"
    ftrain_c = path + "train_controls.csv"
    ftest_c = path + "test_controls.csv"
    
    
    df_train = pd.read_csv(ftrain)
    df_test = pd.read_csv(ftest)
    df_train_c = pd.read_csv(ftrain_c)
    df_test_c = pd.read_csv(ftest_c)
    
    
    if args.include_control:
        df_train = pd.concat(df_train,df_train_c,ignore_index=True)
        df_test = pd.concat(df_test,df_test_c,ignore_index=True)
    
    # Train-validation split at 20%
    
    df_train_n, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    
    df_train_n = df_train_n.copy()
    df_val = df_val.copy()
    
    preproc_csv(df_train_n, "train",data_path_root, args.n_subsample,args.cell_type_split)
    preproc_csv(df_val, "validation",data_path_root, args.n_subsample,args.cell_type_split)
    preproc_csv(df_test, "test",data_path_root, args.n_subsample,args.cell_type_split)