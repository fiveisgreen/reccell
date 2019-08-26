#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:40:40 2019

@author: zcgu, lyan
"""

import torch
import torch.utils.data as D

torch.set_num_threads(1)

class ImagesDS(D.Dataset):
    # class to load training images
    def __init__(self, df, mode='train', channels=range(1,7), subsample=False, device='cpu', transform = None):
        self.records = df.to_records(index=False)
        self.channels = [i-1 for i in channels]
        self.mode = mode
        self.device = device
        self.len = df.shape[0]
        self.transform = transform
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        return torch.load(file_name).float()/255. # torch.from_numpy(np.load(file_name)/255.).float() # normalize to [0,1]  

    def _get_img_path(self, index):
        return self.records[index].path

    # subsampling needed
    def __getitem__(self, index):
        # t0 = time()
        path = self._get_img_path(index)
        img = self._load_img_as_tensor(path)
        # t1 = time()
        img = img[self.channels,...] # torch.cat([img[i,...].unsqueeze(0) for i in self.channels])  #  img[self.channels,...]  # 
        
        if self.transform:
            img = torch.from_numpy(self.transform(img))
        
        # print(t1-t0, time()-t1)
        if self.mode == 'train':
            return img, self.records[index].sirna #.to(self.device), torch.tensor(self.records[index].sirna).long().to(self.device)
        else:
            return img, 1138 #for unknown cells, torch.tensor(self.records[index].plate).long().to(self.device)

    def __len__(self):
        return self.len
        