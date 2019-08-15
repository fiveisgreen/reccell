#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:02:10 2019

@author: zcgu
"""

import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from queue import Queue
from threading import Thread
import h5py 



def detect_nucleolus(img):
    ret, threshed_img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    
    
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
    #                cv2.CHAIN_APPROX_SIMPLE)
    
    # with each contour, draw boundingRect in green
    # a minAreaRect in red and
    # a minEnclosingCircle in blue
    centers = []
    
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # get the min area rect
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(img, [box], 0, (0, 0, 255))
    
        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        centers.append((center + radius))
        # and draw the circle in blue
        img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    return centers 
    
    '''
    print(len(contours))
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    
    cv2.imshow("contours", img)
    
    cv2.imshow("contours", img)
    
    while True:
        key = cv2.waitKey(1)
        if key == 27: #ESC key to break
            break
    
    cv2.destroyAllWindows()
    '''

def load_image(path, cell, batch, plate_no,plate_loc):
    '''
    Input:
        path: string
        cell: string
        batch: int
        plate_no: int
        plate_loc: string
    
    Output
        A list of two list for each site. Each sublist has 6 channel
        
        [[c1,c2,c3,c4,c5,c6][c1,c2,c3,c4,c5,c6]]
          ------- ^ -------  ------- ^ -------
                site 1            site 2
    
    
    Example: load_image("/home/zcgu/workspace/data/kaggle/reccell/recursion-cellular-image-classification/train","HEPG2",3,1,"B02")
    
    '''
    
    if isinstance(batch,str):
        batch = int(batch)
    
    folder_name = "{}/{}-{:02d}/Plate{}".format(path,cell,batch,plate_no)
    
    imgs = []
    for site in range(1,3):
        _ = []
        for channel in range(1,7):
            pathname = "{}/{}_s{}_w{}.png".format(folder_name, plate_loc, site,channel)
            _.append(cv2.imread(pathname, cv2.IMREAD_UNCHANGED))
        imgs.append(_)
    
    return imgs



def create_subsample_2d(img, output_w,output_h,stride,padding = 0,auto_extra_padding=True):
    '''
    Input:
        Image tensor
        output_w: output width
        output_h: output_height
        stride
        padding: default = 0
        auto_extra_padding: default true
     
    Output:
        A list of imgs
        
    Example:
        create_subsample(img,200,200,150,0)
    
    '''
    # Size check
    img_w, img_h = img.shape
    if (img_w + 2* padding) < output_w and (img_h + 2* padding) < output_h :
        raise ValueError('Input image with padding is smaller than output size')
    
    
    # Add padding  
    if padding != 0:
        img_n = np.zeros(((img_w+padding*2),(img_h+padding*2)),dtype=np.uint8)  
        img_n[padding:-padding, padding:-padding] = img  
    
        img_n_w, img_n_h = img_n.shape
    else:
        img_n = img
        img_n_w, img_n_h = (img_w, img_h)
    # Create subsamples
    
    # Initialize pointers
    x_pt = 0
    y_pt = 0
    imgs = []
    
    # --- Move on x
    while( (x_pt + output_w) < img_n_w):
        
        # --- Move on y 
        y_pt = 0
        while ((y_pt + output_h) < img_n_h):
            imgs.append(img_n[x_pt:(x_pt+output_w), y_pt:(y_pt+output_h)])
            # Move pointer
            y_pt = y_pt + stride
        # --- End Move on y

        if (auto_extra_padding):
            _row = np.zeros(((output_w),(output_h)),dtype=np.uint8)  
            _row[:,0:(img_n_h - y_pt)] = img_n[x_pt:(x_pt+output_w), y_pt:]
            imgs.append(_row)
  
        x_pt = x_pt + stride
    # --- End Move on x
    
    
    if (auto_extra_padding):
        _col = np.zeros(((output_w),(img_n_h)),dtype=np.uint8) 
        _col[0:(img_n_w - x_pt), :] = img_n[x_pt:,:]
        
        # --- Move on y 
        y_pt = 0
        while ((y_pt + output_h) < img_n_h):
            imgs.append(_col[:, y_pt:(y_pt+output_h)])
            # Move pointer
            y_pt = y_pt + stride
        # --- End Move on y

        if (auto_extra_padding):
            _row = np.zeros(((output_w),(output_h)),dtype=np.uint8)  
            _row[:,0:(img_n_h - y_pt)] = _col[:, y_pt:]
            imgs.append(_row)

    return imgs


def create_subsample_3d(img, output_w,output_h,stride,padding = 0,auto_extra_padding=True):
    '''
    Input:
        Image tensor
        output_w: output width
        output_h: output_height
        stride
        padding: default = 0
        auto_extra_padding: default true
     
    Output:
        A list of imgs
        
    Example:
        create_subsample(img,200,200,150,0)
    
    '''
    # Size check
    # Size check
    img_c, img_w, img_h = img.shape
    if (img_w + 2* padding) < output_w and (img_h + 2* padding) < output_h :
        raise ValueError('Input image with padding is smaller than output size')
    
    
    # Add padding  
    if padding != 0:
        img_n = np.zeros([img_c, (img_w+padding*2), (img_h+padding*2)], dtype=np.float32)  
        img_n[:, padding:-padding, padding:-padding] = img  
    
        img_c, img_n_w, img_n_h = img_n.shape
    else:
        img_n = img
        img_n_w, img_n_h = (img_w, img_h)
    # Create subsamples
    
    # Initialize pointers
    x_pt = 0
    y_pt = 0
    imgs = []
    
    # --- Move on x
    while( (x_pt + output_w) < img_n_w):
        
        # --- Move on y 
        y_pt = 0
        while ((y_pt + output_h) < img_n_h):
            imgs.append(img_n[:, x_pt:(x_pt+output_w), y_pt:(y_pt+output_h)])
            # Move pointer
            y_pt = y_pt + stride
        # --- End Move on y

        if (auto_extra_padding):
            _row = np.zeros([img_c, (output_w), (output_h)], dtype=np.float32)  
            _row[:, :, 0:(img_n_h - y_pt)] = img_n[:, x_pt:(x_pt+output_w), y_pt:]
            imgs.append(_row)
  
        x_pt = x_pt + stride
    # --- End Move on x
    
    
    if (auto_extra_padding):
        _col = np.zeros([img_c, (output_w), (img_n_h)], dtype=np.float32) 
        _col[:, 0:(img_n_w - x_pt), :] = img_n[:, x_pt:,:]
        
        # --- Move on y 
        y_pt = 0
        while ((y_pt + output_h) < img_n_h):
            imgs.append(_col[:, :, y_pt:(y_pt+output_h)])
            # Move pointer
            y_pt = y_pt + stride
        # --- End Move on y

        if (auto_extra_padding):
            _row = np.zeros([img_c, (output_w),(output_h)], dtype=np.float32)  
            _row[:, :, 0:(img_n_h - y_pt)] = img_n[:, x_pt:(x_pt+output_w), y_pt:]
            imgs.append(_row)

    return imgs

class Row_dumper(Thread):
    
    def __init__(self,queue, dcate,output_w,output_h,stride,data_root_path,data_new_root_path):
        Thread.__init__(self)
        self.queue = queue
        self.dcate = dcate
        self.output_w = output_w
        self.output_h = output_h
        self.stride = stride
        self.data_root_path = data_root_path
        self.data_new_root_path = data_new_root_path
    
    def run(self):
        while True:
            row,_path,idx = self.queue.get()
            try:
                self._dump_subimage(row,_path,idx)
            finally:
                self.queue.task_done()
   
    def _dump_subimage(self, row,_path,idx):
        print("Saving {} -> {}".format(idx,_path))
        cell, batch = tuple(row['experiment'].split("-"))
        imgs = load_image(f'{self.data_root_path}/{self.dcate}', cell, batch, row['plate'],row['well'])
        for site in range(0,2):
            subimgs = create_subsample_3d(np.array(imgs[site]), self.output_w,self.output_h,self.stride)
            for sub_idx, subimg in enumerate(subimgs):
                _fname = "{}/{}_s{}_{:02d}".format(_path,row['well'],(site+1),sub_idx)
                np.save(_fname,subimg.astype(np.uint8))
#            for channel in range(0,6):
#                subimgs = create_subsample_2d(imgs[site][channel], self.output_w,self.output_h,self.stride)
#                for sub_idx, subimg in enumerate(subimgs):
#                    _fname = "{}/{}_s{}_w{}_{:02d}".format(_path,row['well'],(site+1),(channel+1),sub_idx)
#                    
#                    np.save(_fname,subimg)
    
    
    
def _dump_subsample(df,dcate,output_w,output_h,stride,data_root_path,data_new_root_path):
    queue = Queue()
    
    for x in range(16):
        dumper = Row_dumper(queue, dcate,output_w,output_h,stride,data_root_path,data_new_root_path)
        dumper.daemon = True
        dumper.start()
    
    for idx, row in df.iterrows():
        _path = "{}/{}/{}/Plate{}".format(data_new_root_path,dcate,row['experiment'],row['plate'])
        if not os.path.exists(_path):
            os.makedirs(_path)
        queue.put((row,_path,idx))
    
    queue.join()
    

def dump_subsample(output_w,output_h,stride,data_root_path,data_new_root_path):
    # Load CSV files
    #df_train = pd.read_csv(f'{data_root_path}/train.csv')
    df_test = pd.read_csv(f'{data_root_path}/test.csv')
    
    #_dump_subsample(df_train,"train",output_w,output_h,stride,data_root_path,data_new_root_path)
    _dump_subsample(df_test,"test",output_w,output_h,stride,data_root_path,data_new_root_path)

def get_h5_handle(path_name, readonly = True):
    if readonly:
        arg = 'r'
    else:
        arg = 'w'
        
    return h5py.File(path_name, arg)


if __name__ == "__main__":
    output_w = 224
    output_h = 224
    stride = 144
    
    data_root_path = "/data1/lyan/CellularImage/20190721/RecursionCellClass"
    data_new_root_path = "/data1/lyan/CellularImage/20190721/processed"
    
    if not os.path.exists(data_new_root_path):
        os.makedirs(data_new_root_path)
    
    dump_subsample(output_w,output_h,stride,data_root_path,data_new_root_path)
