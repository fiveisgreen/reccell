#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 10:35:41 2019

@author: zcgu
"""
import numpy as np
import pandas as pd
import h5py 


def get_h5_handle(path_name, readonly = True):
    if readonly:
        arg = 'r'
    else:
        arg = 'w'
        
    return h5py.File(path_name, arg)


def 