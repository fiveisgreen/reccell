#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:38:02 2019

@author: zcgu
"""
import numpy as np
from imgaug import augmenters as iaa

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Sometimes(0.5,iaa.Fliplr(0.5)), # horizontally flip 50% of the images
        iaa.Sometimes(0.5,iaa.Flipud(0.5)), # horizontally flip 50% of the images
        iaa.Sometimes(0.1,iaa.AdditiveGaussianNoise(scale=(0, 0.02*255))), # blur images with a sigma of 0 to 3.0
        iaa.Sometimes(0.5,iaa.Affine(rotate=(-20,20), translate_px={"x": (-10, 10), "y": (-5, 5)}))
    ])
      
  def __call__(self, imgs):
    # Input as Pytorch Tensor
    imgs = np.array(imgs)
    
    # [batch_size, channels, height, width] => [batch_size, height, width, channels]
    imgs = np.rollaxis(imgs,1,4)
    res = self.aug(images = imgs)
    return np.rollaxis(res,3,1)