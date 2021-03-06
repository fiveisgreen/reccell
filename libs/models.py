#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:49:25 2019

@author: zcgu, lyan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Get feature extractor
def get_featureExtractor(model):
    if model == "dense121":
        # Return a reference to class
        return Features_Dense121


# DenseNet module
class Features_Dense121(nn.Module):
    def __init__(self, classes=1108, nchannel=6, mode=0, path_to_model="models", pretrain_cp=None):
        # Note:
        #   pass pretrain_cp = None to use pretrained weight from pytorch
        
        super().__init__()
        
        # Mode = 0 - Train mode, return only prediction
        # Mode = 1 - Feat. mode, return only feature before ReLU and Adaptive
        # Mode = 2 - Feat. mode, return input of FC layer
        # Mode = 3 - Hybrid mode, return (prediction, feature before ReLU and Adaptive, input of FC)
        self.mode = mode

        # Load Architecture: DenseNetModel-121
        preloaded = models.densenet121(pretrained=(pretrain_cp is None))
        self.features = preloaded.features    
        self.fc = nn.Linear(1024, classes, bias=True)
        
        # Modify conv0 to adapt n_channel training
        trained_kernel = preloaded.features.conv0.weight
        self.features.conv0 = nn.Conv2d(nchannel, 64, kernel_size=7, stride=2, padding=3,)

        if pretrain_cp:  # pretrained
            checkpoint = torch.load(os.path.join(path_to_model, pretrain_cp))
            self.load_state_dict(checkpoint)
        else:
            with torch.no_grad():
                self.features.conv0.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*nchannel, dim=1)

        del preloaded

    def forward(self, x):
        features = self.features(x)
        if self.mode == 1:
            return features
        else:
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)  # global average pooling
            if self.mode == 2:
                return out
            else:
                pred = self.fc(out)
                if self.mode == 0:
                    return pred
                elif self.mode == 3:
                    return pred, features, out
                else:
                    raise ValueError("Undefined Mode")


# class PredictModel(nn.Module):
#     def __init__(self, classes=1108, channels=[1,2,3,4,5,6], default_w=224, default_h=224, device='cpu'):
#         super().__init__()
#         # DenseNetModel as feature extractor
#         self.nchannel = len(channels)
#         self.features = nn.ModuleList()
#         self.default_h = default_h
#         self.default_w = default_w
#         for i in range(nchannel):
#             self.features.append(DenseFeatures(1, pretrain_local=True, pretrain_cp=f'Model_{i}_DenseNet121.pth').to(device))

#         # classifier with cell type as a feature
#         self.classifier = nn.Linear(1024*nchannel+4, classes, bias=True).to(device)

#     def forward(self, x): # cell type provided as a feature
#         features = []
#         celltypes= x[:,-1]  # batchsize x 1
#         x = x[:,:-1].view([-1, self.nchannel, self.default_w, self.default_h])
#         x = list(torch.split(x, 1, dim=1))
#         for i in range(self.nchannel):
#             fs = self.features[i](x[i])
#             fs = F.relu(fs, inplace=True)
#             fs = F.adaptive_avg_pool2d(fs, (1, 1)).view(fs.size(0), -1)  # global average pooling
#             features.append(fs)
#         z = F.one_hot(celltypes.to(torch.int64),4).squeeze()
#         features.append(z.to(torch.float32))
#         out = torch.cat(features,1)
#         return self.classifier(out)
