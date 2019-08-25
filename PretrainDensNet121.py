# -*- coding: utf-8 -*-
"""
Pretrain the densenets for Kaggle Recursion Cellular Image Classification 2019

author: Le Yan, Paul Gu
"""
import numpy as np 
import pandas as pd

from time import time

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD

import torch.distributed
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel

from torchvision import models, transforms as T
# from torchsummary import summary

# from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
# from ignite.metrics import Loss, Accuracy
# from ignite.contrib.handlers.tqdm_logger import ProgressBar
# from ignite.handlers import  EarlyStopping, ModelCheckpoint

from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# import psutil

# OMP_NUM_THREADS=1
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

#path_data = '/data1/lyan/CellularImage/20190721/processed'  # '/home/lyan/Documents/CellAna/data/processed'  # 
path_data = '../data/kaggle/reccell/data'
# device = 'cuda'
torch.manual_seed(0)

class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def init(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).init()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        yonehot.zero()
        y_onehot = Variable(y_onehot).cuda()
        yonehot.scatter(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits

class ImagesDS(D.Dataset):
    # class to load training images
    def __init__(self, df, mode='train', channels=range(6), subsample=False, device='cpu'):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.device = device
        self.len = df.shape[0]
        
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
        # print(t1-t0, time()-t1)
        if self.mode == 'train':
            return img, self.records[index].sirna #.to(self.device), torch.tensor(self.records[index].sirna).long().to(self.device)
        else:
            return img, 1139 #.to(self.device), torch.tensor(self.records[index].plate).long().to(self.device)

    def __len__(self):
        return self.len

def get_data_loaders(train_batch_size, val_batch_size, channels, classes, device='cpu'):
    # data loaders using ImagesDS class
    
    if classes > 1108:
        df_train = pd.read_csv(path_data+'/train_withControls.csv')
        df_val = pd.read_csv(path_data+'/validation_withControls.csv')
    else:
        df_train = pd.read_csv(path_data+'/train.csv')
        df_val = pd.read_csv(path_data+'/validation.csv')
    # df_train, df_val = train_test_split(df, test_size = 0.10, stratify = df.sirna, random_state=42)
    # df_test = pd.read_csv(path_data+'/test.csv')

    ds = ImagesDS(df_train, mode='train', channels=channels, device=device)
    ds_val = ImagesDS(df_val, mode='train', channels=channels, device=device)
    # ds_test = ImagesDS(df_test, path_data, mode='test', channels=channels)

    train_loader = D.DataLoader(ds, sampler=None, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = D.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader

class DenseNetModel(nn.Module):
    def __init__(self, classes=1108, nchannel=6):
        super().__init__()
        # DenseNetModel
        preloaded = models.densenet121(pretrained=True)
        self.features = preloaded.features
        # train with separated channels
        trained_kernel = preloaded.features.conv0.weight
        self.features.conv0 = nn.Conv2d(nchannel, 64, kernel_size=7, stride=2, padding=3,)
        with torch.no_grad():
            self.features.conv0.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*nchannel, dim=1)

        self.fc = nn.Linear(1024, classes, bias=True)
        # self.features.classifier = nn.Softmax()
        del preloaded

    # def save_checkpoint(self, cpfile='models/Model_pretrained_DenseNet121.pth'):
    #     # checkpoint = torch.load(cpfile)
    #     # self.model.load_state_dict(checkpoint)
    #     torch.save(self.state_dict(), cpfile)

    # def load_checkpoint(self, cpfile='models/Model_pretrained_DenseNet121.pth'):
    #     checkpoint = torch.load(cpfile)
    #     self.load_state_dict(checkpoint)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        # out = self.features.classifier(out)
        return out


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return np.array(res)

def run(train_batch_size, val_batch_size, epochs, lr, log_interval, channels, classes):
    # load model
    model = DenseNetModel(classes, len(channels))

    # load saved weights
    if args.pretrained:
        if len(args.checkpoint)>0:
            cpfile = 'models/'+args.checkpoint+'.pth'
        else:
            cpfile = 'models/Model_pretrained_DenseNet121.pth'
        checkpoint = torch.load(cpfile)
        model.load_state_dict(checkpoint)

    # to gpu
    if torch.cuda.is_available():
        device = "cuda"
        model.cuda(args.gpu)
    else:
        device = "cpu"

    # load data generator
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size, channels, classes, device=device)

    # for parallel
    if args.distributed:
        model = nn.DataParallel(model)

    # set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()   # 
    # exponential decreasing learning rate
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    # process bar
    # pbar = ProgressBar(bar_format='')
    # pbar.attach(trainer, output_transform=lambda x: {'loss': x})
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    # initialize parameters for iterating over epochs
    burn=2 # epoch burn for 
    patience=3  # number of epochs to early stop training
    vl_track=[]   # tracking the validation loss
    save_interval=1 # number of intervals to save weights
    n_saved=5    # number of weights to keep
    tlen = len(train_loader)
    vlen = len(val_loader)
    # print(tlen)
    for epoch in range(epochs):
        # frozen the pretrained layers, tran the fully connected classification layer
        print(f'{epoch+1}/{epochs}')
        print(f"Learning rate: {lr}")
        if epoch == 0:
            for name, child in model.named_children():  # module.
                # pbar.log_message(name)
                if name == 'fc':
                    print(name + ' is unfrozen')
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    print(name + ' are frozen')
                    for param in child.parameters():
                        param.requires_grad = False
        if epoch == burn:
            print("Turn on all the layers")
            # for name, child in model.named_children():  # module.
            for param in model.features.parameters():
                param.requires_grad = True

        # start of the epoch, 
        tloss = 0
        acc = np.zeros(1)
        t0 = time()
        for i, (x, y) in enumerate(train_loader): 
            x = x.to(device)
            y = torch.tensor(y).long().to(device)
            t1 = time()
            optimizer.zero_grad()
            output = model(x)
            # one hot for Binary Cross Entropy
            # target = torch.zeros_like(output, device=device)
            # target[np.arange(x.size(0)), y] = 1
            loss = criterion(output, y)
            loss.backward()
            t2 = time()
            optimizer.step()
            t3 = time()
            tloss += loss.item() 
            acc += accuracy(output, y)
            del loss, output, y, x
            torch.cuda.empty_cache()
            if i % log_interval == 0:
                pbar.desc = desc.format(tloss/(i+1))
                pbar.update(log_interval) 
                # print(t1-t0, t2-t1, t3-t2)
                # print(psutil.cpu_percent())
                # print(psutil.virtual_memory())  # physical memory usage
            t0 = time()  
        # save checkpoints
        if (epoch+1)%save_interval==0:
            ch = ''.join([str(i+1) for i in channels])
            torch.save(model.state_dict(), f'models/Model_pretrained_{ch}_DenseNet121_{epoch+1}.pth')
            if (epoch+1)//save_interval>n_saved:
                os.remove(f'models/Model_pretrained_{ch}_DenseNet121_{epoch+1-save_interval*n_saved}.pth')

        # compute loss and accuracy of validation
        vloss = 0
        vacc  = np.zeros(1)
        for i, (x, y) in enumerate(val_loader): 
            x = x.to(device)
            y = torch.tensor(y).long().to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            vloss += loss.item() 
            vacc += accuracy(output, y)
            del loss, output, y, x
            torch.cuda.empty_cache()
        vl_track.append(vloss)

        print('Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen))
        print('Epoch {} -> Validation Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, vloss/vlen, vacc[0]/vlen))

        # set process bar
        pbar.n = pbar.last_print_n = 0
        # stop training if vloss keeps increasing for patience
        if epoch>=patience and all([vl_track[-1-i]>vl_track[-2-i] for i in range(patience-1)]):
            break

        # update learning
        if epoch>=burn:
            lr_scheduler.step()
            lr = float(optimizer.param_groups[0]['lr'])

    # checkpoint ignite issue https://github.com/pytorch/ignite/pull/182
    pbar.close()

# batch_size = 64
# val_batch_size = 1000
# epochs = 20
# lr = 0.001
# log_interval = 100
# classes = 1139  # 1108 + 30 (pos control) + 1 (neg control)
# channels = [1,2,3,4,5,6]
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=100,
                        help="input batch size for validation (default: 100)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--classes", type=int, default=1139,
                        help="number of classes to train (default: 1139)")
    parser.add_argument("--channels", type=str, default="0,1,2,3,4,5",
                        help="channels to train (default: channel 1,2,3,4,5,6)")
    parser.add_argument("--pretrained", type=bool, default=False,
                        help="channels to train (default: false)")
    parser.add_argument("--checkpoint", type=str, default='',
                        help="channels to train (default: '')")

    # === ADDED FOR DISTRIBUTED >>>
    parser.add_argument("--dist_method", default="file:///home/user/tmp.dat", type=str,
                        help="url or file path used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--world_size", default=1, type=int, help="Number of GPUs to use.")
    parser.add_argument("--rank", default=0, type=int, help="Used for multi-process training.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU number to use.")
    # >>> ADDED FOR DISTRIBUTED ===
    args = parser.parse_args()

    # === ADDED FOR DISTRIBUTED >>>
    args.distributed = args.world_size > 1

    torch.cuda.set_device(args.gpu)
    if args.distributed:
        torch.distributed.init_process_group(args.dist_backend,
                                             init_method=args.dist_method,
                                             world_size=args.world_size, rank=args.rank)
    # >>> ADDED FOR DISTRIBUTED ===

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr, 
        args.log_interval, [int(i) for i in args.channels.split(',')], args.classes)

# @torch.no_grad()
# def prediction(model, loader):
#     preds = np.empty(0)
#     for x, _ in loader: 
#         x = x.to(device)
#         output = model(x)
#         idx = output.max(dim=-1)[1].cpu().numpy()
#         preds = np.append(preds, idx, axis=0)  
#     return preds

# fastest in indexing numpy array with list (than array) than cat torch tensor
# t0  =  time()
# a = np.random.randn(100,32,32)
# for _ in range(1000):
#     c =  a[np.arange(0,100,2)]
# print(time()-t0)
# t0  =  time()
# a = torch.randn((100,32,32))
# for _ in range(1000):
#     c =  torch.cat([a[i].unsqueeze(0) for i in range(0,100,2)])

# submission = pd.read_csv(path_data + '/test.csv')
# submission['sirna'] = ptest.astype(int)
# submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])