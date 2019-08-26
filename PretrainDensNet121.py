# -*- coding: utf-8 -*-
"""
Pretrain the densenets for Kaggle Recursion Cellular Image Classification 2019

author: Le Yan, Paul Gu
"""
import numpy as np 
import pandas as pd
import os

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
# from torchsummary import summary

from tqdm import tqdm, tqdm_notebook

from libs.dataloader import ImagesDS
from libs.model import DenseNetModel

import warnings
warnings.filterwarnings('ignore')

def get_data_loaders(train_batch_size, val_batch_size, channels, classes, device='cpu'):
    # data loaders using ImagesDS class
    
    # if classes > 1108:
    #     df_train = pd.read_csv(args.data_path+'/train_withControls.csv')
    #     df_val = pd.read_csv(args.data_path+'/validation_withControls.csv')
    # else:
    df_train = pd.read_csv(args.data_path+'/train.csv')
    df_val = pd.read_csv(args.data_path+'/validation.csv')

    ds = ImagesDS(df_train, mode='train', channels=channels, device=device)
    ds_val = ImagesDS(df_val, mode='train', channels=channels, device=device)

    train_loader = D.DataLoader(ds, sampler=None, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = D.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader

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
            ep0 = epoch = int(args.checkpoint.rsplit('_',1)[1])
        else:
            cpfile = 'models/Model_pretrained_DenseNet121.pth'
            ep0 = epoch = 1
        checkpoint = torch.load(cpfile)
        model.load_state_dict(checkpoint)
    else:
        ep0 = epoch = 0

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
    criterion = nn.CrossEntropyLoss() # 
    # exponential decreasing learning rate
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    # process bar
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
    while epoch < epochs:
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

        model.train()
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
            if i>0 and i % log_interval == 0:
                pbar.desc = desc.format(tloss/(i+1))
                pbar.update(log_interval) 
                # print(t1-t0, t2-t1, t3-t2)
                # print(psutil.cpu_percent())
                # print(psutil.virtual_memory())  # physical memory usage
        # done epoch
        pbar.desc = desc.format(tloss/tlen)
        pbar.update(tlen%log_interval)
 
        # save checkpoints
        if (epoch+1)%save_interval==0:
            ch = ''.join([str(i) for i in channels])
            torch.save(model.state_dict(), f'models/Model_pretrained_{ch}_DenseNet121_{epoch+1}.pth')
            if (epoch+1)//save_interval>n_saved:
                os.remove(f'models/Model_pretrained_{ch}_DenseNet121_{epoch+1-save_interval*n_saved}.pth')

        model.eval()
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

        # reset process bar
        pbar.desc = desc.format(tloss/(i+1))
        pbar.update(log_interval) 
        pbar.n = pbar.last_print_n = 0
        # stop training if vloss keeps increasing for patience
        if epoch-ep0>=patience and all([vl_track[-1-i]>vl_track[-2-i] for i in range(patience-1)]):
            break

        # update learning
        if epoch>=burn:
            lr_scheduler.step()
            lr = float(optimizer.param_groups[0]['lr'])

        epoch += 1

    # checkpoint ignite issue https://github.com/pytorch/ignite/pull/182
    pbar.close()

def savefeatures(df, features, ch): 
    """
    save features to the paths in dataframe df

    """
    for index, row in df.iterrows():
        _fname = row.path.rsplit('.',1)[0]+f'_{ch}.pt'
        torch.save(features[index].cpu(), _fname)

def pred(batch_size, log_interval, mode, channels, classes):

    # load model
    model = DenseNetModel(classes, len(channels), forfeatures=True)

    ch = ''.join([str(i) for i in channels])
    checkpoint = torch.load(f'models/Model_Pretrained_{ch}_DenseNet121.pth')
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        device = "cuda"
        model.cuda(args.gpu)
    else:
        device = "cpu"

    # load data
    df_test = pd.read_csv(args.data_path+'/'+mode+'.csv')
    ds_test = ImagesDS(df_test, mode=mode, channels=channels)
    test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    pbar = tqdm(
        initial=0, leave=False, total=len(test_loader),
    )

    model.eval()
    # preds = prediction(model, test_loader, df_test)
    with torch.no_grad():
        preds = np.empty(0)
        for i, (x, _) in enumerate(loader): 
            x = x.cuda(args.gpu)
            output, features = model(x)
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)  
            if forfeatures:
                savefeatures(df_test.iloc[i*batch_size:min((i+1)*batch_size, df_test.shape[0])], features, ch)
            iter = i % len(loader) + 1
            if iter % log_interval == 0:
                pbar.update(log_interval)
            del x, output, features

    df = pd.read_csv(args.data_path + '/'+mode+'.csv')
    df['sirna'] = preds.astype(int)
    df = df.set_index('id_code')
    df.to_csv(mode+f'_{ch}_predictions.csv', index=True, columns=['sirna'])  # 'id_code',


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prediction", type=bool, default=False,
                        help="prediction mode (default: False)")
    parser.add_argument("--mode", type=str, default='train',
                        help="prediction mode (default: train)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training (default: 64)")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="input batch size for validation (default: 100)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="number of batches before logging training status")
    parser.add_argument("--classes", type=int, default=1140,
                        help="number of classes to train (default: 1139)")
    parser.add_argument("--channels", type=str, default="1,2,3,4,5,6",
                        help="channels to train (default: channel 1,2,3,4,5,6)")
    parser.add_argument("--pretrained", type=bool, default=False,
                        help="channels to train (default: false)")
    parser.add_argument("--checkpoint", type=str, default='',
                        help="channels to train (default: '')")
    parser.add_argument("--num_workers", type=int, default=1, 
                        help="Number of CPUs to load data (default: 1)")
    parser.add_argument("--data_path", type=str, default='/home/lyan/Documents/CellAna/data/p128', 
                        help="path to the data.") # '/data1/lyan/CellularImage/20190721/processed'  # 

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

    # OMP_NUM_THREADS=1
    torch.set_num_threads(args.num_workers)
    torch.backends.cudnn.benchmark = True

    # device = 'cuda'
    torch.manual_seed(0)

    if args.distributed:
        torch.distributed.init_process_group(args.dist_backend,
                                             init_method=args.dist_method,
                                             world_size=args.world_size, rank=args.rank)
    # >>> ADDED FOR DISTRIBUTED ===

    if not args.prediction:
        run(args.batch_size, args.val_batch_size, args.epochs, args.lr, 
                    args.log_interval, [int(i) for i in args.channels.split(',')], args.classes)
    else:
        pred(args.batch_size, args.log_interval, args.mode, 
                    [int(i) for i in args.channels.split(',')], args.classes)


