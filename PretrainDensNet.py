# -*- coding: utf-8 -*-
"""
Pretrain the densenets for Kaggle Recursion Cellular Image Classification 2019

author: Le Yan, Paul Gu
"""
import numpy as np 
import pandas as pd

from PIL import Image

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

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import  EarlyStopping, ModelCheckpoint

from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

path_data = '../data/kaggle/reccell/recursion-cellular-image-classification'
# device = 'cuda'
torch.manual_seed(0)

# subsampling parameters
origin_w = 512
origin_h = 512
default_w = 224
default_h = 224
stride = 144

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
    
    
    Example: load_image("/recursion-cellular-image-classification/train","HEPG2",3,1,"B02")
    
    '''
    
    folder_name = "{}/{}-{:02d}/Plate{}".format(path,cell,batch,plate_no)
    
    imgs = []
    for site in range(1,3):
        _ = []
        for channel in range(1,7):
            pathname = "{}/{}_s{}_w{}.png".format(folder_name, plate_loc, site, channel)
            _.append(cv2.imread(pathname, cv2.IMREAD_UNCHANGED))
        imgs.append(_)
    
    return imgs

def create_subsample(img, output_w=default_w, output_h=default_h, stride=stride, padding = 0, auto_extra_padding=True):
    '''
    Input:
        Image tensor
    
    '''
    # Size check
    img_c, img_w, img_h = img.shape
    if (img_w + 2* padding) < output_w and (img_h + 2* padding) < output_h :
        raise ValueError('Input image with padding is smaller than output size')
    
    
    # Add padding  
    if padding != 0:
        img_n = torch.zeros([img_c, (img_w+padding*2), (img_h+padding*2)], dtype=torch.float32)  
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
            _row = torch.zeros([img_c, (output_w), (output_h)], dtype=torch.float32)  
            _row[:, :, 0:(img_n_h - y_pt)] = img_n[:, x_pt:(x_pt+output_w), y_pt:]
            imgs.append(_row)
  
        x_pt = x_pt + stride
    # --- End Move on x
    
    
    if (auto_extra_padding):
        _col = torch.zeros([img_c, (output_w), (img_n_h)], dtype=torch.float32) 
        _col[:, 0:(img_n_w - x_pt), :] = img_n[:, x_pt:,:]
        
        # --- Move on y 
        y_pt = 0
        while ((y_pt + output_h) < img_n_h):
            imgs.append(_col[:, :, y_pt:(y_pt+output_h)])
            # Move pointer
            y_pt = y_pt + stride
        # --- End Move on y

        if (auto_extra_padding):
            _row = torch.zeros([img_c, (output_w),(output_h)], dtype=torch.float32)  
            _row[:, :, 0:(img_n_h - y_pt)] = img_n[:, x_pt:(x_pt+output_w), y_pt:]
            imgs.append(_row)

    return imgs
    
class ImagesDS(D.Dataset):
    # class to load training images
    def __init__(self, df, img_dir, mode='train', site=[1,2], channels=[1,2,3,4,5,6], subsample=True):
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.subsample = subsample
        self.nsite = len(site)
        self.w = (origin_w-default_w+stride-1)//stride+1
        self.h = (origin_h-default_h+stride-1)//stride+1
        self.len = df.shape[0]
        
    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, i, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site[i]}_w{channel}.png'])
        
    # subsampling needed
    def __getitem__(self, index):
        img_id = index // (self.w*self.h*self.nsite)
        res = index % (self.w*self.h*self.nsite)
        i = res // (self.w*self.h)
        j = res % (self.w*self.h)
        paths = [self._get_img_path(img_id, i, ch) for ch in self.channels]
        img_channel = [self._load_img_as_tensor(img_path) for img_path in paths]
        img = torch.cat(img_channel)
        if self.subsample:
            img_tmp = create_subsample(img)
            img = img_tmp[j] # torch.stack(img_tmp)
        if self.mode == 'train':
            return img, int(self.records[img_id].sirna)
        else:
            return img, self.records[img_id].id_code

    def __len__(self):
        return self.len*(self.w*self.h)*self.nsite

def get_data_loaders(train_batch_size, val_batch_size, channels, classes):
    # data loaders using ImagesDS class
    df = pd.read_csv(path_data+'/train.csv')
    if classes > 1108:
        df_control = pd.read_csv(path_data+'/train_controls.csv')
        df_control = df_control.drop('well_type', axis=1)  # drop the well type to concat
        df = pd.concat([df, df_control])
    df_train, df_val = train_test_split(df, test_size = 0.035, stratify = df.sirna, random_state=42)
    # df_test = pd.read_csv(path_data+'/test.csv')

    ds = ImagesDS(df_train, path_data, mode='train', channels=channels)
    ds_val = ImagesDS(df_val, path_data, mode='train', channels=channels)
    # ds_test = ImagesDS(df_test, path_data, mode='test', channels=channels)

    # train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # === ADDED FOR DISTRIBUTED >>>
    if args.distributed:
        train_sampler = D.distributed.DistributedSampler(ds)
    else:
        train_sampler = None
    # >>> ADDED FOR DISTRIBUTED ===

    train_loader = D.DataLoader(ds, sampler=train_sampler, batch_size=train_batch_size, shuffle=(train_sampler is None))

    val_loader = D.DataLoader(ds_val, batch_size=val_batch_size, shuffle=True)

    # === DDED FOR DISTRIBUTED >>>
    # return train_loader, val_loader
    return train_loader, val_loader, train_sampler
    # >>> ADDED FOR DISTRIBUTED ===
    # return train_loader, val_loader, test_loader

class DenseNetModel(nn.Module):
    def __init__(self, classes=1108, nchannel=6):
        super().__init__()
        # DenseNetModel
        preloaded = models.densenet121(pretrained=True)
        self.features = preloaded.features
        # train with separated channels
        trained_kernel = preloaded.features.conv0.weight
        self.features.conv0 = nn.Conv2d(nchannel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.features.conv0.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*nchannel, dim=1)

        self.fc = nn.Linear(1024, classes, bias=True)
        self.classifier = nn.Softmax()
        del preloaded

        # # number of features into the fully connected
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        out = self.classifier(out)
        return out


def run(train_batch_size, val_batch_size, epochs, lr, log_interval, channels, classes):
    # =====ADDED FOR DISTRIBUTED >>>
    # train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    train_loader, val_loader, train_sampler = get_data_loaders(train_batch_size, val_batch_size, channels, classes)
    # >>> ADDED FOR DISTRIBUTED ===
    model = DenseNetModel(classes, len(channels))

    if torch.cuda.is_available():
        device = "cuda"
        model.cuda(args.gpu)
    else:
        device = "cpu"

    # === ADDED FOR DISTRIBUTED >>>
    if args.distributed:
        model = DistributedDataParallel(model, [args.gpu])
    # >>> ADDED FOR DISTRIBUTED ===

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(),
    }

    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    # trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    # evaluator = create_supervised_evaluator(model,
                                            # metrics={"accuracy": Accuracy(),
                                            #          "nll": Loss(F.nll_loss)},
                                            # device=device)
    # process bar
    # pbar = ProgressBar(bar_format='')
    # pbar.attach(trainer, output_transform=lambda x: {'loss': x})
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    # === ADDED FOR DISTRIBUTED >>>
    if args.distributed:
        @trainer.on(Events.EPOCH_STARTED)
        def set_epoch(engine):
            train_sampler.set_epoch(engine.state.epoch)
    # >>> ADDED FOR DISTRIBUTED ===

    # train the fully connected first for two epochs, then update the pretrained net
    # @trainer.on(Events.EPOCH_STARTED)
    # def turn_on_layers(engine):
    #     epoch = engine.state.epoch
    #     if epoch == 1:
    #         for name, child in model.named_children():  # module.
    #             # pbar.log_message(name)
    #             if name == 'fc':
    #                 pbar.log_message(name + ' is unfrozen')
    #                 for param in child.parameters():
    #                     param.requires_grad = True
    #             else:
    #                 pbar.log_message(name + ' is frozen')
    #                 for param in child.parameters():
    #                     param.requires_grad = False
    #     if epoch == 3:
    #         pbar.log_message("Turn on all the layers")
    #         for name, child in model.named_children():  # module.
    #             for param in child.parameters():
    #                 param.requires_grad = True

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        # === ADDED FOR DISTRIBUTED >>>
        if args.distributed:
            train_sampler.set_epoch(iter + 1)
        # >>> ADDED FOR DISTRIBUTED ===

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    # average loss and accuracy of the training and validation

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        metrics = val_evaluator.run(train_loader).metrics
        tqdm.write(
            "Training Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "
            .format(engine.state.epoch, 
                metrics["loss"], 
                metrics["accuracy"] )
        )
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_and_display_val_metrics(engine):
        metrics = val_evaluator.run(val_loader).metrics
        tqdm.write(
            "Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "
            .format(engine.state.epoch, 
                metrics['loss'], 
                metrics['accuracy'] )
        )

    # exponential decreasing learning rate
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(engine):
        lr_scheduler.step()
        lr = float(optimizer.param_groups[0]['lr'])
        print("Learning rate: {}".format(lr))

    # early stopping handles
    handler = EarlyStopping(patience=6, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    checkpoints = ModelCheckpoint('models', 'Model_{}'.format(''.join([str(i) for i in channels])), save_interval=3, n_saved=3, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {'DenseNet121': model})

    trainer.run(train_loader, max_epochs=epochs)
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
    parser.add_argument("--val_batch_size", type=int, default=1000,
                        help="input batch size for validation (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument("--classes", type=int, default=1139,
                        help="number of classes to train (default: 1139)")
    parser.add_argument("--channels", type=int, default=1,
                        help="channels to train (default: channel 1)")

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
        args.log_interval, [args.channels], args.classes)

# model.eval()
# with torch.no_grad():
#     preds = np.empty(0)
#     for x, _ in tqdm_notebook(tloader): 
#         x = x.to(device)
#         output = model(x)
#         idx = output.max(dim=-1)[1].cpu().numpy()
#         preds = np.append(preds, idx, axis=0)

# submission = pd.read_csv(path_data + '/test.csv')
# submission['sirna'] = preds.astype(int)
# submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])