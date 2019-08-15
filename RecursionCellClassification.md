

# DenseNet model for Kaggle Recursion Cell Classification Competition

Project: [https://www.kaggle.com/c/recursion-cellular-image-classification](https://www.kaggle.com/c/recursion-cellular-image-classification)

Steps following: https://github.com/mingweihe/ImageNet

`cross entropy loss vs binary cross entropy with logit loss: multi classes vs binary sigmoid`

## 0. Set up environment

Follow env_setup.md for Ubuntu or env_setup_CentOS.md for CentOS 7 machines.

## 1. Download Dataset

Download competition dataset using kaggle api: https://github.com/Kaggle/kaggle-api

```bash
$ pip install kaggle --upgrade
```

To obtain APT credentials, please follow the instruction on https://github.com/Kaggle/kaggle-api. Once done, you will have a `kaggle.json` downloaded. 

Create a directory in your home

```bash
$ mkdir ~/.kaggle
```

and move the credentials into the directory,

```bash
$ mv kaggle.json ~/.kaggle/
```

For security, you can block the read access of other users by

```bash
$ chmod 600 ~/.kaggle/kaggle.json
```

You can also choose to export your Kaggle username and token to the environment:

```bash
$ export KAGGLE_USERNAME=datadinosaur
$ export KAGGLE_KEY=xxxxxxxxxxxxxx
```

Download the dataset with command lines:

```bash
$ kaggle competitions {list, files, download, submit, submissions, leaderboard}
$ kaggle datasets {list, files, download, create, version, init}
$ kaggle kernels {list, init, push, pull, output, status}
$ kaggle config {view, set, unset}
```

For the ImageNet Object Localization competition

```bash
$ kaggle competitions download recursion-cell-classification-challenge
```

* Dataset is about 87G and unzip is necessary later, please select the directory with enough storage.

## 2. Preprocess image data

### 2.1 Difficulties

1. large number of classes (1108 * 4 for four different cell types) compared to relatively small dataset, unbalancely sampled for four different cell types
2. large number of `.png` images take long CPU time to load, leave GPU time wasted.

### 2.2 Subsampling

1. 

### 2.3 Save as `.npy` type with all six channels grouped

![](/Users/leyan/Documents/NozomiFans/docs/docs/npyshape.png)

It takes much less time to load a `.npy` numpy array with all six channels to a pytorch tensor on GPU (~0.024s) than load the `.png` raw image using pillow with only one channel (~8.6s).

![](/Users/leyan/Documents/NozomiFans/docs/docs/npytime.png)

Dataset becomes 200G after subsampling and saved in an uncompressed form.

### 2.4 Create new csv tables for preprocessed images and dataloaders read `.npy` according to csv



## 3. Pretrain feature extractors for different channels

Assuming features from different channels

##4 Merge features from different channels

### 4.1 Forward different channels through the corresponding pretrained feature extractors 

###4.2 Concatenate ALL (6 x 1024) features and train a classifier 

###4.n Take channel intensities as features

## 5 Deal with different cell types

### 5.1 Take cell types as One-Hot feature

### 5.2 Train separate models for different cell types

TODO

###5.3 Bayesian inference

TODO

##6. Sample ensembling

We have experiment replication site 1 and 2, and loosely overlapped subregions (nine 224 x 224 from one 512 x 512 origin).

### 6.1 Majority ensembling

Take the most majority of 18 copies of predictions as the prediction

TODO: test improvement on val

###6.2 XGBoost ensembling

TODO: train a LightGBM model to determine the class from the 18 predictions of classes.

## 7. Improve accuracy

### 7.1 Valid augmentations

Rotation, shifts (translations), flip? (horizontal+vertical=rotation(180)), fill mode = 0?

###7.2 Model ensembling

##