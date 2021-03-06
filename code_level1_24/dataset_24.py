import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
import cvlib as cv
from PIL import Image
from tqdm.notebook import tqdm
from time import time
import math

import multiprocessing as mp

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter
from pandas_streaming.df import train_test_apart_stratify

def get_img_mean_std(img_dir):
    """
    RGB 평균 및 표준편차를 수집하는 함수입니다.
    Args:
        img_dir: 학습 이미지 폴더(images)의 root directory(./train/train/images)
    Returns:
        mean, std
    """
    img_info = dict(means=[], stds=[])
    path = []
    filenames = os.listdir(img_dir)
    for i in filenames:
        if not i.startswith("._"):
            img_name = os.listdir(os.path.join(img_dir, i))
            for j in img_name:
                if not j.startswith('._'):
                    img = np.array(Image.open(os.path.join(img_dir,i+'/'+j)))
                    # (0,1,2)중 0번과 1번에 대해 평균
                    # 512, 384 에 대한 평균이 3장 나옴
                    img_info['means'].append(img.mean(axis=(0,1)))
                    img_info['stds'].append(img.std(axis=(0,1)))

    return np.mean(img_info["means"], axis=0) / 255., np.mean(img_info["stds"], axis=0) / 255.

class MaskDataset(data.Dataset):
    def __init__(self, img_dir, val_ratio, transform=None):
        """
        img_dir: 학습 이미지 폴더(images)의 root directory(./train/train/images)
        transforms안 넣으면 totensor변환만 해서 내보냄
        """
        self.transform = transform
        self.img_dir = img_dir
        self.val_ratio = val_ratio
        self.path = []
        self.label = []
        self.indexs = [] # 추가
        self.groups = [] # 추가
        self.class_num = 18
        self.setup()

    
    def setup(self):
        cnt = 0 # 추가
        filenames = os.listdir(self.img_dir)
        for i in filenames:
            if not i.startswith("._"):
                img_name = os.listdir(os.path.join(self.img_dir, i))
                for j in img_name:
                    if not j.startswith('._'):
                        self.path.append(i+'/'+j)
                        id, gender, race, age = i.split('_')
                        gender = 0 if gender == 'male' else 1
                        age = int(age)
                        age_range = 0 if age < 30 else 1 if age < 60 else 2
                        if 'incorrect' in j:
                            mask = 1
                        elif 'mask' in j:
                            mask = 0
                        elif 'normal' in j:
                            mask = 2
                        self.label.append(mask * 6 + gender * 3 + age_range)
                        self.indexs.append(cnt) # 추가
                        self.groups.append(id) # 추가
                        cnt += 1 # 추가


    def __getitem__(self, index):
        y = self.label[index]
        img_path = self.path[index]
    
        # img = Image.open(os.path.join(self.img_dir,img_path))
        img = np.array(Image.open(os.path.join(self.img_dir,img_path)))
        if self.transform != None:
            X = self.transform(image=img)["image"]
            # X = self.transform(img)
        else:
            tt = transforms.ToTensor()
            X = tt(img)
        return X, y


    def split_dataset(self):
        df = pd.DataFrame({"indexs":self.indexs, "groups":self.groups, "labels":self.label})

        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)
        train_index = train["indexs"].tolist()
        valid_index = valid["indexs"].tolist()
        return  [Subset(self, train_index), Subset(self, valid_index)]
        

    def __len__(self):
        return len(self.path)

def getDataloader(dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

def getDataloader_split(dataset, batch_size):
    train_dataset, val_dataset = dataset.split_dataset()
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    
    # 추출된 Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )

    return train_loader, val_loader

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = np.array(Image.open(self.img_paths[index]))

        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.img_paths)

import cvlib as cv
class TestDataset_fc(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])

        faces, confidences = cv.detect_face(image)
        width, height = 200, 250
        d_centerX, d_centerY = 190, 250
        
        max_conf = 0
        centerX=None
        for face, conf in zip(faces, confidences):
            if(conf>max_conf):
                max_conf = conf
                centerX, centerY = (face[0]+face[2])/2 , (face[1]+face[3])/2
                
        if not centerX:
            centerX, centerY = d_centerX, d_centerY

        startX = centerX//1 - width//2 
        startY = centerY//1 - height//2
        endX = startX + width
        endY = startY + height

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image=image.crop([startX,startY,endX,endY])
        

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)