import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.notebook import tqdm
from time import time
import math
from pytz import timezone
import datetime as dt
import argparse
import random

import seaborn as sns
import multiprocessing as mp

from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, GaussianBlur, RandomRotation, ColorJitter
from efficientnet_pytorch import EfficientNet

from dataset_24 import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def train_all_data(data_dir, model_dir, args):
    seed_everything(args.seed)

    print("Train all data start")

    mean, std  = np.array([0.56019358, 0.52410121, 0.501457  ]), np.array([0.23318603, 0.24300033, 0.24567522])
    # dataset 및 Transform정의
    albu_transform = albumentations.Compose([
        albumentations.Resize(int(512 / 2), int(384/ 2)),
        albumentations.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    dataset = MaskDataset(
        img_dir = data_dir,
        transform=albu_transform,
    )
    all_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4, 
        shuffle=True
    )

    # backborn모델 불러오기
    vision_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=18)
    torch.nn.init.xavier_uniform_(vision_model._fc.weight)
    stdv = 1. / math.sqrt(vision_model._fc.weight.size(1))
    vision_model._fc.bias.data.uniform_(-stdv, stdv)

    # 학습에 필요한 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_model.to(device)
    LEARNING_RATE = args.lr 
    NUM_EPOCH = args.epochs
    class_weights = torch.FloatTensor([1.48816029143898, 1.9926829268292683, 9.843373493975903, 1.116120218579235, 1.0,  7.495412844036697, 7.4408014571949, 9.963414634146341, 49.21686746987952, 5.580601092896175, 5.0, 37.477064220183486,
                                            7.4408014571949, 9.963414634146341, 49.21686746987952, 5.580601092896175, 5.0, 37.477064220183486]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights )
    optimizer = torch.optim.Adam(vision_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H-%M"))
    
    torch.cuda.empty_cache()
    print('train start')
    for e in range(NUM_EPOCH):
        vision_model.train()
        for _, (images, labels) in enumerate(all_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

            with torch.set_grad_enabled(True): # train 모드일 시에는 gradient를 계산
                logits = vision_model(images)
                loss = loss_fn(logits, labels)
                
                _, preds = torch.max(logits, 1)
                loss.backward() 
                optimizer.step()
        print(f'{e+1} epoch end')

    print("학습 종료!")
    os.makedirs("/save_model", exist_ok=True)
    torch.save(vision_model,f"/save_model/efficientnet-b7.pt")


if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        from dotenv import load_dotenv
        import os
        load_dotenv(verbose=True)

        # Data and model checkpoints directories
        parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 2021)')
        parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 15)')
        parser.add_argument('--dataset', type=str, default='dataset', help='dataset augmentation type (default: dataset)')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')

        # Container environment
        parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/images')
        parser.add_argument('--model_dir', type=str, default='/save_model')


        args = parser.parse_args()
        print(args)

        data_dir = args.data_dir
        model_dir = args.model_dir

        train_all_data(data_dir, model_dir, args)