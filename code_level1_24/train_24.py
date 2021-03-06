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


def train_24(data_dir, model_dir, args):
    seed_everything(args.seed)
    mean, std  = np.array([0.56019358, 0.52410121, 0.501457  ]), np.array([0.23318603, 0.24300033, 0.24567522])
    # dataset 및 Transform정의
    albu_transform = albumentations.Compose([
        albumentations.Resize(int(512 / 2), int(384/ 2)),
        albumentations.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    dataset = MaskDataset(
        img_dir = data_dir,
        val_ratio = 0.2,
        transform=albu_transform
    )

    train_loader, val_loader = getDataloader_split(dataset, args.batch_size)
    loader={'train':train_loader, 'val':val_loader}

    # 모델 불러오기
    vision_model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=18)
    torch.nn.init.xavier_uniform_(vision_model._fc.weight)
    stdv = 1. / math.sqrt(vision_model._fc.weight.size(1))
    vision_model._fc.bias.data.uniform_(-stdv, stdv)

    # 학습에 필요한 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_model.to(device)
    LEARNING_RATE = args.lr 
    NUM_EPOCH = args.epochs

    class_weights = torch.FloatTensor([1.48816029143898, 1.9926829268292683, 9.843373493975903, 1.116120218579235, 1.0,
 7.495412844036697, 7.4408014571949, 9.963414634146341, 49.21686746987952, 5.580601092896175, 5.0, 37.477064220183486,
 7.4408014571949, 9.963414634146341, 49.21686746987952, 5.580601092896175, 5.0, 37.477064220183486]).to(device)
 
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss_fn = create_criterion(args.criterion)
    optimizer = torch.optim.Adam(vision_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    now = (dt.datetime.now().astimezone(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H-%M"))

    best_val_f1 = 0.
    best_val_loss = 9999.

    torch.cuda.empty_cache()
    os.makedirs("/checkpoint", exist_ok=True)
    print('train_start')
    for epoch in range(NUM_EPOCH):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_acc = 0.
            n_iter = 0
            epoch_f1 = 0
            
            if phase == "train":
                vision_model.train()
            elif phase == "val":
                vision_model.eval()

            for ind, (images, labels) in enumerate(loader[phase]):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # parameter gradient를 업데이트 전 초기화함

                with torch.set_grad_enabled(phase == "train"): # train 모드일 시에는 gradient를 계산
                    logits = vision_model(images)
                    _, preds = torch.max(logits, 1)
                    loss = loss_fn(logits, labels)

                    if phase == "train":
                        loss.backward() 
                        optimizer.step()
            
                epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
                n_iter += 1
                
                running_loss += loss.item() * images.size(0) # 한 Batch에서의 loss 값, images.size(0) = batch size
                running_acc += torch.sum(preds == labels.data) # 한 Batch에서의 Accuracy 값
                
                
        # 한 epoch이 모두 종료되었을 때,
        epoch_loss = running_loss / len(loader[phase].dataset)
        epoch_acc = running_acc / len(loader[phase].dataset)
        epoch_f1 = epoch_f1/n_iter
        scheduler.step(epoch_loss)
        print(f"epoch-{epoch}의 {phase}-데이터 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, 평균 f1:{epoch_f1}")
        if phase == "val" and best_val_f1 < epoch_f1: 
            best_val_f1 = epoch_f1
            torch.save(vision_model, f"/checkpoint/model_{now}.pt")
            counter = 0
        else:
            counter += 1
        if phase == "val" and best_val_loss > epoch_loss: 
            best_val_loss = epoch_loss
    print("학습 종료!")
    print(f"최고 accuracy : {best_val_f1:3f}, 최고 낮은 loss : {best_val_loss:3f}")


if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        from dotenv import load_dotenv
        import os
        load_dotenv(verbose=True)

        # Data and model checkpoints directories
        parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 42)')
        parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 15)')
        parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
        parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')

        # Container environment
        parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/images')
        parser.add_argument('--model_dir', type=str, default='/checkpoint')

        args = parser.parse_args()
        print(args)

        data_dir = args.data_dir
        model_dir = args.model_dir

        train_24(data_dir, model_dir, args)