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


def inference_24(data_dir, model_dir, args):
    test_dir = data_dir
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = model_dir
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    # test dataset생성
    mean, std  = np.array([0.56019358, 0.52410121, 0.501457  ]), np.array([0.23318603, 0.24300033, 0.24567522])
    albu_transform = albumentations.Compose([
        albumentations.Resize(int(512 / 2), int(384/ 2)),
        albumentations.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    dataset = TestDataset(image_paths, albu_transform)
    loader = DataLoader(
        dataset,
        shuffle=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vision_model = torch.load(os.path.join(model_dir, 'efficientnet-b7.pt'))
    vision_model.to(device)
    vision_model.eval()

    all_predictions = []
    for images in loader:
        with torch.no_grad():
            images = images.to(device)
            pred = vision_model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions
    os.makedirs("submission")
    submission.to_csv(f'submission/submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Container environment
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/eval')
    parser.add_argument('--model_dir', type=str, default='./save_model')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    inference_24(data_dir, model_dir, args)