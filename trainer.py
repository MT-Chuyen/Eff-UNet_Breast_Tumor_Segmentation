from get_ds import Breast
from model import UNet
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
from trainer import get_method

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #The mean and std of ImageNet,Medical images must be counted separately 
    ]
)

target_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

Data_transfered = Breast(root ='/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks',transform = transform, target_transform=target_transform)
print(len(Data_transfered))

from torch.utils.data.dataset import random_split
train_ds, test_ds = random_split(Data_transfered, [0.8, 0.2])

train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

print("Training Samples: {}".format(len(train_ds)))
print("Testing Samples: {}".format(len(test_ds)))
def trainer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = 1)

    model = get_method(args).to(device)
    optimizer = Adam(params = model.parameters(), lr = 0.001)

    epochs = 30

    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_dl))

    loss_fn = nn.CrossEntropyLoss()
    
    print(len(Data_transfered))
    old_loss = 1e26
    best_dct = None
    last_dct = None
    for epoch in range(epochs):
        model.train()
        tr_total_loss = 0
        tr_total_dice = 0
        tr_total_iou = 0
        for train_img, train_mask in tqdm(train_dl):
            train_img = train_img.to(device)
            train_mask = train_mask.to(device)

            train_gen_mask = model(train_img)        
            train_loss = loss_fn(train_gen_mask, train_mask)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()        
            
            tr_total_loss += train_loss.cpu().item()        
                
        mean_train_loss = tr_total_loss/len(train_dl)

        if mean_train_loss <= old_loss:
            old_loss = mean_train_loss
            best_dct = model.state_dict()
        
        last_dct = model.state_dict()

        print(f"Epoch: {epoch} - TrainLoss: {mean_train_loss}")
    model.load_state_dict(best_dct)