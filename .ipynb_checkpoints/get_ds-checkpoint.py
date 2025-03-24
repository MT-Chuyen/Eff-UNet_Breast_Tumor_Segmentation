from torch.utils.data import Dataset 
from PIL import Image
import torch
import glob
import albumentations as A
import cv2
import numpy as np
from torchvision import transforms
root = '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks'
class Breast(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.msks = glob.glob(root + '/*_tumor.png')
        normal = [
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case045.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case061.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case209.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case213.png',
        ]
        du = glob.glob(root+ '/*other*.png')
        self.imgs = [path for path in glob.glob(root+ '/*') if path not in (self.msks+du+normal)]


        # Ensure that the number of images and masks match
        assert len(self.imgs) == len(self.msks), "Number of images and masks don't match" 
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        msk_path = self.msks[idx]

        img = Image.open(img_path)
        msk = Image.open(msk_path)

        # Remove the 4th channel (if present)
        img = img.convert('RGB') if img.mode == 'RGBA' else img
        msk = msk.convert('L')  # Convert mask to grayscale (optional)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            msk = self.target_transform(msk)

        return (img, msk)
    

class WithAubumentations(Dataset):
    def __init__(self, root):
        self.resize = A.Compose([
            A.Resize(256, 256),
        ])

        self.aug_transforms = A.Compose([
            #Pixel-level Transforms
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2),
            # #Spatial-level Transforms
            A.HorizontalFlip(p=0.2),
            A.RandomCrop(width=256, height=256, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            # #Mixing-level Transforms
            # A.MixUp(p=0.2),
        ], additional_targets={'mask': 'mask'})

        self.norm = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self._segs = sorted(glob.glob(root + '/*_tumor.png'))
        normal = [
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case045.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case061.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case209.png',
            '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case213.png',
        ]
        du = glob.glob(root + '/*other*.png')
        self._images = sorted([path for path in glob.glob(root + '/*') if path not in (self._segs + du + normal)])

        print("Data Set Setting Up")
        print(len(self._images), len(self._segs))

    @staticmethod
    def process_mask(x):
        # Ensure mask is binary (0 and 1)
        x = x.to(dtype=torch.torch.float)
        return x

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image = Image.open(self._images[idx]).convert("RGB")
        mask = Image.open(self._segs[idx])
        img = image.convert('RGB') if image.mode == 'RGBA' else image
        msk = mask.convert('L')  # Convert mask to grayscale
        image = np.array(img)
        mask = np.array(msk)

        resized = self.resize(image=image, mask=mask)
        transformed = self.aug_transforms(image=resized['image'], mask=resized['mask'])
        transformed_img = self.norm(image=transformed["image"])["image"]
        transformed_mask = transformed["mask"]

        torch_img = torch.from_numpy(transformed_img).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_mask).unsqueeze(0).float()  # Add channel dimension

        return torch_img, self.process_mask(torch_mask)

import os, sys
from typing import *
import cv2
from PIL import Image
from rich.progress import track
import numpy as np
import argparse
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torch.nn.functional as F
from torch import Tensor
import albumentations as A

 
class Customvocalfolds(Dataset):
    def __init__(self, root ):
 
  
        self.resize = A.Compose(
            [
                A.Resize(256, 256),
            ]
        )

        self.aug_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode = cv2.BORDER_CONSTANT),
            ]
        )

        self.norm = A.Compose(
            [
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self._images = glob.glob(root+ "/img/*/*/*")
        self._segs = glob.glob(root+ "/annot/*/*/*")

        print("Data Set Setting Up")
        print(len(self._images),len(self._segs))

    @staticmethod
    def process_mask(x):
        uniques = torch.unique(x, sorted = True)
        if uniques.shape[0] > 3:
            x[x == 0] = uniques[2]
            uniques = torch.unique(x, sorted = True)
        for i, v in enumerate(uniques):
            x[x == v] = i
        
        x = x.to(dtype=torch.long)
        onehot = F.one_hot(x.squeeze(1), 7).permute(0, 3, 1, 2)[0].float()
        return onehot

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self._images[idx]).convert("RGB"))
        mask = np.array(Image.open(self._segs[idx]))

        resized = self.resize(image = image, mask = mask)
 
        transformed = self.aug_transforms(image = resized['image'], mask = resized['mask'])
        transformed_img = self.norm(image=transformed["image"])["image"]
        transformed_mask = transformed["mask"]


        torch_img = torch.from_numpy(transformed_img).permute(-1, 0, 1).float()
        torch_mask = torch.from_numpy(transformed_mask).unsqueeze(-1).permute(-1, 0, 1).float()

        return torch_img, self.process_mask(torch_mask)
    
data_breast_path = '/media/mountHDD3/data_storage/biomedical_data/Dataset/BrEaST/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks'
data_vocalfolds_path ="/media/mountHDD3/data_storage/z2h/vocalfolds"

def get_dataset(args):
    if args.ds == 'breast':
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),   # Randomly flip horizontally
                transforms.RandomRotation(30),     # Randomly rotate by +/- 10 degrees
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Color jitter
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
            ]
        )
        target_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        Data_transfered = Breast(root = data_breast_path, transform = transform, target_transform=target_transform)
        return Data_transfered
    
    if args.ds == 'breast_albumentation':
        return WithAubumentations(root = data_breast_path)
    # if args.ds == 'busi':
    if args.ds == 'vocalfolds': 
        return Customvocalfolds(root= data_vocalfolds_path)

