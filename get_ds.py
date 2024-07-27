from torch.utils.data import Dataset 
from PIL import Image
import torch
import glob
import albumentations as A
import cv2
import numpy as np
root = '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks'
class Breast(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.msks = glob.glob(root + '/*_tumor.png')
        normal = [
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case045.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case061.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case209.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case213.png',
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
            A.HorizontalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=3, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        ])

        self.norm = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        
        self._segs = sorted(glob.glob(root + '/*_tumor.png'))
        normal = [
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case045.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case061.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case209.png',
            '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks/case213.png',
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
