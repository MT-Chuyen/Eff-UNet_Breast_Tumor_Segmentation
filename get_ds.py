
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import glob
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