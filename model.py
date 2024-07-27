import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.dv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OnlyUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OnlyUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.up(x)
        return self.conv(x1)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetEfficientNetB7(nn.Module):
    def __init__(self, n_classes):
        super(UNetEfficientNetB7, self).__init__()
        self.encoder = models.efficientnet_b7(pretrained=True).features

        self.doubleconv = DoubleConv(3, 64)
        self.down1 = self.encoder[:2]   # Output: 32 channels
        self.down2 = self.encoder[2:4]  # Output: 48 channels
        self.down3 = self.encoder[4:6]  # Output: 80 channels
        self.down4 = self.encoder[6:8]  # Output: 224 channels
        self.down5 = self.encoder[8:]   # Output: 640 channels

        self.up1 = Up(3200, 256)
        self.up2 = Up(480, 128)
        self.up3 = Up(208, 64)
        self.up4 = Up(64 + 32, 32)
        self.up5 = Up(32 + 64, 32)

        self.out = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.doubleconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        logits = self.out(x)
        return logits


class UNetEfficientNetB7_without_sc2(nn.Module):
    def __init__(self, n_classes):
        super(UNetEfficientNetB7_without_sc2, self).__init__()
        self.encoder = models.efficientnet_b7(pretrained=True).features

        self.doubleconv = DoubleConv(3, 64)
        self.down1 = self.encoder[:2]   # Output: 32 channels
        self.down2 = self.encoder[2:4]  # Output: 48 channels
        self.down3 = self.encoder[4:6]  # Output: 80 channels
        self.down4 = self.encoder[6:8]  # Output: 224 channels
        self.down5 = self.encoder[8:]   # Output: 640 channels

        self.up1 = Up(3200, 256)
        self.up2 = Up(480, 128)
        self.up3 = Up(208, 64)
        self.up4 = Up(64 + 32, 32)
        self.up5 = OnlyUp(32, 32)

        self.out = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.doubleconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x)

        logits = self.out(x)
        return logits
    
def get_method(args):
    if args.model_name == "eub7":
        return UNetEfficientNetB7(n_classes= 1)
    if args.model_name == "eub7without2":
        return UNetEfficientNetB7_without_sc2(n_classes=1)