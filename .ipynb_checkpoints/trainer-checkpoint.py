from get_ds import Breast, WithAubumentations, get_dataset
import os
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from model import get_method
from torch.utils.data.dataset import random_split
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, jaccard_score
from torchvision import transforms
parameter_path = '/media/mountHDD2/chuyenmt/BrEaST/Eff-UNet_Breast_Tumor_Segmentation/Training Parameter'
data_path = '/media/mountHDD2/chuyenmt/BrEaST/BrEaST-Lesions_USG-images_and_masks'

 

def trainer(args):

    wandb.init(
        project="Eff-UNet",
        config=args,
    )
    # Data_transfered = WithAubumentations(root = data_path)

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),

    #     ]
    # )
    # target_transform = transforms.Compose(
    #     [
    #         transforms.Resize((256, 256)),
    #         transforms.ToTensor(),
    #     ]
    # )
    # Data_transfered = Breast(root = data_path,transform = transform, target_transform=target_transform)
    Data_transfered = get_dataset(args)

    train_ds, test_ds = random_split(Data_transfered, [0.8, 0.2])
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    print("Training Samples: {}".format(len(train_ds)))
    print("Testing Samples: {}".format(len(test_ds)))
    print(len(Data_transfered))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = 1)
    model = get_method(args).to(device)
    optimizer = Adam(params = model.parameters(), lr = 0.001)
    epochs = 100
    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_dl))
    loss_fn = nn.BCEWithLogitsLoss()
    
    old_loss = 1e26
    best_dct = None
    last_dst = None
    #Training
    for epoch in range(epochs):
        model.train()
        tr_total_loss = 0
        for train_img, train_mask in tqdm(train_dl):
            train_img = train_img.to(device)
            train_mask = train_mask.to(device)

            train_gen_mask = model(train_img)
            train_loss = loss_fn(train_gen_mask, train_mask)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            scheduler.step()

            tr_total_loss += train_loss.item()
            metrics = {
                "train/train_loss": train_loss,
                "train/epoch": epoch + 1,

            }
            wandb.log(metrics)
    #Validing
        model.eval()
        with torch.no_grad():
            va_total_loss = 0
            for valid_img, valid_mask in tqdm(test_dl):
                valid_img = valid_img.to(device)
                valid_mask = valid_mask.to(device)

                valid_gen_img = model(valid_img)
                valid_loss = loss_fn(valid_gen_img, valid_mask)

                va_total_loss += valid_loss.item()
 
                val_metrics = {
                    "val/val_loss": valid_loss,
                }
                wandb.log(val_metrics)
        
        mean_train_loss = tr_total_loss/len(train_dl)
        mean_valid_loss = va_total_loss/len(test_dl)
        mean_metrics = {
                "train/mean_train_loss": mean_train_loss,
                "train/epoch": epoch + 1,
                "val/mean_valid_loss": mean_valid_loss,
                "val/epoch": epoch + 1,
        }
        wandb.log(mean_metrics)
        
        if mean_valid_loss <= old_loss:
            old_loss = mean_valid_loss
            best_dct = model.state_dict()

        last_dct = model.state_dict()


        print(f"Epoch: {epoch + 1} - TrainLoss: {mean_train_loss} - ValidLoss: {mean_valid_loss}")

    torch.save(best_dct, os.path.join(parameter_path, args.model_name))

    #Testing
    def dice_coefficient(pred, target): # Hàm tính toán Dice coefficient
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    model.eval()
    test_total_loss = 0
    test_total_dice = 0
    test_total_iou = 0
    num_batches = len(test_dl)

    with torch.no_grad():
        for test_img, test_mask in tqdm(test_dl):
            test_img = test_img
            test_mask = test_mask

            test_gen_mask = model(test_img)
            test_loss = loss_fn(test_gen_mask, test_mask)

            test_total_loss += test_loss.cpu().item()
            test_total_dice += dice_coefficient(test_gen_mask, test_mask).cpu().item()

            # Chuyển đổi dự đoán thành nhị phân
            binary_test_gen_mask = (torch.sigmoid(test_gen_mask) > 0.5).float()

            # Ensure both arrays are binary by thresholding test_mask
            binary_test_mask = (test_mask > 0.5).float()

            test_total_iou += jaccard_score(binary_test_mask.cpu().numpy().flatten(),
                                        binary_test_gen_mask.cpu().numpy().flatten())

    mean_test_loss = test_total_loss / num_batches
    mean_test_dice = test_total_dice / num_batches
    mean_test_iou = test_total_iou / num_batches

    print(f"Test Loss: {mean_test_loss}")
    print(f"Test Dice Coefficient: {mean_test_dice}")
    print(f"Test IoU: {mean_test_iou}")

    test__metrics = {
        'test/test_loss': mean_test_loss,
        'test/test_dice': mean_test_dice,
        'test/iou': mean_test_iou,
    }
    wandb.log(test__metrics)

    wandb.finish()

     
 