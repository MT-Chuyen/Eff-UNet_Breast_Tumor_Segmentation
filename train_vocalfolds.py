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
from torch import Tensor

def trainer_vocalfolds(args):

    wandb.init(
        project="Eff-UNet",
        config=args,
    )
 
    Data_transfered = get_dataset(args)

    train_ds, test_ds = random_split(Data_transfered, [0.8, 0.2])
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    print("Training Samples: {}".format(len(train_ds)))
    print("Testing Samples: {}".format(len(test_ds)))
    print(len(Data_transfered))
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = 1)
    # device = torch.device("cpu")
 

    from torchmetrics.segmentation import GeneralizedDiceScore
    gds = GeneralizedDiceScore(num_classes=7).to(device)

    from torchmetrics.segmentation import MeanIoU
    miou = MeanIoU(num_classes=7).to(device)
    model = get_method(args).to(device)


    optimizer = Adam(params = model.parameters(), lr = 0.001)

    epochs = 5

    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_dl))

    loss_fn = nn.CrossEntropyLoss()
 
    
    old_loss = 1e26
    best_dct = None
    last_dst = None
    #Training
    for epoch in range(epochs):
        model.train()
        tr_total_loss = 0

        gds_score_total = 0
        miou_score_total = 0  # Initialize the total mIoU score

        # Training loop
        for train_img, train_mask in tqdm(train_dl):
            train_img = train_img.to(device)
            train_mask = train_mask.to(device)

            train_gen_mask = model(train_img)
            train_loss = loss_fn(train_gen_mask, train_mask)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            tr_total_loss += train_loss.cpu().item()

            # Ensure the predicted mask has the correct shape
            train_gen_mask_arg = torch.argmax(train_gen_mask, dim=1)
            train_mask_arg = torch.argmax(train_mask, dim=1)

            # One-hot encode the masks for gds calculation
            train_gen_mask_onehot = F.one_hot(train_gen_mask_arg, num_classes=7).permute(0, 3, 1, 2)
            train_mask_onehot = F.one_hot(train_mask_arg, num_classes=7).permute(0, 3, 1, 2)

            gds_score = gds(train_gen_mask_onehot, train_mask_onehot)
            gds_score_total += gds_score.cpu().item()

            # Compute mIoU using one-hot encoded tensors
            miou_score = miou(train_gen_mask_onehot, train_mask_onehot)
            miou_score_total += miou_score.cpu().item()

        mean_train_loss = tr_total_loss / len(train_dl)
        mean_miou_score = miou_score_total / len(train_dl)  # Compute average mIoU

        # Validation
        model.eval()
        val_total_loss = 0
        val_gds_score_total = 0
        val_miou_score_total = 0

        with torch.no_grad():
            for val_img, val_mask in tqdm(test_dl):
                val_img = val_img.to(device)
                val_mask = val_mask.to(device)

                val_gen_mask = model(val_img)
                val_loss = loss_fn(val_gen_mask, val_mask)
                val_total_loss += val_loss.cpu().item()

                val_gen_mask_arg = torch.argmax(val_gen_mask, dim=1)
                val_mask_arg = torch.argmax(val_mask, dim=1)

                # One-hot encode the masks for gds calculation
                val_gen_mask_onehot = F.one_hot(val_gen_mask_arg, num_classes=7).permute(0, 3, 1, 2)
                val_mask_onehot = F.one_hot(val_mask_arg, num_classes=7).permute(0, 3, 1, 2)

                val_gds_score = gds(val_gen_mask_onehot, val_mask_onehot)
                val_gds_score_total += val_gds_score.cpu().item()

                # Compute mIoU using one-hot encoded tensors
                val_miou_score = miou(val_gen_mask_onehot, val_mask_onehot)
                val_miou_score_total += val_miou_score.cpu().item()

        mean_val_loss = val_total_loss / len(test_dl)
        mean_val_miou_score = val_miou_score_total / len(test_dl)  # Compute average mIoU for validation

        print(f"Epoch: {epoch+1} - TrainLoss: {mean_train_loss} - TrainGeneralDicescore: {gds_score_total/len(train_dl)} - TrainMeanIoU: {mean_miou_score}")
        print(f"Epoch: {epoch+1} - ValLoss: {mean_val_loss} - ValGeneralDicescore: {val_gds_score_total/len(test_dl)} - ValMeanIoU: {mean_val_miou_score}")
        mean_metrics = {
                "train/mean_train_loss": mean_train_loss,
                "train/gds_score_total": gds_score_total/len(train_dl),
                "train/mean_miou_score": mean_miou_score,
                "train/epoch": epoch + 1,
                "val/mean_valid_loss": mean_val_loss,
                "val/val_gds_score_total": val_gds_score_total/len(test_dl),
                "val/mean_val_miou_score": mean_val_miou_score,
                "val/epoch": epoch + 1,
        }
        
        # Update the best model if needed
        if mean_val_loss < old_loss:
            old_loss = mean_val_loss
            best_dct = model.state_dict()
        wandb.log(mean_metrics)

    model.load_state_dict(best_dct)

    wandb.finish()

     
 