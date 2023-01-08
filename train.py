import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import UNET
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split
from dataset import Crack500
from loss_function import *
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 0
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODLE = True
TRAIN_IMG_DIR = "D:/pix2pixHD/dataset_train/train_images"
TRAIN_MASK_DIR = "D:/pix2pixHD/dataset_train/train_mask"
VAL_IMG_DIR = "D:/pix2pixHD/dataset_train/val_images"
VAL_MASK_DIR = "D:/pix2pixHD/dataset_train/val_mask"
MODEL = UNET(in_channels=3, out_channels=1).to(DEVICE)
 
def train_fn(loader, model, optimizer,  scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = Exponential_function_loss(predictions, targets)
        #print(loss)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_transform = val_transform

    model = MODEL
    #loss_fn = Exponential_function_loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders("D:/pix2pixHD/dataset/train/image/", "D:/pix2pixHD/dataset/train/label/",
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(
            torch.load("D:/pix2pixHD/dataset_train/my_checkpoint.pth.tar"),
            model,
            optimizer,
        )
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_fn(train_loader, model, optimizer, scaler)
        print("time :", (time.time() - start))

        # save model
        if SAVE_MODLE:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader,
            model,
            folder="D:/pix2pixHD/ACS/saved_images/",
            device=DEVICE,
        )


if __name__ == "__main__":
    main()