import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if __name__ == "__main__":
    from model import UNET
class Crack500(Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        print(len(image_list), len(mask_list))
        self.image_list = image_list[1:100]
        self.mask_list = mask_list[1:100]
        self.transform = transform
        #self.images = os.listdir(image_dir)
        #self.label = os.listdir(mask_dir)[1:100]  # 디렉토리 안 파일명들 리스트로 만듦

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        
        #     self.image_dir, self.images[index]
        # )  # 폴더 경로 + image 파일 명
        mask_path =  self.mask_list[index]
        #     self.mask_dir, self.label[index].replace(".jpg", ".png")
        # )  # mask는 파일명 .jpg가 아니라 _mask.gif
        image = np.array(Image.open(img_path).convert("RGB"))  # PIL쓰려면 numpy array 필요
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask는 Gray scale로 불러온다 "L" 사용

        # 마스크 이미지 라벨 설정 (0.0,255.0)
        mask[mask ==255] = 1.0
        # 마스크 흰색이면 1.0으로 correct label로 sigmoid activation에 입력
        # augmentation 안되있으면 argumentation 실행
        if self.transform is not None:  # None is Null
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        return image,mask
    


class Crack500_result(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        self.images = os.listdir(image_list)
        #self.label = os.listdir(mask_dir)  # 디렉토리 안 파일명들 리스트로 만듦

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, index):
        img_path = self.image_list[index]
  # 폴더 경로 + image 파일 명
        # print(img_path[27:])
        # name = img_path[27:]
        image = np.array(Image.open(img_path).convert("RGB"))  # PIL쓰려면 numpy array 필요
        if self.transform is not None:  # None is Null
            augmentation = self.transform(image=image)
            image = augmentation["image"]
        #print(image)
        return image
