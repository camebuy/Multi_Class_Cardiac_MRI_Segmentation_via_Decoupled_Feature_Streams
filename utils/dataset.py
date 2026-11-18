"""Existing slice loading, augmentation and label mapping."""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_names(path, file_path):
    # Preserve historical split behavior for compatibility with existing experiments.
    with open(file_path, "r", encoding="UTF-8") as f:
        data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "masks", name) + ".png" for name in data]
    return images, masks


def load_data(path,val_name=None):
    train_names_path = f"{path}/train.txt"
    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val_{val_name}.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Reading Image """
        # 读取图片 (H, W, 3)
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        """ Reading Mask """
        # 读取 Mask (H, W) - 注意：必须以灰度模式读取
        # Stored grayscale codes are mapped below to BG=0, RV=1, Myo=2, LV=3.
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        """ Applying Data Augmentation """
        if self.transform is not None:
            # Albumentations 库通常要求 mask 也是 (H, W) 或者是 (H, W, 1)
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        """ Image Processing """
        # Resize 到指定大小
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        image = image / 255.0  # 归一化到 [0, 1]

        """ Mask Processing """
        # 1. Resize (使用最近邻插值，防止产生 1.5 这种不存在的类别)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask[(mask != 127) & (mask != 85) & (mask != 170) & (mask != 191) & (mask != 255)] = 0  # 背景 (Background)
        mask[(mask == 127) | (mask == 85)] = 1  # 右心室 (Right Ventricle - RV)
        mask[(mask == 191) | (mask == 170)] = 2  # 心肌 (Myocardium - Myo)
        mask[mask == 255] = 3  # 左心室 (Left Ventricle - LV)
        return torch.from_numpy(image).float(), torch.from_numpy(mask).long()

    def __len__(self):
        return self.n_samples
