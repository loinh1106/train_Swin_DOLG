import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath.replace("$",""))[:,:,::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.mode in ['train', 'val']:
            return torch.tensor(image), torch.tensor(row.id_encode)
        else:
            return torch.tensor(image)


def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        # albumentations.HorizontalFlip(p=0.5),
        # albumentations.ImageCompression(quality_lower=99, quality_upper=100),
        # albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.Resize(image_size, image_size),
        # albumentations.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val


def get_df(trainCSVPath):
    split_data_dir = trainCSVPath.split("/")
    data_dir = '/'.join(trainCSVPath.split("/")[:-1])
    df_train = pd.read_csv(trainCSVPath)

    if 'filepath' not in df_train.columns:
        df_train['filepath'] = df_train.apply(lambda x: f'{data_dir}/{split_data_dir[-1][:-4]}/{x.id}s/{x.image_name}', axis = 1)

    class_id2idx = {class_id: idx for idx, class_id in enumerate(sorted(df_train['id'].unique()))}
    df_train['id_encode'] = df_train['id'].map(class_id2idx)

    out_dim = df_train.id_encode.nunique()

    return df_train, out_dim
