#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

from torchvision import transforms
import utils
from PIL import Image

IMAGE_SIZE = 256



__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']

class MyDataset(Dataset):

    def __init__(self,
                 input_path) \
            -> None:

        super().__init__()
        self.data = []
        self.file_ids = []
        self.sequences=[]
        self.song=[]

        self.compose_transform = transforms.Compose([transforms.ToTensor()])
        self.dataset_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),  # scale shortest side to image_size
            transforms.CenterCrop(IMAGE_SIZE),  # crop center image_size out
            transforms.ToTensor(),  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()  # normalize with ImageNet values
        ])

        img_pairs = [('1', '3'), ('1', '2'), ('1', '4'), ('1', '6'), ('2', '3'), ('2', '5'), ('2', '6'), ('3', '4'),
                     ('3', '5'), ('4', '6'), ('4', '5'), ('5', '6')]

        folders = []
        [folders.append(os.path.split(name)[1]) for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path,name))]
        print(f'len folders : {len(folders)}')
        for i, folder in enumerate(folders):
            if i > 300:
                break
            for pair in img_pairs:
                x, file_id1 = self._load_file(os.path.join(input_path, folder, f'{pair[0]}.png'))
                y, file_id2 = self._load_file(os.path.join(input_path, folder, f'{pair[1]}.png'))
                self.file_ids.append((file_id1, file_id2))
                self.data.append((x,y))
        self.data = np.array(self.data)
        self.file_ids = np.array(self.file_ids)

    def _load_file(self, file_path: Path):

        file_name = os.path.split(file_path)[1].split('.')[0]
        im = Image.open(file_path)
        transformed_im = self.dataset_transform(im)
        return (transformed_im, int(file_name))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        x = self.data[item][0]
        y=self.data[item][1]
        x_id=self.file_ids[item][0]
        y_id=self.file_ids[item][1]

        return (x, x_id, y, y_id)

# EOF