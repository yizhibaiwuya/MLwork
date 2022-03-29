import logging.config
import os
from typing import List

import PIL
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger()


class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None):
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)
        
        #img_path = os.path.join("dataset", self.dataset, img_name)
        img_path = os.path.join("/data/yxs", self.dataset, img_name)
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        sample["image_name"] = img_name
        return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    assert dataset in [
        "CIFAR10",
        "CIFAR100",
    ]
    mean = {
        "CIFAR10": (0.4914, 0.4822, 0.4465),
        "CIFAR100": (0.5071, 0.4867, 0.4408),
    }

    std = {
        "CIFAR10": (0.2023, 0.1994, 0.2010),
        "CIFAR100": (0.2675, 0.2565, 0.2761),
    }

    classes = {
        "CIFAR10": 10,
        "CIFAR100": 100,
    }

    in_channels = {
        "CIFAR10": 3,
        "CIFAR100": 3,
    }

    inp_size = {
        "CIFAR10": 32,
        "CIFAR100": 32,
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )
