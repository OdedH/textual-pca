from torchvision import datasets
import random
from typing import Any, Callable, Optional, Tuple, List
import torch
import os
import cv2
from PIL import Image
import json
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class ImageSetDataset(Dataset):
    """
    This dataset is for returning a set of images (and based on the dataset here)
    """

    def __init__(self, images_base_path, transform, set_size=1):
        self.images = [x for x in sorted(os.listdir(images_base_path)) if x.endswith(".jpg")]
        self.transform = transform
        self.images_base_path = images_base_path
        self.set_size = set_size

    def __len__(self):
        return len(self.images)//self.set_size

    def __getitem__(self, index):
        images_names = self.images[index*self.set_size:(index+1)*self.set_size]
        image_paths = [self.images_base_path + image_name for image_name in images_names]
        images = torch.cat([self.transform(Image.open(image_path)).unsqueeze(0) for image_path in image_paths])
        ids = images_names
        # data_dict = {'images': images, 'ids': ids}

        return images


class DatasetFromPath(Dataset):
    def __init__(self, root, transform):
        self.images = [x for x in sorted(os.listdir(root)) if x.endswith(".jpg") or x.endswith(".png") or x.endswith(".jpeg")]
        self.transform = transform
        self.images_base_path = root

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = self.images_base_path + image_name
        return  self.transform(Image.open(image_path))

class FlowersImageOnlyDataset(datasets.Flowers102):
    """
    A CelebA dataset containing only the image
    """

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image

class PetsImageOnlyDataset(datasets.OxfordIIITPet):
    """
    A CelebA dataset containing only the image
    """

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image
