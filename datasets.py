import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, factor=4):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // factor, hr_height // factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)


class STLDataset(ImageDataset):
    def __init__(self, root, factor=4, train=True, download=True):
        super(STLDataset, self).__init__(root, (96, 96), factor)
        if download:
            # use torchvision to download
            _ = datasets.STL10(root, download=True)
        self.path = os.path.join(root, 'stl10_binary')
        if train:
            self.data = np.concatenate((self.load_images(os.path.join(self.path, 'train_X.bin')), self.load_images(os.path.join(self.path, 'unlabeled_X.bin'))))
        else:
            self.data = self.load_images(os.path.join(self.path, 'test_X.bin'))

        self.PIL_Augmentation = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.ColorJitter(hue=.025, saturation=.15),
                                                    transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.data)

    def load_images(self, path):
        # from stl10_input.py
        with open(path, 'rb') as f:
            return np.fromfile(f, dtype=np.uint8).reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1))

    def __getitem__(self, item):
        img = self.PIL_Augmentation(self.data[item])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}
