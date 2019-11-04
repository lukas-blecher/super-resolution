import glob
import random
import os
import numpy as np
import h5py
import pandas as pd
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
    if len(tensors.shape) < 4:
        for c in range(3):
            tensors[c, ...].mul_(std[c]).add_(mean[c])
    else:
        for c in range(3):
            tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
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
    def __init__(self, root, train=True, download=True):
        super(STLDataset, self).__init__(root, (96, 96))
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


def unpack_data_nfeaturemaps(data_x, etaBins=100, phiBins=200, phi_offset=0, nfeaturemaps=1):
    img = torch.zeros(data_x.shape[0], etaBins, phiBins, nfeaturemaps)
    for j in range(data_x.shape[0]):
        for i in range(1, data_x.shape[1], nfeaturemaps+2):
            temp = int(data_x[j, i])
            if temp != 0 or i == 0:
                eta = temp % etaBins
                phi = (temp-eta)//phiBins
                if phi_offset != 0:
                    phi = (phi + phi_offset) % phiBins
                for k in range(nfeaturemaps):
                    img[j, eta, phi, k] = data_x[j, i+1+k]
            else:
                break
    return img.float()


class SumPool2d(torch.nn.Module):
    def __init__(self, k=4, stride=None):
        '''Applies a 2D sum pooling over an input signal composed of several input planes'''
        super(SumPool2d, self).__init__()
        self.pool = torch.nn.AvgPool2d(k, stride=stride)
        self.kernel_size = k*k

    def forward(self, x):
        return self.kernel_size*self.pool(x)


class EventDataset(Dataset):
    def __init__(self, file, amount=None, etaBins=100, phiBins=200, factor=2):
        ''' file is a path to a h5 file containing the data'''
        super(EventDataset, self).__init__()
        self.phiBins = phiBins
        self.etaBins = etaBins
        with h5py.File(file, 'r') as f:
            # List all groups
            a_group_key = list(f.keys())[0]
            # Get the data
            self.data = torch.Tensor(f[a_group_key])
        if amount is not None:
            self.data = self.data[:amount]
        self.pool = SumPool2d(factor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = unpack_data_nfeaturemaps(self.data[item][None, ...], self.etaBins, self.phiBins).permute(0, 3, 1, 2)
        img_lr = self.pool(img)[0]
        img_hr = img[0].clone()
        return {"lr": img_lr, "hr": img_hr}


def extract(data, etaBins, phiBins, channels=1):
    reconstruction = torch.zeros((channels, etaBins, phiBins))
    for i in range(data.shape[1]):
        pos_index = data[0, i]
        phi = int(pos_index//etaBins)
        eta = int(pos_index % etaBins)
        reconstruction[0, eta, phi] = data[1, i]
    return reconstruction.float()


class EventDatasetText(Dataset):

    def __init__(self, path, amount=None, etaBins=100, phiBins=200, factor=2):
        super(EventDatasetText, self).__init__()
        self.phiBins = phiBins
        self.etaBins = etaBins
        if amount is not None:
            self.data = self.data[:amount]
        self.pool = SumPool2d(factor)
        self.files = sorted(glob.glob(os.path.join(path, '*.txt')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        data = np.loadtxt(self.files[item], unpack=True, delimiter=' ', usecols=(0, 2))
        img = extract(data, self.etaBins, self.phiBins)[None, ...]
        img_lr = self.pool(img)[0]
        img_hr = img[0].clone()
        return {"lr": img_lr, "hr": img_hr}


class JetDataset(Dataset):

    def __init__(self, path, amount=None, etaBins=40, phiBins=40, factor=2):
        super(JetDataset, self).__init__()
        self.phiBins = phiBins
        self.etaBins = etaBins
        if amount is not None:
            self.data = self.data[:amount]
        self.pool = SumPool2d(factor)
        self.df = pd.read_hdf(path,'table')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img = torch.FloatTensor(self.df.iloc[item]).view(1, 1, self.etaBins, self.phiBins)*70
        img_lr = self.pool(img)[0]
        img_hr = img[0].clone()
        return {"lr": img_lr, "hr": img_hr}


def get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor=2):
    if dataset_type == 'h5':
        return EventDataset(dataset_path, etaBins=hr_height, phiBins=hr_width, factor=factor)
    elif dataset_type == 'txt':
        return EventDatasetText(dataset_path, etaBins=hr_height, phiBins=hr_width, factor=factor)
    elif dataset_type == 'jet':
        return JetDataset(dataset_path, etaBins=hr_height, phiBins=hr_width, factor=factor)
