import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import numpy as np
from itertools import product


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2, power=1):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        self.thres = 0
        self.power = power

    def forward(self, x):
        # x = F.pad(x, (1, 1, 0, 0), mode='circular')  # phi padding
        if not self.training:
            x = x**self.power
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        if self.training:
            return out
        else:
            return F.hardshrink(F.relu(out)**(1/self.power), lambd=self.thres)


def discriminator_block(in_filters, out_filters, first_block=False):
    layers = []
    layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
    if not first_block:
        layers.append(nn.BatchNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
    layers.append(nn.BatchNorm2d(out_filters))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Markovian_Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Markovian_Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int((1 if in_height % 16 != 0 else 0)+in_height / 2 ** 4), int((1 if in_width % 16 != 0 else 0)+in_width / 2 ** 4)

        self.output_shape = (1, patch_h, patch_w)

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class Standard_Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Standard_Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        self.output_shape = (1,)
        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        self.conv = nn.Sequential(*layers)
        # fully connected layers
        self.fc = nn.Sequential(nn.Linear(512*int(1+in_height / 2 ** 4)*int(1+in_width / 2 ** 4), 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, img):
        return self.fc(self.conv(img).view(img.shape[0], -1))


class DiffableHistogram(nn.Module):
    '''Modified version of https://discuss.pytorch.org/t/differentiable-torch-histc/25865/2 by Tony-Y
    If `bins` is a sequence the histogram will be defined by the edges specified in `bins`. If it is an integer 
    the histogram will consist of equally sized bins.
    `sigma` is a parameter of how strickly the differentiable histogram will approach the discrete histogram. For big values the gradients may explode.
    `batchwise` is a boolean that indicates whether the histogram will be taken over the whole data or batchwise.

    The input data should be in the shape of [batch_size, channels, height, width]
    The output data will be in the shape of [batch_size, -1] if `batchwise` is True else [1, -1]
    '''

    def __init__(self, bins, min=0, max=1, sigma=25, batchwise=False):
        super(DiffableHistogram, self).__init__()
        self.bins = torch.Tensor(bins)

        self.sigma = sigma
        self.batchwise = batchwise
        if type(bins) is int:
            self.delta = (max-min)/bins*torch.ones(bins-1)
            self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        else:
            self.delta = torch.Tensor([np.diff(bins)])
            self.centers = self.bins[:-1]+.5*self.delta

    def forward(self, x):
        batches = len(x) if (len(x.shape) == 4 and self.batchwise) else 1
        x = x.view(batches, -1)
        x = x[:, None, :] - self.centers[..., None]
        x = torch.sigmoid(self.sigma * (x + self.delta[..., None]/2)) - torch.sigmoid(self.sigma * (x - self.delta[..., None]/2))
        return x.sum(2)

    def to(self, device):
        self.centers = self.centers.to(device)
        self.delta = self.delta.to(device)
        return self


def normal(x, sig=1, mu=0):
    return np.exp(-.5*((x-mu)/sig)**2)/np.sqrt(2*np.pi*sig*sig)


def naive_upsample(img):
    '''A function that can upsample a batch of 1 channel images in a naive way'''
    up = torch.zeros(len(img), 1, *tuple(np.array(img.shape[-2:])*2))
    for batch in range(len(img)):
        for i, j in product(range(img.shape[-2]), range(img.shape[-1])):
            if img[batch, 0, i, j] == 0:
                continue
            rand = img[batch, 0, i, j]*normal(torch.rand(4))
            new_vals = torch.zeros(1)
            while new_vals.sum() <= 0:
                new_vals = torch.where(torch.randint(0, 2, (4,)) == 1, torch.zeros(4), rand)
            for k, (a, b) in enumerate(product(torch.arange(2)+2*i, torch.arange(2)+2*j)):
                up[batch, 0, a, b] = new_vals[k]/new_vals.sum()*img[batch, 0, i, j]
    return up


class NaiveGenerator(nn.Module):
    def __init__(self, num_upsample):
        super(NaiveGenerator, self).__init__()
        self.num_upsample = num_upsample

    def forward(self, x):
        for _ in range(self.num_upsample):
            x = naive_upsample(x)
        return x
