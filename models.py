import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from itertools import product


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2, drop_rate=0):
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
        self.drop = drop_rate > 0
        if self.drop:
            self.drop1 = nn.Dropout2d(drop_rate)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        if self.drop:
            out = self.drop1(out)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2, drop_rate=0):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters, drop_rate=drop_rate), DenseResidualBlock(filters, drop_rate=drop_rate), DenseResidualBlock(filters, drop_rate=drop_rate)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    '''attempts: 1) add transposed convolutions 2) add some res blocks after upsampling'''
    def __init__(self, channels=1, filters=64, num_res_blocks=10, num_upsample=1, power=1, multiplier=1, drop_rate=0, res_scale=0.2, use_transposed_conv=False, fully_tconv_upsample=False, num_final_layer_res=0, uniform_init=False):
        super(GeneratorRRDB, self).__init__()
        self.num_final_layer_res = num_final_layer_res

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters, res_scale=res_scale, drop_rate=drop_rate) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        if use_transposed_conv:
            for ii in range(num_upsample):
                if ii % 2 == 0:
                    upsample_layers += [
                        nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(),
                        nn.PixelShuffle(upscale_factor=2),
                    ]
                else:
                    upsample_layers += [nn.ConvTranspose2d(filters, filters, kernel_size = 2, stride = 2, padding = 0), nn.LeakyReLU()]
        
        elif fully_tconv_upsample:
            for _ in range(num_upsample): 
                upsample_layers += [nn.ConvTranspose2d(filters, filters, kernel_size = 2, stride = 2, padding = 0), nn.LeakyReLU()]
        else:
            for _ in range(num_upsample):
                upsample_layers += [
                    nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                    nn.PixelShuffle(upscale_factor=2),
                ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        if num_final_layer_res > 0:
            self.res_blocks_final = nn.Sequential(*[ResidualInResidualDenseBlock(filters, res_scale=res_scale, drop_rate=drop_rate) for _ in range(num_final_layer_res)])

        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        self.thres = 0
        self.power = nn.Parameter(torch.Tensor([power]), False)
        self.multiplier = nn.Parameter(torch.Tensor([multiplier]), False)

        if uniform_init:
            self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def out(self, x, pow=torch.ones(1)):
        if self.training:
            return F.hardshrink(x, lambd=self.thres**pow.item())
        else:
            return F.hardshrink(F.relu(x), lambd=self.thres**pow.item())

    def forward(self, x):
        # x = F.pad(x, (1, 1, 0, 0), mode='circular')  # phi padding
        x = self.multiplier*(x**self.power)
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        if self.num_final_layer_res > 0:
            out3 = self.res_blocks_final(out)
            out = torch.add(out3, out)
        out = self.conv3(out)/self.multiplier
        self.srs = self.out(out, self.power)
        if self.power != 1:
            out = F.relu(out)**(1/self.power)
        return self.out(out)




def discriminator_block(in_filters, out_filters, stride=(1, 2)):
    layers = []
    layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride[0], padding=1))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=stride[1], padding=1))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Markovian_Discriminator(nn.Module):
    def __init__(self, input_shape, channels=[16, 32, 32, 64]):
        super(Markovian_Discriminator, self).__init__()
        self.channels = channels
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def stride2(x):
            return int(np.ceil(x/2))
        patch_h, patch_w = in_height, in_width

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate(self.channels):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
            patch_h = stride2(patch_h)
            patch_w = stride2(patch_w)

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.output_shape = (1, patch_h, patch_w)
        self.model = nn.Sequential(*layers)

    def forward(self, img, *args):
        return self.model(img)


class Standard_Discriminator(Markovian_Discriminator):
    def __init__(self, input_shape, channels):
        super(Standard_Discriminator, self).__init__(input_shape, channels)
        self.model = self.model[:-1]
        # fully connected layers
        self.fc = nn.Sequential(nn.Linear(self.channels[-1]*self.output_shape[-2]*self.output_shape[-1], 1024), nn.ReLU(), nn.Linear(1024, 1))
        self.output_shape = (1,)

    def forward(self, img, *args):
        return self.fc(self.model(img).view(img.shape[0], -1))


class Conditional_Discriminator(nn.Module):
    def __init__(self, input_shape, channels=[32, 64, 128, 256], num_upsample=3):
        super(Conditional_Discriminator, self).__init__()
        self.channels = channels
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def stride2(x):
            return int(np.ceil(x/2))
        patch_h, patch_w = in_height, in_width

        hrlayers, clayers, endlayers = [], [], []
        in_filters = in_channels
        for i, out_filters in enumerate(self.channels):
            if i < num_upsample:
                hrlayers.extend(discriminator_block(in_filters, out_filters))
                clayers.extend(discriminator_block(in_filters, out_filters, stride=(1, 1)))
            elif i == num_upsample:
                endlayers.extend(discriminator_block(in_filters*2, out_filters))
            else:
                endlayers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
            patch_h = stride2(patch_h)
            patch_w = stride2(patch_w)

        endlayers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        self.output_shape = (1, patch_h, patch_w)
        self.model_hr = nn.Sequential(*hrlayers)
        self.model_c = nn.Sequential(*clayers)
        self.endmodel = nn.Sequential(*endlayers)

    def forward(self, img, cond):
        img = self.model_hr(img)
        cond = self.model_c(cond)
        return self.endmodel(torch.cat([img, cond], 1))

class Wasserstein_Discriminator(nn.Module):
    def __init__(self, input_shape, channels=[32, 64, 128, 256], num_upsample=3):
        super(Wasserstein_Discriminator, self).__init__()
        self.channels = channels
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def stride2(x):
            return int(np.ceil(x/2))
        patch_h, patch_w = in_height, in_width

        hrlayers, clayers, endlayers = [], [], []
        in_filters = in_channels
        for i, out_filters in enumerate(self.channels):
            if i < num_upsample:
                hrlayers.extend(discriminator_block(in_filters, out_filters))
                clayers.extend(discriminator_block(in_filters, out_filters, stride=(1, 1)))
            elif i == num_upsample:
                endlayers.extend(discriminator_block(in_filters*2, out_filters))
            else:
                endlayers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
            patch_h = stride2(patch_h)
            patch_w = stride2(patch_w)

        endlayers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        self.output_shape = (1, patch_h, patch_w)
        self.out_channels, self.out_height, self.out_width = self.output_shape
        self.model_hr = nn.Sequential(*hrlayers)
        self.model_c = nn.Sequential(*clayers)
        self.endmodel = nn.Sequential(*endlayers)
        self.fc = nn.Linear(self.out_channels*self.out_height*self.out_width, 1)

    def forward(self, img, cond):
        img = self.model_hr(img)
        cond = self.model_c(cond)
        combine = self.endmodel(torch.cat([img, cond], 1))
        out = self.fc(combine.view(-1, self.out_channels*self.out_height*self.out_width))
        return out

class Wasserstein_PatchDiscriminator(nn.Module):
    def __init__(self, input_shape, channels=[16, 32, 32, 64]):
        super(Wasserstein_PatchDiscriminator, self).__init__()
        self.channels = channels
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape

        def stride2(x):
            return int(np.ceil(x/2))
        patch_h, patch_w = in_height, in_width

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate(self.channels):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters
            patch_h = stride2(patch_h)
            patch_w = stride2(patch_w)

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.output_shape = (1, patch_h, patch_w)
        self.out_channels, self.out_height, self.out_width = self.output_shape
        self.model = nn.Sequential(*layers)

        self.fc = nn.Linear(self.out_channels*self.out_height*self.out_width, 1)

    def forward(self, img, *args):
        out = self.model(img)
        out = self.fc(out.view(-1, self.out_channels*self.out_height*self.out_width))
        return out

class SumPool2d(torch.nn.Module):
    def __init__(self, k=4, stride=None):
        '''Applies a 2D sum pooling over an input signal composed of several input planes'''
        super(SumPool2d, self).__init__()
        self.pool = torch.nn.AvgPool2d(k, stride=stride)
        self.kernel_size = k*k

    def forward(self, x):
        return self.kernel_size*self.pool(x)


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
            self.centers = float(min) + self.delta * (torch.arange(bins-1).float() + 0.5)
        else:
            self.delta = torch.Tensor([np.diff(bins)]).float()
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

def weight_reset(m):
    """reset the weights of a model"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def uniform_reset(m):
    """resets by initializing uniformly"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)