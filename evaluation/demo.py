import sys
import os
# add home directory to pythonpath
if os.path.split(os.path.abspath('.'))[1] == 'super-resolution':
    sys.path.insert(0, '.')
else:
    sys.path.insert(0, '..')
from datasets import *
from evaluation.PSNR_SSIM_metric import calculate_ssim
from evaluation.PSNR_SSIM_metric import calculate_psnr
from torch.utils.data import DataLoader
from models import GeneratorRRDB
from options.default import default_dict
import argparse
from collections import namedtuple
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def toArray(x):
    if len(x.shape) == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()[..., 0]
    else:
        return x.permute(1, 2, 0).cpu().numpy()[..., 0]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = JetDataset(args.input)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=0
    )
    generator = GeneratorRRDB(1, filters=64, num_res_blocks=args.residual_blocks).to(device).eval()
    generator.load_state_dict(torch.load(args.model))
    criterion = torch.nn.L1Loss()
    sumpool = SumPool2d()
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"].to(device)

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach()
        
        with torch.no_grad():
            gen_lr = sumpool(gen_hr).detach()
            print("HR L1Loss: %.3e, LR L1Loss: %.3e" % (criterion(gen_hr, imgs_hr).item(), criterion(gen_lr, imgs_lr).item()))

        show(imgs_lr, imgs_hr, gen_hr, gen_lr)


def show(lr, hr, pred, pred_lr):
    plt.figure()
    M, N = 4, len(lr)
    counter = np.arange(1, 1+N*M).reshape(N, M)
    for i in range(N):
        subplot(N, M, counter[i, 0], toArray(lr[i]), 'low resolution input')
        subplot(N, M, counter[i, 1], toArray(hr[i]), 'high resolution ground truth')
        subplot(N, M, counter[i, 2], toArray(pred[i]), 'model prediction')
        subplot(N, M, counter[i, 3], toArray(pred_lr[i]), 'downsampled prediction')
    plt.show()


def subplot(N, M, num, img, title=''):
    plt.subplot(N, M, num)
    if num <= M:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$\\eta$')
    plt.xlabel('$\\phi$')
    plt.imshow(img, cmap='gray_r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to weights")
    parser.add_argument("-r", "--residual_blocks", type=int, default=10, help="Number of residual blocks")
    parser.add_argument("-o", "--output", type=str, default=None, help="Where to save the images")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Number of images to show at once")
    parser.add_argument("--no_shuffle", action="store_false",help="Don't shuffle the images")
    args = parser.parse_args()
    main(args)
