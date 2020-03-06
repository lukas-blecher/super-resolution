import sys
import os
# add home directory to pythonpath
if os.path.split(os.path.abspath('.'))[1] == 'super-resolution':
    sys.path.insert(0, '.')
else:
    sys.path.insert(0, '..')
import matplotlib.pyplot as plt
import matplotlib.colors as col
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from collections import namedtuple
import argparse
from options.default import default_dict
from models import GeneratorRRDB
from torch.utils.data import DataLoader
from datasets import *

def toArray(x):
    if len(x.shape) == 4:
        return x.permute(0, 2, 3, 1).cpu().numpy()[..., 0]
    else:
        return x.permute(1, 2, 0).cpu().numpy()[..., 0]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(args.dataset_type, args.input, *args.hw, args.factor, pre=args.pre_factor, threshold=args.E_thres, N=args.n_hardest)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=0
    )
    generator = GeneratorRRDB(1, filters=64, num_res_blocks=args.residual_blocks, num_upsample=int(np.log2(args.factor)), power=args.scaling_power, res_scale=args.res_scale).to(device).eval()
    generator.thres = args.threshold
    generator.load_state_dict(torch.load(args.model, map_location=device))
    criterion = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    sumpool = SumPool2d(args.factor)
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"].to(device)

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach()

        with torch.no_grad():
            gen_lr = sumpool(gen_hr).detach()
            gen_nnz = gen_hr[gen_hr > 0].view(-1)
            en_loss = 0
            if len(gen_nnz) > 0:
                real_nnz = imgs_hr[imgs_hr > 0].view(-1)
                e_min = torch.min(torch.cat((gen_nnz, real_nnz), 0)).item()
                e_max = torch.max(torch.cat((gen_nnz, real_nnz), 0)).item()
                gen_hist = torch.histc(gen_nnz, 10, min=e_min, max=e_max).float()
                real_hist = torch.histc(real_nnz, 10, min=e_min, max=e_max).float()
                en_loss = mse(gen_hist, real_hist)
            print("HR L1Loss: %.3e, LR L1Loss: %.3e, Energy distribution loss %.3e" % (criterion(gen_hr, imgs_hr).item(), criterion(gen_lr, imgs_lr).item(), en_loss))

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
    global colors, vmax
    plt.subplot(N, M, num)
    if num <= M:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('$\\eta$')
    plt.xlabel('$\\varphi$')
    if colors:
        plt.imshow(img, cmap='jet', norm=col.LogNorm(), vmax=vmax)
    else:
        plt.imshow(img, cmap='gray', vmax=vmax)


def upsample_empty(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = GeneratorRRDB(1, filters=64, num_res_blocks=args.residual_blocks, num_upsample=int(np.log2(args.factor)), power=args.scaling_power, res_scale=args.res_scale).to(device).eval()
    generator.thres = args.threshold
    generator.load_state_dict(torch.load(args.model, map_location=device))
    sumpool = SumPool2d(args.factor)

    empty_hr = torch.zeros([1,1,*args.hw])
    empty_lr = sumpool(empty_hr)
    empty_sr = generator(empty_lr).detach()
    nnz = len([val for val in empty_sr.numpy().squeeze().flatten() if val > args.threshold])
    print(nnz)
    global colors, vmax
    plt.figure()
    plt.title("upsampled empty image")
    if colors:
        plt.imshow(toArray(empty_sr).squeeze(), cmap='jet', norm=col.LogNorm(), vmax=vmax)
    else:
        plt.imshow(toArray(empty_sr).squeeze(), cmap='gray', vmax=vmax)
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input file")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to weights")
    parser.add_argument("-r", "--residual_blocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("-o", "--output", type=str, default=None, help="Where to save the images")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Number of images to show at once")
    parser.add_argument("-f", "--factor", type=int, default=2, help="factor to super resolve (multiple of 2)")
    parser.add_argument("-p", "--pre_factor", type=int, default=1, help="factor to downsample before giving data to the model")
    parser.add_argument("--scaling_power", type=float, default=1, help="input data is raised to this power")
    parser.add_argument("-t", "--dataset_type", choices=["txt", "h5", "jet", "spjet"], default="spjet", help="what kind of dataset")
    parser.add_argument("--hw", type=int, nargs='+', default=[80, 80], help="height and width of the image")
    parser.add_argument("--no_shuffle", action="store_false", help="Don't shuffle the images")
    parser.add_argument("--no_colors", action="store_false", help="Don't use colors in the plot")
    parser.add_argument("--threshold", type=float, default=1e-4, help="threshold for pixel activation")
    parser.add_argument("--vmax", type=float, default=None, help="maximum value in the plots")
    parser.add_argument("--n_hardest", type=int, default=None, help="how many of the hardest constituents should be in the ground truth")
    parser.add_argument("--E_thres", type=float, default=None, help="Energy threshold for the ground truth and the generator")
    parser.add_argument("--res_scale", type=float, default=default_dict['res_scale'], help="residual scaling factor for the generator")
    parser.add_argument("--empty", action="store_true", help="upsample an empty image")

    args = parser.parse_args()
    global colors, vmax
    vmax = args.vmax
    colors = args.no_colors
    if not args.empty:
        main(args)
    else:
        upsample_empty(args)