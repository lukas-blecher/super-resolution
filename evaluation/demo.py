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
from utils import *

manual_image = []

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
    generator = GeneratorRRDB(1, filters=64, num_res_blocks=args.residual_blocks, num_upsample=int(np.log2(args.factor)), power=args.scaling_power, res_scale=args.res_scale, use_transposed_conv=args.use_transposed_conv, fully_tconv_upsample=args.fully_transposed_conv, num_final_layer_res=args.num_final_res_blocks).to(device).eval()
    generator.thres = args.threshold
    generator.load_state_dict(torch.load(args.model, map_location=device))
    criterion = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    sumpool = SumPool2d(args.factor)

    if args.manual_image is not None:
        global manual_image
        for ii in manual_image:
            print("Proccessing image at location: " + str(ii))
            truth_imgs = dataset.__getitem__(ii)
            truth_hr = truth_imgs["hr"].numpy()
            truth_lr = truth_imgs["lr"].unsqueeze(0)
            gen_hr = generator(truth_lr).detach()
            gen_lr = sumpool(gen_hr).squeeze(0).numpy()
            gen_hr = gen_hr.squeeze(0).numpy()
            truth_lr = truth_lr.squeeze(0).numpy()
            print(truth_hr.shape, truth_lr.shape, gen_hr.shape, gen_lr.shape)
            np.save(args.output+"image_"+str(ii)+"_truth_hr.npy", truth_hr)
            np.save(args.output+"image_"+str(ii)+"_truth_lr.npy", truth_lr)
            np.save(args.output+"image_"+str(ii)+"_gen_hr.npy", gen_hr)
            np.save(args.output+"image_"+str(ii)+"_gen_lr.npy", gen_lr)
        return

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
    generator = GeneratorRRDB(1, filters=64, num_res_blocks=args.residual_blocks, num_upsample=int(np.log2(args.factor)), power=args.scaling_power, res_scale=args.res_scale, use_transposed_conv=args.use_transposed_conv, fully_tconv_upsample=args.fully_transposed_conv, num_final_layer_res=args.num_final_res_blocks).to(device).eval()
    generator.thres = args.threshold
    generator.load_state_dict(torch.load(args.model, map_location=device))
    sumpool = SumPool2d(args.factor)

    empty_hr = torch.zeros([1,1,*args.hw])
    empty_lr = sumpool(empty_hr)
    noise = torch.abs(torch.randn(empty_hr.shape))
    noise = noise / (torch.max(noise).item())
    indices = np.random.choice(np.arange(noise.numpy().flatten().size), replace=False, size=int(noise.numpy().flatten().size)-150) #choose indices randomly
    noise[np.unravel_index(indices, noise.shape)] = 0 #and set them to zero
    noise_hr=empty_hr+noise
    noise_lr = sumpool(noise_hr) 

    empty_sr = generator(empty_lr).detach()
    noise_sr = generator(noise_lr).detach()
    print(empty_hr.shape,empty_lr.shape,empty_sr.shape) #delete
    nnz = len([val for val in empty_sr.numpy().squeeze().flatten() if val > args.threshold])
    noisennz = len([val for val in noise_sr.numpy().squeeze().flatten() if val > args.threshold])
    hrnoisennz = len([val for val in noise_hr.numpy().squeeze().flatten() if val > args.threshold])
    print("upsampled empty picture nnz: {}".format(nnz))
    print("upsampled soft noise picture nnz: {}".format(noisennz))
    print("hr soft noise picture nnz: {}".format(hrnoisennz))
    global colors, vmax
    plt.figure()
    plt.subplot(221)
    plt.title("empty hr image")
    plt.imshow(toArray(empty_hr).squeeze(), cmap='gray', vmax=vmax)
    plt.subplot(222)
    plt.title("empty sr image")
    plt.imshow(toArray(empty_sr).squeeze(), cmap='gray', vmax=vmax)
    plt.subplot(223)
    plt.title("soft noise hr image")
    plt.imshow(toArray(noise_hr).squeeze(), cmap='gray', vmax=vmax)
    plt.subplot(224)
    plt.title("soft noise sr image")
    plt.imshow(toArray(noise_sr).squeeze(), cmap='gray', vmax=vmax)
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
    parser.add_argument("-t", "--dataset_type", choices=["txt", "h5", "jet", "spjet","hrlrjet"], default="hrlrjet", help="what kind of dataset")
    parser.add_argument("--hw", type=int, nargs='+', default=[80, 80], help="height and width of the image")
    parser.add_argument("--no_shuffle", action="store_false", help="Don't shuffle the images")
    parser.add_argument("--no_colors", action="store_false", help="Don't use colors in the plot")
    parser.add_argument("--threshold", type=float, default=1e-4, help="threshold for pixel activation")
    parser.add_argument("--vmax", type=float, default=None, help="maximum value in the plots")
    parser.add_argument("--n_hardest", type=int, default=None, help="how many of the hardest constituents should be in the ground truth")
    parser.add_argument("--E_thres", type=float, default=None, help="Energy threshold for the ground truth and the generator")
    parser.add_argument("--res_scale", type=float, default=default_dict['res_scale'], help="residual scaling factor for the generator")
    parser.add_argument("--empty", action="store_true", help="upsample an empty image")
    parser.add_argument("--use_transposed_conv", type=str_to_bool, default=False, help="Whether to use transposed convolutions in upsampling")
    parser.add_argument("--fully_transposed_conv", type=str_to_bool, default=False, help="Whether to ONLY use transposed convolutions in upsampling")
    parser.add_argument("--num_final_res_blocks", type=int, default=0, help="Whether to add res blocks AFTER upsampling")
    parser.add_argument("--manual_image", type=int, nargs="+", help="which images to manually save as np arrays")


    args = parser.parse_args()
    global colors, vmax
    colors = args.no_colors
    if args.manual_image is not None:
        manual_image = args.manual_image
        print(manual_image)
    if not args.empty:
        main(args)
    else:
        upsample_empty(args)