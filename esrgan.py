
import argparse
import os
import numpy as np
import math
import itertools
import sys
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from evaluation.eval import calculate_metrics

import torch.nn as nn
import torch.nn.functional as F
import torch


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_path", type=str, default="../data/train.h5", help="path to the dataset")
    parser.add_argument("--dataset_type", choices=['h5', 'txt'], default="txt", help="how is the dataset saved")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=100, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=200, help="high res. image width")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=0.0015, help="adversarial loss weight")
    parser.add_argument("--lambda_lr", type=float, default=0.05, help="pixel-wise loss weight for the low resolution L1 pixel loss")
    parser.add_argument("--lambda_hist", type=float, default=0.01, help="energy distribution loss weight")
    parser.add_argument("--batchwise_hist", type=str_to_bool, default=True, help="whether to use all images in a batch to calculate the energy distribution")
    parser.add_argument("--bins", type=int, default=10, help="number of bins in the energy distribution histogram")
    parser.add_argument("--root", type=str, default='', help="root directory for the model")
    parser.add_argument("--name", type=str, default=None, help='name of the model')
    parser.add_argument("--load_checkpoint", type=str, default=None, help='path to the generator weights to start the training with')
    parser.add_argument("--report_freq", type=int, default=10, help='report frequency determines how often the loss is printed')
    parser.add_argument("--model_path", type=str, default="saved_models", help="directory where the model is saved/should be saved")
    parser.add_argument("--discriminator", choices=['patch', 'standard'], default='patch', help="discriminator model to use")
    parser.add_argument("--relativistic", type=str_to_bool, default=True, help="whether to use relativistic average GAN")
    parser.add_argument("--save", type=str_to_bool, default=True, help="whether to save the model weights or not")
    parser.add_argument("--validation_path", type=str, default=None, help="Path to validation data. Validating when creating a new checkpoint")
    # number of batches to train from instead of number of epochs.
    # If specified the training will be interrupted after N_BATCHES of training.
    parser.add_argument("--n_batches", type=int, default=-1, help="number of batches of training")
    parser.add_argument("--n_checkpoints", default=-1, type=int, help="number of checkpoints during training (if used dominates checkpoint_interval)")
    opt = parser.parse_args()
    print(opt)
    return opt


def train(opt):
    model_name = '' if opt.name is None else (opt.name + '_')
    start_epoch = 0
    # create dictionary containing model infomation
    info = {'epochs': start_epoch}
    info_path = os.path.join(opt.model_path, model_name+'info.json')
    try:
        opt_dict = opt._asdict()
    except AttributeError:
        opt_dict = vars(opt)
    for key in ['name', 'residual_blocks', 'lr', 'b1', 'b2', 'dataset_path', 'lambda_lr', 'lambda_adv', 'discriminator', 'relativistic']:
        info[key] = opt_dict[key]

    os.makedirs(os.path.join(opt.root, opt.model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    if opt.discriminator == 'patch':
        discriminator = Markovian_Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    elif opt.discriminator == 'standard':
        discriminator = Standard_Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)

    # Losses
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = nn.L1Loss().to(device)
    criterion_hist = nn.MSELoss().to(device)

    if opt.load_checkpoint:
        # Load pretrained models
        generator.load_state_dict(torch.load(opt.load_checkpoint))
        generator_file = os.path.basename(opt.load_checkpoint)
        discriminator.load_state_dict(torch.load(opt.load_checkpoint.replace(generator_file, generator_file.replace('generator', 'discriminator'))))
        # extract model name if no name specified
        if model_name == '':
            model_name = generator_file.split('generator')[0]
            info_path = os.path.join(opt.model_path, model_name+'info.json')
        try:
            with open(info_path, 'r') as info_file:
                info = json.load(info_file)
        except FileNotFoundError:
            pass

    if opt.sample_interval != -1:
        image_dir = os.path.join(opt.root, "images/%straining" % model_name)
        os.makedirs(image_dir, exist_ok=True)

    checkpoint_interval = opt.checkpoint_interval
    if opt.n_checkpoints != -1:
        checkpoint_interval = np.inf

    # if we don't use this setting, need to set to inf
    n_batches = opt.n_batches
    if n_batches == -1:
        n_batches = np.inf

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    if opt.dataset_type == 'h5':
        dataset = JetDataset(opt.dataset_path, etaBins=opt.hr_height, phiBins=opt.hr_width)
    elif opt.dataset_type == 'txt':
        dataset = JetDatasetText(opt.dataset_path, etaBins=opt.hr_height, phiBins=opt.hr_width)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True
    )

    eps = 1e-10
    pool = SumPool2d().to(device)
    # ----------
    #  Training
    # ----------
    total_batches = len(dataloader)*(opt.n_epochs - start_epoch) if n_batches == np.inf else n_batches
    batches_done = -1
    for epoch in range(start_epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            batches_done += 1
            #batches_done = epoch * len(dataloader) + i
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                if batches_done % opt.report_freq == 0:
                    print(
                        "[Batch %d/%d] [Epoch %d/%d] [G pixel: %f]"
                        % (i, total_batches, epoch, opt.n_epochs,  loss_pixel.item())
                    )
                continue

            # Measure pixel-wise loss against ground truth for downsampled images
            loss_lr_pixel = criterion_pixel(pool(gen_hr), imgs_lr)

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            if opt.relativistic:
                # Adversarial loss (relativistic average GAN)
                loss_GAN = criterion_GAN(eps + pred_fake - pred_real.mean(0, keepdim=True), valid)
            else:
                loss_GAN = criterion_GAN(eps + pred_fake, valid)

            # calculate the energy distribution loss
            # first calculate the both histograms
            gen_nnz=gen_hr[gen_hr > 0]
            real_nnz=imgs_hr[imgs_hr > 0]
            e_min = torch.min(torch.cat((gen_nnz, real_nnz), 0)).item()
            e_max = torch.max(torch.cat((gen_nnz, real_nnz), 0)).item()
            histogram = SoftHistogram(opt.bins, e_min, e_max, batchwise=opt.batchwise_hist).to(device)
            gen_hist = histogram(gen_nnz)
            real_hist = histogram(real_nnz)
            loss_hist = criterion_hist(gen_hist, real_hist).mean(0)
            
            # Total generator loss
            loss_G = loss_pixel + opt.lambda_adv * loss_GAN + opt.lambda_lr * loss_lr_pixel + opt.lambda_hist * loss_hist

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())
            if opt.relativistic:
                # Adversarial loss for real and fake images (relativistic average GAN)
                loss_real = criterion_GAN(eps + pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = criterion_GAN(eps + pred_fake - pred_real.mean(0, keepdim=True), fake)
            else:
                loss_real = criterion_GAN(eps + pred_real, valid)
                loss_fake = criterion_GAN(eps + pred_fake, fake)
            #print(pred_fake[0].item(),pred_fake.mean(0, keepdim=True)[0].item(),loss_fake.item(),pred_real[0].item(),loss_real.item(),pred_real.mean(0, keepdim=True)[0].item())
            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            if batches_done % opt.report_freq == 0:
                print(
                    "[Batch %d/%d] [Epoch %d/%d] [D loss: %e] [G loss: %f, adv: %f, pixel: %f, lr pixel: %f, hist: %f]"
                    % (
                        i,
                        total_batches,
                        epoch,
                        opt.n_epochs,
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                        loss_lr_pixel.item(),
                        loss_hist.item(),
                    )
                )
            # check if loss is NaN
            if any(l != l for l in [loss_D.item(), loss_G.item(), loss_GAN.item(), loss_pixel.item()]):
                raise ValueError('loss is NaN')
            if batches_done % opt.sample_interval == 0 and not opt.sample_interval == -1:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                save_image(img_grid, os.path.join(opt.root, image_dir, "%d.png" % batches_done), nrow=1, normalize=False)

            if (checkpoint_interval != np.inf and (batches_done+1) % checkpoint_interval == 0) or (
                    checkpoint_interval == np.inf and (batches_done+1) % (total_batches//opt.n_checkpoints) == 0):
                if opt.save:
                    # Save model checkpoints
                    torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%d.pth" % (model_name, epoch)))
                    torch.save(discriminator.state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator_%d.pth" % (model_name, epoch)))
                    print('Saved model to %s' % opt.model_path)
                if opt.validation_path:
                    print('Validation')
                    output_path = opt.output_path if 'output_path' in dir(opt) else None
                    val_results = calculate_metrics(opt.validation_path, generator, device, output_path, opt.batch_size, opt.n_cpu)
                    val_results['epoch'] = epoch
                    val_results['batch'] = batches_done
                    generator.train()
                    try:
                        info['validation'].append(val_results)
                    except KeyError:
                        info['validation'] = [val_results]

            if batches_done == total_batches:
                if opt.validation_path:
                    return info['validation']
                else:
                    return
        info['epochs'] += 1
        if opt.save:
            with open(info_path, 'w') as outfile:
                json.dump(info, outfile)


if __name__ == "__main__":
    print('pytorch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name('cuda'))
    print('cuda version:', torch.version.cuda)
    print('cudnn version:', torch.backends.cudnn.version())

    opt = get_parser()
    train(opt)
