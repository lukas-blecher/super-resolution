
import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from options.default import default
from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

gpu = 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=default.epoch, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=default.n_epochs, help="number of epochs of training")
    parser.add_argument("--dataset_path", type=str, default=default.dataset_path, help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=default.batch_size, help="size of the batches")
    parser.add_argument("--factor", type=int, default=default.factor, help="upscaling factor")
    parser.add_argument("--lr", type=float, default=default.lr, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=default.b1, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=default.b2, help="adam: decay of second order momentum offirst order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=default.n_cpu, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=default.hr_height, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=default.hr_width, help="high res. image width")
    parser.add_argument("--channels", type=int, default=default.channels, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=default.sample_interval, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=default.checkpoint_interval, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=default.residual_blocks, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=default.warmup_batches, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=default.lambda_adv, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=default.lambda_pixel, help="pixel-wise loss weight")
    parser.add_argument("--lambda_cont", type=float, default=default.lambda_cont, help="content loss weight")
    parser.add_argument("--lambda_reg", type=float, default=0, help="gp loss weight")
    parser.add_argument("--root", type=str, default=default.root, help="root directory for the model")
    parser.add_argument("--name", type=str, default=default.name, help='name of the model')
    parser.add_argument("--report_freq", type=int, default=default.report_freq, help='report frequency determines how often the loss is printed')
    parser.add_argument("--model_path", type=str, default=default.model_path, help="where the model is saved/should be saved")
    parser.add_argument("--d_channels", type=int, nargs='+', default=[64, 128, 256, 512], help="number of channels for the discriminator")
    # number of batches to train from instead of number of epochs.
    # If specified the training will be interrupted after N_BATCHES of training.
    parser.add_argument("--n_batches", type=int, default=default.n_batches, help="number of batches of training")
    parser.add_argument("--n_checkpoints", default=default.n_checkpoints, type=int, help="number of checkpoints during training (if used dominates checkpoint_interval)")
    opt = parser.parse_args()
    print(opt)
    return opt


def get_gpu_index():
    try:
        os.system('qstat > q.txt')
        q = open('q.txt', 'r').read()
        ids = [x.split('.gpu02')[0] for x in q.split('\n')[2:-1]]
        os.system('qstat -f %s > q.txt' % ids[-1])
        f = open('q.txt', 'r').read()
    except:
        pass
    try:
        os.remove('q.txt')
    except:
        pass
    return int([x for x in f.split('\n') if 'exec_host' in x][0].split('/')[1])


def train(opt):

    os.makedirs(os.path.join(opt.root, opt.model_path), exist_ok=True)

    device = torch.device("cuda:%i" % gpu if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(np.log2(opt.factor))).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)
    model_name = '' if opt.name is None else (opt.name + '_')
    os.makedirs(os.path.join(opt.root, "images/%straining" % model_name), exist_ok=True)
    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(
            os.path.join(opt.root, opt.model_path, "%sgenerator_%d.pth" % (model_name, opt.epoch))))
        discriminator.load_state_dict(torch.load(
            os.path.join(opt.root, opt.model_path, "%sdiscriminator_%d.pth" % (model_name, opt.epoch))))

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

    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    dataloader = DataLoader(
        #ImageDataset(opt.dataset_path, hr_shape=hr_shape),
        STLDataset(opt.dataset_path, factor=opt.factor),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    # ----------
    #  Training
    # ----------
    total_batches = len(dataloader)*(opt.n_epochs - opt.epoch) if n_batches == np.inf else n_batches
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure model input
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"].to(device)
            batch_size = len(imgs_lr)
            # Adversarial ground truths
            valid = torch.ones(batch_size, *discriminator.output_shape).to(device)
            fake = torch.zeros(batch_size, *discriminator.output_shape).to(device)

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
                        "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                    )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = (criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid) +
                        criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake))/2

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = opt.lambda_cont * loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            if opt.lambda_reg > 0:
                # generate interpolation between real and fake data
                epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
                interpolation = epsilon*imgs_hr+(1-epsilon)*gen_hr.detach()
                interpolation.requires_grad = True
                pred_interpolation = discriminator(interpolation)
                gradients = torch.autograd.grad(outputs=pred_interpolation, inputs=interpolation, grad_outputs=valid,
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_reg/2
                loss_D += gradient_penalty
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            if batches_done % opt.report_freq == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_content.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                    )
                )

            if batches_done % opt.sample_interval == 0 and not opt.sample_interval == -1:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.factor)
                img_grid = denormalize(torch.cat((imgs_hr, imgs_lr, gen_hr), -1))
                save_image(img_grid, os.path.join(opt.root, "images/%straining/%d.png" % (model_name, batches_done)), nrow=1, normalize=False)

            if (checkpoint_interval != np.inf and batches_done % checkpoint_interval == 0) or (
                    checkpoint_interval == np.inf and (batches_done+1) % (total_batches//opt.n_checkpoints) == 0):
                number = epoch if n_batches == np.inf else (batches_done+1)//(total_batches//opt.n_checkpoints)
                # Save model checkpoints
                torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%d.pth" % (model_name, number)))
                torch.save(discriminator.state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator_%d.pth" % (model_name, number)))
                print('Saved model to %s' % opt.model_path)
            if batches_done == total_batches:
                return


if __name__ == "__main__":
    opt = get_parser()
    try:
        gpu = get_gpu_index()
        num_gpus = torch.cuda.device_count()
        if gpu >= num_gpus:
            gpu = np.random.randint(num_gpus)
    except Exception as e:
        print(e)

    train(opt)
