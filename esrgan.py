
import argparse
import os
import numpy as np
import math
import itertools
import sys
import json
from collections import namedtuple
from sklearn.cluster import KMeans

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from options.default import default, default_dict
from evaluation.eval import calculate_metrics, distribution

import torch.nn as nn
import torch.nn.functional as F
import torch


def str_to_bool(value):
    if value is None:
        return None
    elif value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=default.factor, help="factor to upsample the input image")
    parser.add_argument("--pre_factor", type=int, default=default.pre_factor, help="factor to donwsample the input image before training")
    parser.add_argument("--n_epochs", type=int, default=default.n_epochs, help="number of epochs of training")
    parser.add_argument("--dataset_path", type=str, default=default.dataset_path, help="path to the dataset")
    parser.add_argument("--dataset_type", choices=['h5', 'txt', 'jet', 'spjet'], default=default.dataset_type, help="how is the dataset saved")
    parser.add_argument("--batch_size", type=int, default=default.batch_size, help="size of the batches")
    parser.add_argument("--lr", type=float, default=default.lr, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=default.b1, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=default.b2, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=default.n_cpu, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=default.hr_height, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=default.hr_width, help="high res. image width")
    parser.add_argument("--channels", type=int, default=default.channels, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=default.sample_interval, help="interval between saving image samples")
    parser.add_argument("--image_path", type=str, default=default.image_path, help="where to save the images during trianing")
    parser.add_argument("--checkpoint_interval", type=int, default=default.checkpoint_interval, help="batch interval between model checkpoints")
    parser.add_argument("--validation_interval", type=int, default=default.validation_interval, help="batch interval between validation samples")
    parser.add_argument("--evaluation_interval", type=int, default=default.evaluation_interval, help="batch interval between evaluations (histograms)")
    parser.add_argument("--residual_blocks", type=int, default=default.residual_blocks, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=default.warmup_batches, help="number of batches with pixel-wise loss only")
    parser.add_argument("--learn_warmup", type=str_to_bool, default=default.learn_warmup, help="whether to learn during warmup phase or not")
    parser.add_argument("--pixel_multiplier", type=float, default=default.pixel_multiplier, help="multiply the image by this factors")
    parser.add_argument("--lambda_adv", type=float, default=default.lambda_adv, help="adversarial loss weight")
    parser.add_argument("--lambda_lr", type=float, default=default.lambda_lr, help="pixel-wise loss weight for the low resolution L1 pixel loss")
    parser.add_argument("--lambda_hist", type=float, default=default.lambda_hist, help="energy distribution loss weight")
    parser.add_argument("--lambda_nnz", type=float, default=default.lambda_nnz, help="loss weight for amount of non zero pixels")
    parser.add_argument("--lambda_mask", type=float, default=default.lambda_mask, help="loss weight for hr mask")
    parser.add_argument("--scaling_power", type=float, nargs='+', default=default.scaling_power, help="to what power to raise the input image")
    parser.add_argument("--batchwise_hist", type=str_to_bool, default=default.batchwise_hist, help="whether to use all images in a batch to calculate the energy distribution")
    parser.add_argument("--sigma", type=float, default=default.sigma, help="Sigma parameter for the differentiable histogram")
    parser.add_argument("--bins", type=int, default=default.bins, help="number of bins in the energy distribution histogram")
    parser.add_argument("--root", type=str, default=default.root, help="root directory for the model")
    parser.add_argument("--name", type=str, default=default.name, help='name of the model')
    parser.add_argument("--load_checkpoint", type=str, default=default.load_checkpoint, help='path to the generator weights to start the training with')
    parser.add_argument("--report_freq", type=int, default=default.report_freq, help='report frequency determines how often the loss is printed')
    parser.add_argument("--model_path", type=str, default=default.model_path, help="directory where the model is saved/should be saved")
    parser.add_argument("--discriminator", choices=['patch', 'standard'], default=default.discriminator, help="discriminator model to use")
    parser.add_argument("--relativistic", type=str_to_bool, default=default.relativistic, help="whether to use relativistic average GAN")
    parser.add_argument("--save", type=str_to_bool, default=default.save, help="whether to save the model weights or not")
    parser.add_argument("--save_info", type=str_to_bool, default=default.save_info, help="whether to save the info.json file or not")
    parser.add_argument("--validation_path", type=str, default=default.validation_path, help="Path to validation data. Validating when creating a new checkpoint")
    parser.add_argument("--testset_path", type=str, default=default.testset_path, help="Path to the test set. Is used for the histograms during evaluation")
    # number of batches to train from instead of number of epochs.
    # If specified the training will be interrupted after N_BATCHES of training.
    parser.add_argument("--n_batches", type=int, default=default.n_batches, help="number of batches of training")
    parser.add_argument("--n_checkpoints", default=default.n_checkpoints, type=int, help="number of checkpoints during training (if used dominates checkpoint_interval)")
    parser.add_argument("--n_validations", type=int, default=default.n_validations, help="number of validation points during training (if used dominates validation_interval)")
    parser.add_argument("--n_evaluation", type=int, default=default.n_evaluation, help="number of histograms to compute during trianing")
    parser.add_argument("--default", type=str, default=default.default, help="Path to a json file. When this option is provided, all unspecified arguments will be taken from the json file")

    opt = parser.parse_args()
    if opt.default:
        given = vars(opt)
        with open(opt.default, 'r') as f:
            arguments = json.load(f)
        # reduce to only non default arguments
        given = {key: given[key] for key in given.keys() if default_dict[key] != given[key]}
        # add all arguments from opt.default
        arguments = {**given, **{key: arguments[key] for key in arguments if key not in given}}
        # add remaining default arguments if not all keys are specified
        arguments = {**arguments, **{key: default_dict[key] for key in default_dict if key not in arguments}}
        opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
    print(opt)
    return opt


def train(opt):
    model_name = '' if opt.name is None else (opt.name + '_')
    start_epoch = 0
    # check if enough warmup_batches are specified
    if opt.lambda_hist > 0:
        assert opt.warmup_batches > 0, "if distribution learning is enabled, warmup_batches needs to be greater than 0."
    # create dictionary containing model infomation
    info = {'epochs': start_epoch}
    info_path = os.path.join(opt.model_path, model_name+'info.json')
    try:
        opt_dict = opt._asdict()
    except AttributeError:
        opt_dict = vars(opt)
    for key in ['name', 'residual_blocks', 'factor', 'lr', 'b1', 'b2', 'dataset_path', 'dataset_type', 'lambda_lr', 'lambda_adv', 'lambda_hist', 'lambda_nnz', 'lambda_mask', 'discriminator', 'relativistic', 'warmup_batches', 'scaling_power']:
        info[key] = opt_dict[key]

    os.makedirs(os.path.join(opt.root, opt.model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = MultiGenerator(opt.scaling_power, opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(np.log2(opt.factor))).to(device)
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
        image_dir = os.path.join(opt.root, opt.image_path, "%straining" % model_name)
        os.makedirs(image_dir, exist_ok=True)

    checkpoint_interval = opt.checkpoint_interval
    if opt.n_checkpoints != -1:
        checkpoint_interval = np.inf

    validation_interval = opt.validation_interval
    if opt.n_validations != -1:
        validation_interval = np.inf

    evaluation_interval = opt.evaluation_interval
    if opt.n_evaluation != -1:
        evaluation_interval = np.inf

    # if we don't use this setting, need to set to inf
    n_batches = opt.n_batches
    if n_batches == -1:
        n_batches = np.inf

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # LR Scheduler
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, verbose=True, patience=5)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, verbose=False, patience=5)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    dataset = get_dataset(opt.dataset_type, opt.dataset_path, opt.hr_height, opt.hr_width, opt.factor, pre=opt.pre_factor, power=opt.scaling_power)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True
    )
    eps = 1e-10
    pool = SumPool2d(opt.factor).to(device)
    e_max = 50  # random initialization number for the maximal pixel value
    nnz = []  # list with nonzero values during the first few batches
    binedges = []  # list with bin edges for energy distribution training
    # ----------
    #  Training
    # ----------
    loss_dict = info['loss'] if 'loss' in info else {loss: [] for loss in ['d_loss', 'g_loss', 'adv_loss', 'pixel_loss', 'lr_loss', 'hist_loss', 'nnz_loss', 'mask_loss']}
    # if trainig is continued the batch number needs to be increased by the number of batches already trained on
    try:
        batches_trained = int(info['batches_done'])
    except KeyError:
        batches_trained = 0

    start_epoch = info['epochs']
    total_batches = len(dataloader)*(opt.n_epochs - start_epoch) if n_batches == np.inf else n_batches
    batches_done = batches_trained-1

    # function for saving info.json
    save_info_file = opt.save_info if opt.save_info is not None else opt.save

    def save_info():
        if save_info_file:
            with open(info_path, 'w') as outfile:
                info['loss'] = loss_dict
                info['batches_done'] = batches_done
                json.dump(info, outfile)

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
            # clip the powers to [.1, 1]
            generator.pow.constraint()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                if opt.learn_warmup:
                    loss_pixel.backward()
                    optimizer_G.step()
                    for gs in ['g_loss', 'pixel_loss']:
                        loss_dict[gs].append(loss_pixel.item())
                    if batches_done % opt.report_freq == 0:
                        print(
                            "[Batch %d/%d] [Epoch %d/%d] [G pixel: %f]"
                            % (i, total_batches, epoch, opt.n_epochs,  loss_pixel.item())
                        )
                if opt.lambda_hist > 0:
                    # find a good value to cut off the distribution
                    imgs_hr = imgs_hr.cpu().view(-1).numpy()
                    nnz.extend(list(imgs_hr[imgs_hr > 0]))
                    c, b = np.histogram(nnz, 100)
                    e_max = b[(np.cumsum(c) > len(nnz)*.9).argmax()]  # set e_max to the value where 90% of the data is smaller
                continue
            elif batches_done == opt.warmup_batches:
                if opt.lambda_hist > 0:
                    print("found e_max to be %.2f" % e_max)
                    sorted_nnz = np.sort(nnz)
                    sorted_nnz = sorted_nnz[sorted_nnz <= e_max]
                    k_mean = KMeans(n_clusters=opt.bins, random_state=0).fit(sorted_nnz.reshape(-1, 1))
                    binedges = np.sort(k_mean.cluster_centers_.flatten())
                    binedges = np.array([0, *(np.diff(binedges)/2+binedges[:-1]), e_max])
                    info['binedges'] = list(binedges)
                del nnz
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

            # Total generator loss
            loss_G = loss_pixel + opt.lambda_adv * loss_GAN + opt.lambda_lr * loss_lr_pixel
            loss_hist, loss_nnz, loss_mask = torch.Tensor([np.nan]), torch.Tensor([np.nan]), torch.Tensor([np.nan])

            if opt.lambda_hist > 0:
                # calculate the energy distribution loss
                # first calculate the both histograms
                gen_nnz = gen_hr[gen_hr > 0]
                real_nnz = imgs_hr[imgs_hr > 0]
                histogram = DiffableHistogram(binedges, sigma=opt.sigma, batchwise=opt.batchwise_hist).to(device)
                gen_hist = histogram(gen_nnz)
                real_hist = histogram(real_nnz)
                loss_hist = criterion_hist(gen_hist, real_hist)
                loss_G = loss_G + opt.lambda_hist * loss_hist
            if opt.lambda_nnz > 0:
                gen_nnz = softgreater(gen_hr, 0, 50000).sum(1).sum(1).sum(1)
                target = (imgs_hr > 0).sum(1).sum(1).sum(1).float().to(device)
                loss_nnz = criterion_hist(gen_nnz, target)
                loss_G = loss_G + opt.lambda_nnz * loss_nnz
            if opt.lambda_mask > 0:
                gen_mask = nnz_mask(gen_hr)
                real_mask = nnz_mask(imgs_hr)
                loss_mask = criterion_pixel(gen_mask, real_mask)
                loss_G = loss_G + opt.lambda_mask * loss_mask

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
            # save loss to dict

            if batches_done % opt.report_freq == 0:
                for v, l in zip(loss_dict.values(), [loss_D.item(), loss_G.item(), loss_GAN.item(), loss_pixel.item(), loss_lr_pixel.item(), loss_hist.item(), loss_nnz.item(), loss_mask.item()]):
                    v.append(l)
                print("[Batch %d] [D loss: %e] [G loss: %f, adv: %f, pixel: %f, lr pixel: %f, hist: %.1f, nnz: %f, mask: %f]"
                      % (batches_done, *[l[-1] for l in loss_dict.values()],))

            # check if loss is NaN
            if any(l != l for l in [loss_D.item(), loss_G.item()]):
                raise ValueError('loss is NaN')
            if batches_done % opt.sample_interval == 0 and not opt.sample_interval == -1:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.factor)
                img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
                save_image(img_grid, os.path.join(opt.root, image_dir, "%d.png" % batches_done), nrow=1, normalize=False)

            if (checkpoint_interval != np.inf and (batches_done+1) % checkpoint_interval == 0) or (
                    checkpoint_interval == np.inf and (batches_done+1) % (total_batches//opt.n_checkpoints) == 0):
                if opt.save:
                    # Save model checkpoints
                    torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%d.pth" % (model_name, epoch)))
                    torch.save(discriminator.state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator_%d.pth" % (model_name, epoch)))
                    print('Saved model to %s' % opt.model_path)
                    save_info()

            if ((validation_interval != np.inf and (batches_done+1) % validation_interval == 0) or (
                    validation_interval == np.inf and (batches_done+1) % (total_batches//opt.n_validations) == 0)) and opt.validation_path is not None:
                print('Validation')
                output_path = opt.output_path if 'output_path' in dir(opt) else None
                val_results = calculate_metrics(opt.validation_path, opt.dataset_type, generator, device, output_path, opt.batch_size,
                                                opt.n_cpu, opt.bins, opt.hr_height, opt.hr_width, opt.factor, pre=opt.pre_factor)
                val_results['epoch'] = epoch
                val_results['batch'] = batches_done
                # If necessary lower the learning rate
                try:
                    scheduler_G.step(val_results['metrics']['hr_l1']['mean'])
                    scheduler_D.step(val_results['metrics']['hr_l1']['mean'])
                except KeyError:
                    pass

                generator.train()
                try:
                    info['validation'].append(val_results)
                    # check if all metrics yield the same results and interrupt training if true. likley no changes in future
                    stop_training = True
                    for key in val_results.keys():
                        if key in ('epoch', 'batch') or not stop_training:
                            continue
                        # check the last few validation results
                        for i in range(np.clip(len(info['validation'])-4, 0, None), len(info['validation'])-1):
                            if info['validation'][i][key] != info['validation'][i+1][key]:
                                stop_training = False
                    if stop_training:
                        print('stopping training because validation results are exactly the same')
                        return info['validation']

                except KeyError:
                    info['validation'] = [val_results]

            if (evaluation_interval != np.inf and (batches_done+1) % evaluation_interval == 0) or (
                    evaluation_interval == np.inf and (batches_done+1) % (total_batches//opt.n_evaluation) == 0):
                distribution(opt.testset_path, opt.dataset_type, generator, device, os.path.join(image_dir, '%06d_hist.png' % batches_done),
                             30, 0, 30, opt.hr_height, opt.hr_width, opt.factor, 5000, pre=opt.pre_factor, mode=['max', 'nnz', 'meannnz'])
                generator.train()
            if batches_done == total_batches:
                save_info()
                if opt.validation_path:
                    return info['validation']
                else:
                    return
        info['epochs'] += 1
        save_info()


def softgreater(x, val, sigma=5000, delta=0):
    # differentiable verions of torch.where(x>val)
    return torch.sigmoid(sigma * (x-val+delta))


def nnz_mask(x, sigma=5e4):
    return torch.sigmoid(sigma*x)


if __name__ == "__main__":
    print('pytorch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name('cuda'))
    print('cuda version:', torch.version.cuda)
    print('cudnn version:', torch.backends.cudnn.version())

    opt = get_parser()
    train(opt)
