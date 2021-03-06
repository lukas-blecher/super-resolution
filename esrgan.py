
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

from models import *
from datasets import *
from utils import *
from options.default import default, default_dict
from evaluation.eval import calculate_metrics, distribution

import torch.nn as nn
import torch.nn.functional as F
import torch

gpu = 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=default.factor, help="factor to upsample the input image")
    parser.add_argument("--pre_factor", type=int, default=default.pre_factor, help="factor to donwsample the input image before training")
    parser.add_argument("--n_epochs", type=int, default=default.n_epochs, help="number of epochs of training")
    parser.add_argument("--dataset_path", type=str, default=default.dataset_path, help="path to the dataset")
    parser.add_argument("--dataset_type", choices=['h5', 'txt', 'jet', 'spjet', 'hrlrjet'], default=default.dataset_type, help="how is the dataset saved")
    parser.add_argument("--batch_size", type=int, default=default.batch_size, help="size of the batches")
    parser.add_argument("--lr", type=float, default=default.lr, help="adam: learning rate")
    parser.add_argument("--lr_g", type=float, default=default.lr_g, help="adam: learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=default.lr_d, help="adam: learning rate for discriminator")
    parser.add_argument("--b1", type=float, default=default.b1, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=default.b2, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--l2decay", type=float, default=default.l2decay, help="adam: L2 regularization parameter")
    parser.add_argument("--n_cpu", type=int, default=default.n_cpu, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=default.hr_height, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=default.hr_width, help="high res. image width")
    parser.add_argument("--channels", type=int, default=default.channels, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=default.sample_interval, help="interval between saving image samples")
    parser.add_argument("--image_path", type=str, default=default.image_path, help="where to save the images during trianing")
    parser.add_argument("--checkpoint_interval", type=int, default=default.checkpoint_interval, help="batch interval between model checkpoints")
    parser.add_argument("--validation_interval", type=int, default=default.validation_interval, help="batch interval between validation samples")
    parser.add_argument("--evaluation_interval", type=int, default=default.evaluation_interval, help="batch interval between evaluations (histograms)")
    parser.add_argument("--update_d", type=int, default=default.update_d, help="every nth batch the discriminator will be updated")
    parser.add_argument("--update_g", type=int, default=default.update_g, help="every nth batch the generator will be updated")
    parser.add_argument("--residual_blocks", type=int, default=default.residual_blocks, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=default.warmup_batches, help="number of batches with pixel-wise loss only")
    parser.add_argument("--learn_warmup", type=str_to_bool, default=default.learn_warmup, help="whether to learn during warmup phase or not")
    parser.add_argument("--pixel_multiplier", type=float, default=default.pixel_multiplier, help="multiply the image by this factors")
    parser.add_argument("--lambda_pix", type=float, default=default.lambda_pix, help="loss weight for high resolution pixel difference")
    parser.add_argument("--lambda_hr", type=float, default=default.lambda_hr, help="loss weight for high resolution pixel difference")
    parser.add_argument("--lambda_adv", type=float, default=default.lambda_adv, help="adversarial loss weight")
    parser.add_argument("--lambda_lr", type=float, default=default.lambda_lr, help="pixel-wise loss weight for the low resolution L1 pixel loss")
    parser.add_argument("--lambda_hist", type=float, default=default.lambda_hist, help="energy distribution loss weight")
    parser.add_argument("--lambda_wasser", type=float, default=default.lambda_wasser, help="Wasserstein distance loss weight")
    parser.add_argument("--lambda_nnz", type=float, default=default.lambda_nnz, help="loss weight for amount of non zero pixels")
    parser.add_argument("--lambda_mask", type=float, default=default.lambda_mask, help="loss weight for hr mask")
    parser.add_argument("--lambda_pow", type=float, default=default.lambda_pow, help="loss weight for multiple hr L1 pixel loss")
    parser.add_argument("--lambda_hit", type=float, default=default.lambda_hit, help="loss weight for mean image loss")
    parser.add_argument("--scaling_power", type=float, default=default.scaling_power, help="to what power to raise the input image")
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
    parser.add_argument("--conditional", type=str_to_bool, default=default.conditional, help="whether to use conditional discriminator")
    parser.add_argument("--save", type=str_to_bool, default=default.save, help="whether to save the model weights or not")
    parser.add_argument("--save_info", type=str_to_bool, default=default.save_info, help="whether to save the info.json file or not")
    parser.add_argument("--validation_path", type=str, default=default.validation_path, help="Path to validation data. Validating when creating a new checkpoint")
    parser.add_argument("--testset_path", type=str, default=default.testset_path, help="Path to the test set. Is used for the histograms during evaluation")
    parser.add_argument("--plot_grad", type=str_to_bool, default=default.plot_grad, help="Whether to save the gradients for each layer to the IMAGE_PATH every REPORT_FREQ")
    parser.add_argument("--smart_save", type=str_to_bool, default=default.smart_save, help="If this option is used the model will only be saved if the evalidation result is better than before\
                                                                                            (when the best overlay for the histograms for ground truth and model prediction is found)")
    parser.add_argument("-N", type=int, default=default.N, help="Amount of images to check during evaluation")
    parser.add_argument("--wait", nargs='+', default=default.wait, help="how many batches to wait until a certain loss is used. Usage example: --wait hist 2e4 lr 100")
    parser.add_argument("--d_threshold", type=float, default=default.d_threshold, help="only train discriminator if the loss is below this threshold")
    parser.add_argument("--sinkhorn_eps", type=float, default=default.sinkhorn_eps, help="epsilon for sinkhorn distance")
    parser.add_argument("--d_channels", type=int, default=default.d_channels, nargs='+', help="how the discriminator is constructed eg --d_channels 16 32 32 64")
    parser.add_argument("--n_hardest", type=int, default=default.n_hardest, help="how many of the hardest constituents should be in the ground truth")
    parser.add_argument("--E_thres", type=float, default=default.E_thres, help="Energy threshold for the ground truth and the generator")
    parser.add_argument("--noise_factor", type=float, default=default.noise_factor, help="factor by which random noise is added, relative to 1Gev")
    parser.add_argument("--set_seed", type=int, default=default.set_seed, help="if used seed will be set to SEED. Else a random seed will be used")
    parser.add_argument('--deterministic', type=bool, default=default.deterministic, help='set numpy and cuda to run deterministically')
    parser.add_argument("--eval_modes", nargs='+', type=str, default=default.eval_modes, help="what modes to calculate the distributions for during evaluation")
    parser.add_argument("--drop_rate", type=float, default=default.drop_rate, help="drop rate for the Generator")
    parser.add_argument("--res_scale", type=float, default=default.res_scale, help="residual weighting factor")
    parser.add_argument("--lambda_reg", type=float, default=default.lambda_reg, help="Regularization weighting factor for gradient penalty")
    parser.add_argument("--hit_threshold", type=float, default=default.hit_threshold, help='threshold for generating the hitogram')
    parser.add_argument("--emd_save", type=str_to_bool, default=default.emd_save, help="Whether to save the Energy moving distance has decreased. smart_save needs to be enabled.")
    # parser.add_argument("--learn_powers", type=str_to_bool, default=default.learn_powers, help="whether to learn the powers of the MultiGenerator")
    # number of batches to train from instead of number of epochs.
    # If specified the training will be interrupted after N_BATCHES of training.
    parser.add_argument("--n_batches", type=int, default=default.n_batches, help="number of batches of training")
    parser.add_argument("--n_checkpoints", default=default.n_checkpoints, type=int, help="number of checkpoints during training (if used dominates checkpoint_interval)")
    parser.add_argument("--n_validations", type=int, default=default.n_validations, help="number of validation points during training (if used dominates validation_interval)")
    parser.add_argument("--n_evaluation", type=int, default=default.n_evaluation, help="number of histograms to compute during trianing")
    parser.add_argument("--default", type=str, default=default.default, help="Path to a json file. When this option is provided, all unspecified arguments will be taken from the json file")
    # modify generator
    parser.add_argument("--use_transposed_conv", type=str_to_bool, default=False, help="Whether to use transposed convolutions in upsampling")
    parser.add_argument("--fully_transposed_conv", type=str_to_bool, default=False, help="Whether to ONLY use transposed convolutions in upsampling")
    parser.add_argument("--num_final_res_blocks", type=int, default=0, help="Whether to add res blocks AFTER upsampling")
    parser.add_argument("--second_discr_reset_interval", type=int, default=default.second_discr_reset_interval, help="Interval in batches done after which 2nd discr weights are reseted")
    parser.add_argument("--uniform_init", type=str_to_bool, default=default.uniform_init, help="use xavier uniform init for generator")
    parser.add_argument("--uniform_reset", type=str_to_bool, default=default.uniform_reset, help="use xavier uniform init for discriminator reset")
    #Wasserstein GAN
    parser.add_argument("--wasserstein", type=float, default=-1, help="whether to use wasserstein conditional criticand corresponding lambda")
    parser.add_argument('--save_late', type=int, default=default.save_late, help='saves the weights after nth batch, regardles of performance' )
    #set zero lists
    parser.add_argument('--set_zero_def', nargs='+', default=[], choices=['hr', 'lr', 'adv', 'nnz', 'mask', 'hist', 'wasser', 'hito'], help='sets the losses in the list to zero when processing the default picture')
    parser.add_argument('--set_zero_pow', nargs='+', default=[], choices=['hr', 'lr', 'adv', 'nnz', 'mask', 'hist', 'wasser', 'hito'], help='sets the losses in the list to zero when processing the power picture')
    #nth constituent eval mode
    parser.add_argument('--nth_jet_eval_mode', choices=['hr', 'lr', 'all'], default='hr', help='what histograms contribute to the eval results for the nth hardest jets')
    parser.add_argument('--split_eval', type=bool, default=False, help='compares the eval results seperately when determining the best savepoint')
    opt = parser.parse_args()

    if opt.split_eval:
        assert 'hitogram' in opt.eval_modes, 'need hitogram in modes to split eval'
        if not any('E_' in mode for mode in opt.eval_modes):
            assert False, 'need at least one jet in eval modes for eval splitting'

    if opt.default:
        given = vars(opt)
        with open(opt.default, 'r') as f:
            arguments = json.load(f)
        if 'argument' in arguments:  # check if an info.json file was given
            arguments = arguments['argument']
        # reduce to only non default arguments
        given = {key: given[key] for key in given.keys() if default_dict[key] != given[key]}
        # add all arguments from opt.default
        arguments = {**given, **{key: arguments[key] for key in arguments if key not in given}}
        # add remaining default arguments if not all keys are specified
        arguments = {**arguments, **{key: default_dict[key] for key in default_dict if key not in arguments}}
        opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
    print(opt)
    return opt


def train(opt, **kwargs):
    model_name = '' if opt.name is None else (opt.name + '_')
    start_epoch = 0
    global gpu
    if 'gpu' in kwargs:
        gpu = kwargs['gpu']
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
    # add the whole dictionary to the info file
    info['argument'] = opt_dict
    lambdas = [opt.lambda_pix, opt.lambda_pow]

    os.makedirs(os.path.join(opt.root, opt.model_path), exist_ok=True)
    device = torch.device("cuda:%i" % gpu if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)
    # set seed
    if opt.set_seed > 0:
        seed = opt.set_seed
    #else:
        #seed = np.random.randint(2**31-1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        info['seed'] = seed

    # combine def and pow zero lists:
    set_zero_list = []
    set_zero_list.append(opt.set_zero_def)
    set_zero_list.append(opt.set_zero_pow) 

    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(
        np.log2(opt.factor)), multiplier=opt.pixel_multiplier, power=opt.scaling_power, drop_rate=opt.drop_rate, res_scale=opt.res_scale, use_transposed_conv=opt.use_transposed_conv, fully_tconv_upsample=opt.fully_transposed_conv, num_final_layer_res=opt.num_final_res_blocks, uniform_init=opt.uniform_init).to(device)
    if opt.E_thres:
        generator.thres = opt.E_thres
    Discriminators = pointerList()
    if opt.second_discr_reset_interval > 0:
        SecondDiscriminators = pointerList()
    for k in range(2):
        if lambdas[k] > 0:
            if (opt.discriminator == 'patch' and not opt.conditional and opt.wasserstein < 0):
                print("patch")
                discriminator = Markovian_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)
            elif (opt.discriminator == 'standard' and not opt.conditional and opt.wasserstein < 0):
                print("std")
                discriminator = Standard_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)
            
            elif (opt.discriminator == 'patch' and not opt.conditional and opt.wasserstein > 0):
                print('patch wasserstein')
                discriminator = Wasserstein_PatchDiscriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)
            
            elif opt.conditional:
                print("conditional")
                if opt.wasserstein > 0:
                    print("conditional wasserstein")
                    discriminator = Wasserstein_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels, num_upsample=int(np.log2(opt.factor))).to(device)
                else:
                    discriminator = Conditional_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels, num_upsample=int(np.log2(opt.factor))).to(device)
            Discriminators[k] = discriminator
    if opt.second_discr_reset_interval > 0:
        for k in range(2):
            if lambdas[k] > 0:
                if (opt.discriminator == 'patch' and not opt.conditional and opt.wasserstein < 0):
                    discriminator = Markovian_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)
                elif (opt.discriminator == 'standard' and not opt.conditional and opt.wasserstein < 0):
                    discriminator = Standard_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)
                
                elif (opt.discriminator == 'patch' and not opt.conditional and opt.wasserstein > 0):
                    discriminator = Wasserstein_PatchDiscriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels).to(device)

                elif opt.conditional:
                    if opt.wasserstein > 0:
                        discriminator = Wasserstein_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels, num_upsample=int(np.log2(opt.factor))).to(device)
                    else:
                        discriminator = Conditional_Discriminator(input_shape=(opt.channels, *hr_shape), channels=opt.d_channels, num_upsample=int(np.log2(opt.factor))).to(device)
                SecondDiscriminators[k] = discriminator

    discriminator_outshape = Discriminators.get(0).output_shape
    if opt.wasserstein > 0:
        discriminator_outshape = (opt.batch_size, 1)

    # Losses
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = nn.L1Loss().to(device)
    mse = nn.MSELoss().to(device)
    criterion_hist = pointerList()
   #criterion_hit = nn.KLDivLoss(reduction='batchmean')
    load_chk = False
    if opt.load_checkpoint:
        load_chk = True
        # Load pretrained models
        generator.load_state_dict(torch.load(opt.load_checkpoint, map_location=device))
        generator_file = os.path.basename(opt.load_checkpoint)
        for k in range(2):
            if lambdas[k] > 0:
                try:
                    Discriminators[k].load_state_dict(torch.load(opt.load_checkpoint.replace(
                        generator_file, generator_file.replace('generator', ['discriminator', 'discriminator_pow'][k])), map_location=device))
                except FileNotFoundError:
                    pass
        # extract model name if no name specified
        if model_name == '':
            model_name = generator_file.split('generator')[0]
            info_path = os.path.join(opt.model_path, model_name+'info.json')
        try:
            with open(info_path, 'r') as info_file:
                info = json.load(info_file)

            #make a savecopy of old info file
            info_path_old = info_path.replace('.json', '_old.json')
            with open(info_path_old, 'w') as old_info:
                json.dump(info, old_info)
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
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g if opt.lr_g > 0 else opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.l2decay)
    optimizer_D = pointerList()
    optimizer_secondD = pointerList()
    scheduler_D = pointerList()
    for k in range(2):
        if lambdas[k] > 0:
            optimizer_D[k] = torch.optim.Adam(Discriminators[k].parameters(), lr=opt.lr_d if opt.lr_d > 0 else opt.lr, betas=(opt.b1, opt.b2))
            if opt.second_discr_reset_interval > 0:
                optimizer_secondD[k] = torch.optim.Adam(SecondDiscriminators[k].parameters(), lr=opt.lr_d if opt.lr_d > 0 else opt.lr, betas=(opt.b1, opt.b2))
            scheduler_D[k] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D[k], verbose=False, patience=5)
    # LR Scheduler
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, verbose=True, patience=5)
    dataset = get_dataset(opt.dataset_type, opt.dataset_path, opt.hr_height, opt.hr_width, opt.factor, pre=opt.pre_factor, threshold=opt.E_thres, N=opt.n_hardest,noise_factor=opt.noise_factor)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        # pin_memory=True
    )
    eps = 1e-7
    pool = SumPool2d(opt.factor).to(device)
    WasserDist = SinkhornDistance(opt.sinkhorn_eps, 100, 'sum').to(device)
    e_max = 50  # random initialization number for the maximal pixel value
    nnz = []  # list with nonzero values during the first few batches
    binedges = []  # list with bin edges for energy distribution training
    histograms = pointerList()
    best_eval_result, best_emd_result = float('inf'), float('inf')
    if opt.load_checkpoint:
        try:
            best_eval_result = list(info["saved_batch"].values())[-1][1] #set best_save to latest save
            print('Loaded best eval result: ', best_eval_result)
        except KeyError:
            print('couldnt load best save')
    if opt.split_eval:
        best_eval_split = [float(10e20), float(10e20)]
        if opt.load_checkpoint:
            try:
                best_eval_split = list(info["saved_split"].values())[-1][1]
            except KeyError:
                print('couldnt load best split')

    if opt.lambda_hit > 0:
        genhit_ls = [] # list for generated hitograms
        gthit_ls = []  # list for ground truth hitograms
        batch_ls = []  # list for batches done info
        vmin = None
        vmax = None

    # ----------
    #  Training
    # ----------
    if opt.second_discr_reset_interval > 0:
        loss_dict = info['loss'] if 'loss' in info else {loss: [] for loss in ['d_loss_def', 'd_loss_pow', 'd2_loss_def', 'd2_loss_pow', 'g_loss', 'def_loss',
                                                                            'pow_loss', 'adv_loss','adv_loss_pow', 'pixel_loss','pixel_loss_pow', 'lr_loss','lr_loss_pow', 'hist_loss','hist_loss_pow', 'nnz_loss','nnz_loss_pow', 'mask_loss','mask_loss_pow', 'wasser_loss','wasser_loss_pow', 'hit_loss','hit_loss_pow', 'wasser_dist','wasser_dist_pow']}
    else:
        loss_dict = info['loss'] if 'loss' in info else {loss: [] for loss in ['d_loss_def', 'd_loss_pow', 'g_loss', 'def_loss',
                                                                            'pow_loss', 'adv_loss','adv_loss_pow', 'pixel_loss','pixel_loss_pow', 'lr_loss','lr_loss_pow', 'hist_loss','hist_loss_pow', 'nnz_loss','nnz_loss_pow', 'mask_loss','mask_loss_pow', 'wasser_loss','wasser_loss_pow', 'hit_loss','hit_loss_pow', 'wasser_dist','wasser_dist_pow']}                                                                        
    # if trainig is continued the batch number needs to be increased by the number of batches already trained on
    try:
        batches_trained = int(info['batches_done'])
    except KeyError:
        batches_trained = 0

    start_epoch = info['epochs'] # >0 if training is continued
    total_batches = len(dataloader)*(opt.n_epochs - start_epoch) if n_batches == np.inf else (n_batches + batches_trained)
    batches_done = batches_trained-1

    # function for saving info.json
    save_info_file = opt.save_info if opt.save_info is not None else opt.save

    def save_info():
        if save_info_file:
            with open(info_path, 'w') as outfile:
                info['loss'] = loss_dict
                info['batches_done'] = batches_done
                json.dump(info, outfile)

    def save_weights(epoch):
        if opt.save:
            if opt.load_checkpoint:
                torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%d_continued.pth" % (model_name, epoch)))
                for k in range(2):
                    if lambdas[k] > 0:
                        torch.save(Discriminators[k].state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator%s_%d_continued.pth" % (model_name, ['', '_pow'][k], epoch)))
            else:
                torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%d.pth" % (model_name, epoch)))
                for k in range(2):
                    if lambdas[k] > 0:
                        torch.save(Discriminators[k].state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator%s_%d.pth" % (model_name, ['', '_pow'][k], epoch)))

            print('Saved model to %s' % opt.model_path)
            save_info()

    def wait(short):
        if short in opt.wait:
            if float(opt.wait[opt.wait.index(short)+1]) > batches_done:
                return False
        return True

    for epoch in range(start_epoch, opt.n_epochs + start_epoch): #if training is continued need to add starting epochs
        for i, imgs in enumerate(dataloader):
            batches_done += 1
            # batches_done = epoch * len(dataloader) + i
            # Configure model input
            imgs_lr = imgs["lr"].to(device).float()
            imgs_hr = imgs["hr"].to(device).float()

            batch_size = imgs_lr.size(0)
            # Adversarial ground truths
            valid = torch.ones(batch_size, *discriminator_outshape, requires_grad=False).to(device).float()
            fake = torch.zeros(batch_size, *discriminator_outshape, requires_grad=False).to(device).float()

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()
            if not load_chk or opt.lambda_hist > 0:
                if batches_done - batches_trained < opt.warmup_batches:
                    # Warm-up (pixel-wise loss only)
                    if opt.learn_warmup:
                        # Generate a high resolution image from low resolution input
                        gen_hr = generator(imgs_lr)
                        # Measure pixel-wise loss against ground truth
                        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

                        loss_pixel.backward()
                        optimizer_G.step()
                        if batches_done % opt.report_freq == 0:
                            for gs in ['g_loss', 'pixel_loss']:
                                loss_dict[gs].append(loss_pixel.item())
                            print(
                                "[Batch %d/%d] [Epoch %d/%d] [G pixel: %f]"
                                % (i, total_batches, epoch, opt.n_epochs,  loss_pixel.item())
                            )
                    if opt.lambda_hist > 0:
                        # find a good value to cut off the distribution
                        imgs_hr = imgs_hr.cpu().view(-1).numpy()
                        nnz.extend(list(imgs_hr[imgs_hr > 0]))
                    continue
                elif batches_done - batches_trained == opt.warmup_batches:
                    if opt.lambda_hist > 0:
                        nnz = np.array(nnz)
                        for k in range(2):
                            if lambdas[k] > 0:
                                c, b = np.histogram(nnz**[1, opt.scaling_power][k], 100)
                                e_max = b[(np.cumsum(c) > len(nnz**[1, opt.scaling_power][k])*.9).argmax()]  # set e_max to the value where 90% of the data is smaller
                                print("found e_max to be %.2f" % e_max)
                                sorted_nnz = np.sort(nnz)
                                sorted_nnz = sorted_nnz[sorted_nnz <= e_max]
                                k_mean = KMeans(n_clusters=opt.bins, random_state=0).fit(sorted_nnz.reshape(-1, 1))
                                binedgesk = np.sort(k_mean.cluster_centers_.flatten())
                                binedges.append(np.array([0, *(np.diff(binedgesk)/2+binedgesk[:-1]), e_max]))
                                info['binedges%i' % k] = list(binedges[-1])
                                histograms[k] = DiffableHistogram(binedges[-1], sigma=opt.sigma, batchwise=opt.batchwise_hist).to(device)
                                criterion_hist[k] = KLD_hist(torch.from_numpy(binedges[-1])).to(device)
                    del nnz
            if i == opt.warmup_batches or (batches_done % opt.update_g == 0):
                # Main training loop
                loss_pixel, loss_lr_pixel, loss_GAN, loss_hist, loss_nnz, loss_mask, loss_wasser, loss_hit, w_loss = [
                    pointerList() for _ in range(9)]
                
                for k in range(2):
                    loss_pixel[k], loss_lr_pixel[k], loss_GAN[k], loss_hist[k], loss_nnz[k], loss_mask[k], loss_wasser[k], loss_hit[k], w_loss[k] = [
                        torch.zeros(1, device=device, dtype=torch.float32) for _ in range(9)]

                loss_G, loss_def, loss_pow = [torch.zeros(1, device=device, dtype=torch.float32) for _ in range(3)]
                # Generate a high resolution image from low resolution input
                generated = pointerList(generator(imgs_lr))
                generated.append(generator.srs)
                ground_truth = pointerList(imgs_hr, imgs_hr**opt.scaling_power)
                gen_lr = pool(generated[0])
                generated_lr = pointerList(gen_lr, gen_lr**opt.scaling_power)
                ground_truth_lr = pointerList(imgs_lr, imgs_lr**opt.scaling_power)
                # check for nan in the tensors:
                '''for l,pl in enumerate([generated, ground_truth, generated_lr, ground_truth_lr]):
                    for k in range(len(pl)):
                        plksum = pl.get(k).sum()
                        if plksum != plksum:
                            print('%f in list %i, index %i of %i. Shape: %s'%(plksum, l, k, len(pl), str(pl.get(k).shape)))'''

                tot_loss = pointerList(loss_def, loss_pow)
                # iterate over both the normal image and the image raised to opt.scaling_power
                for k in range(2):
                    if lambdas[k] > 0:
                        if opt.lambda_hr > 0 and wait('hr') and not 'hr' in set_zero_list[k]:
                            # Measure pixel-wise loss against ground truth
                            loss_pixel[k] = criterion_pixel(generated[k].mean(0)[None, ...], ground_truth[k].mean(0)[None, ...])
                        if opt.lambda_lr > 0 and wait('lr') and not 'lr' in set_zero_list[k]:
                            # Measure pixel-wise loss against ground truth for downsampled images
                            loss_lr_pixel[k] = criterion_pixel(generated_lr[k], ground_truth_lr[k])
                        if opt.lambda_adv > 0 and wait('adv') and not 'adv' in set_zero_list[k]:
                            # Extract validity generated[k]s from discriminator
                            pred_real = Discriminators[k](ground_truth[k], ground_truth_lr[k]).detach()
                            pred_fake = Discriminators[k](generated[k], generated_lr[k])
                            if opt.second_discr_reset_interval > 0:
                                pred_real2 = SecondDiscriminators[k](ground_truth[k], ground_truth_lr[k]).detach()
                                pred_fake2 = SecondDiscriminators[k](generated[k], generated_lr[k])
                            if opt.relativistic:
                                # Adversarial loss (relativistic average GAN)
                                if opt.second_discr_reset_interval > 0:
                                    loss_GAN[k] = .5*(.5*(criterion_GAN(eps + pred_fake - pred_real.mean(0, keepdim=True), valid) +
                                            criterion_GAN(eps + pred_real - pred_fake.mean(0, keepdim=True), fake)) +
                                            .5*(criterion_GAN(eps + pred_fake2 - pred_real2.mean(0, keepdim=True), valid) +
                                            criterion_GAN(eps + pred_real2 - pred_fake2.mean(0, keepdim=True), fake))
                                            )
                                else:
                                    loss_GAN[k] = .5*(criterion_GAN(eps + pred_fake - pred_real.mean(0, keepdim=True), valid) +
                                            criterion_GAN(eps + pred_real - pred_fake.mean(0, keepdim=True), fake))
                            else:
                                loss_GAN[k] = criterion_GAN(eps + pred_fake, valid)

                        if opt.wasserstein > 0:
                            w_fake = Discriminators[k](generated[k], generated_lr[k])
                            w_loss[k] = -torch.mean(w_fake)
                            if opt.second_discr_reset_interval > 0:
                                w_fake2 = SecondDiscriminators[k](generated[k], generated_lr[k])
                                w_loss2 = -torch.mean(w_fake2)
                                w_loss[k] = .5*(w_loss + w_loss2)
                            


                        if opt.lambda_nnz > 0 and wait('nnz') and not 'nnz' in set_zero_list[k]:
                            gen_nnz = softgreater(generated[k], 0, 50000).sum(1).sum(1).sum(1)
                            target = (ground_truth[k] > 0).sum(1).sum(1).sum(1).float().to(device)
                            loss_nnz[k] = mse(gen_nnz, target)
                        if opt.lambda_mask > 0 and wait('mask') and not 'mask' in set_zero_list[k]:
                            gen_mask = nnz_mask(generated[k])
                            real_mask = nnz_mask(ground_truth[k])
                            loss_mask[k] = criterion_pixel(gen_mask, real_mask)
                        if opt.lambda_hist > 0 and wait('hist') and not 'hist' in set_zero_list[k]:
                            # calculate the energy distribution loss
                            # first calculate the both histograms
                            gen_nnz = generated[k][generated[k] > 0]
                            real_nnz = ground_truth[k][ground_truth[k] > 0]
                            gen_hist = histograms[k](gen_nnz)
                            real_hist = histograms[k](real_nnz)
                            loss_hist[k] = criterion_hist[k](gen_hist, real_hist)
                            # print(gen_hist,real_hist,loss_hist)
                        if opt.lambda_wasser > 0 and wait('wasser') and not 'wasser' in set_zero_list[k]:
                            gen_sort, _ = torch.sort(generated[k].view(batch_size, -1), 1)
                            real_sort, _ = torch.sort(ground_truth[k].view(batch_size, -1), 1)
                            loss_wasser[k], _, _ = WasserDist(cut_smaller(gen_sort)[..., None], cut_smaller(real_sort)[..., None])
                        if opt.lambda_hit > 0 and wait('hit') and not 'hito' in set_zero_list[k]:
                            gen_hit = get_hitogram(generated[k], opt.factor, opt.hit_threshold, opt.sigma)#+eps
                            target = get_hitogram(ground_truth[k], opt.factor, opt.hit_threshold, opt.sigma)
                            #loss_hit = criterion_hit((gen_hit/gen_hit.sum()).log(), target/(target.sum()))
                            loss_hit[k] = mse(gen_hit, target)

                        tot_loss[k] = opt.lambda_hr * loss_pixel[k] + opt.lambda_adv * loss_GAN[k] + opt.lambda_lr * loss_lr_pixel[k] + opt.lambda_nnz * \
                            loss_nnz[k] + opt.lambda_mask * loss_mask[k] + opt.lambda_hist * loss_hist[k] + opt.lambda_wasser * loss_wasser[k] + opt.lambda_hit * loss_hit[k] + opt.wasserstein * w_loss[k]
                        # Total generator loss
                        loss_G += lambdas[k] * tot_loss[k]
                loss_G.backward()
                # torch.nn.utils.clip_grad_value_(generator.parameters(), 1)
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if i == opt.warmup_batches or (batches_done % opt.update_d == 0):
                loss_D_tot = [torch.zeros(1, device=device) for _ in range(2)]
                if opt.second_discr_reset_interval > 0:
                    loss_secondD_tot = [torch.zeros(1, device=device) for _ in range(2)]
                for k in range(2):
                    lam = lambdas[k]
                    if lam > 0:
                        optimizer_D[k].zero_grad()
                        pred_real = Discriminators[k](ground_truth[k], ground_truth_lr[k])
                        pred_fake = Discriminators[k](generated[k].detach(), ground_truth_lr[k])
                        if opt.second_discr_reset_interval > 0:
                            optimizer_secondD[k].zero_grad()
                            pred_real2 = SecondDiscriminators[k](ground_truth[k], ground_truth_lr[k])
                            pred_fake2 = SecondDiscriminators[k](generated[k].detach(), ground_truth_lr[k]) 
                        if opt.wasserstein < 0: # use minmax if not wasserstein gan                         
                            if opt.relativistic:
                                # Adversarial loss for real and fake images (relativistic average GAN)
                                loss_real = criterion_GAN(eps + pred_real - pred_fake.mean(0, keepdim=True), valid)
                                loss_fake = criterion_GAN(eps + pred_fake - pred_real.mean(0, keepdim=True), fake)
                                if opt.second_discr_reset_interval > 0:   
                                    loss_real2 = criterion_GAN(eps + pred_real2 - pred_fake2.mean(0, keepdim=True), valid)
                                    loss_fake2 = criterion_GAN(eps + pred_fake2 - pred_real2.mean(0, keepdim=True), fake)
                            else:
                                loss_real = criterion_GAN(eps + pred_real, valid)
                                loss_fake = criterion_GAN(eps + pred_fake, fake)
                            # print(pred_fake[0].item(),pred_fake.mean(0, keepdim=True)[0].item(),loss_fake.item(),pred_real[0].item(),loss_real.item(),pred_real.mean(0, keepdim=True)[0].item())
                            # Total loss
                            loss_D = (loss_real + loss_fake) / 2
                            if opt.second_discr_reset_interval > 0:
                                loss_secondD = (loss_real2 + loss_fake2) / 2
                        else:
                            loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) # wasserstein formula: max over params, so minimize -1* that 
                            if opt.second_discr_reset_interval > 0:
                                loss_secondD = -torch.mean(pred_real2) + torch.mean(pred_fake2)

                        if opt.lambda_reg > 0:
                            # generate interpolation between real and fake data
                            epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
                            interpolation = epsilon*ground_truth[k]+(1-epsilon)*generated[k].detach()
                            interpolation.requires_grad = True
                            pred_interpolation = Discriminators[k](interpolation, ground_truth_lr[k])
                            gradients = torch.autograd.grad(outputs=pred_interpolation, inputs=interpolation, grad_outputs=valid,
                                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
                            gradients = gradients.view(batch_size, -1)
                            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_reg/2
                            loss_D += gradient_penalty

                            if opt.second_discr_reset_interval > 0:
                                pred_interpolation2 = SecondDiscriminators[k](interpolation, ground_truth_lr[k])
                                gradients2 = torch.autograd.grad(outputs=pred_interpolation2, inputs=interpolation, grad_outputs=valid,
                                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                                gradients2 = gradients2.view(batch_size, -1)
                                gradient_penalty2 = ((gradients2.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_reg/2
                                loss_secondD += gradient_penalty2                               

                        loss_D.backward()
                        if opt.second_discr_reset_interval > 0:
                            loss_secondD.backward()
                            loss_secondD_tot[k] = loss_secondD
                        # torch.nn.utils.clip_grad_value_(Discriminators[k].parameters(), 1)
                        loss_D_tot[k] = loss_D
                        # only train discriminator if it is not already too good
                        if (loss_D.item() > opt.d_threshold):
                            optimizer_D[k].step()
                            if opt.second_discr_reset_interval > 0:
                                optimizer_secondD[k].step()

            # --------------
            #  Log Progress
            # --------------
            # save loss to dict
            if batches_done % opt.report_freq == 0:
                if opt.second_discr_reset_interval > 0:
                    for v, l in zip(loss_dict.values(), [loss_D_tot[0].item(), loss_D_tot[1].item(), loss_secondD_tot[0].item(), loss_secondD_tot[1].item(), loss_G.item(), tot_loss[0].item(), tot_loss[1].item(), loss_GAN[0].item(),loss_GAN[1].item(), loss_pixel[0].item(),loss_pixel[1].item(), loss_lr_pixel[0].item(),loss_lr_pixel[1].item(), loss_hist[0].item(),loss_hist[1].item(), loss_nnz[0].item(),loss_nnz[1].item(), loss_mask[0].item(),loss_mask[1].item(), loss_wasser[0].item(),loss_wasser[1].item(), loss_hit[0].item(),loss_hit[1].item(), w_loss[0].item(),w_loss[1].item()]):
                        v.append(l)
                    print("[Batch %d] [D def: %f, pow: %f] [2ndD def: %f, pow: %f] [G loss: %f [def: %f, pow: %f], adv: %f, adv pow: %f, pixel: %f, pixel pow: %f, lr pixel: %f, lr pixel pow: %f, hist: %f, hist pow: %f, nnz: %f, nnz pow: %f, mask: %f, mask pow: %f, wasser: %f, wasser pow: %f, hit: %f, hit pow: %f, wasserdist: %f, wasserdist pow: %f]"
                        % (batches_done, *[l[-1] for l in loss_dict.values()],))
                else:
                    for v, l in zip(loss_dict.values(), [loss_D_tot[0].item(), loss_D_tot[1].item(), loss_G.item(), tot_loss[0].item(), tot_loss[1].item(), loss_GAN[0].item(),loss_GAN[1].item(), loss_pixel[0].item(),loss_pixel[1].item(), loss_lr_pixel[0].item(),loss_lr_pixel[1].item(), loss_hist[0].item(),loss_hist[1].item(), loss_nnz[0].item(),loss_nnz[1].item(), loss_mask[0].item(),loss_mask[1].item(), loss_wasser[0].item(),loss_wasser[1].item(), loss_hit[0].item(),loss_hit[1].item(), w_loss[0].item(),w_loss[1].item()]):
                        v.append(l)
                    print("[Batch %d] [D def: %f, pow: %f] [G loss: %f [def: %f, pow: %f], adv: %f, adv pow: %f, pixel: %f, pixel pow: %f, lr pixel: %f, lr pixel pow: %f, hist: %f, hist pow: %f, nnz: %f, nnz pow: %f, mask: %f, mask pow: %f, wasser: %f, wasser pow: %f, hit: %f, hit pow: %f, wasserdist: %f, wasserdist pow: %f]"
                        % (batches_done, *[l[-1] for l in loss_dict.values()],))                    

            # check if loss is NaN
            if any(l != l for l in [loss_D_tot[0].item(), loss_D_tot[1].item(), loss_G.item()]):
                save_info()
                raise ValueError('loss is NaN\n[Batch %d] [D loss: %e] [G loss: %f [def: %f, pow: %f], adv: %f, pixel: %f, lr pixel: %f, hist: %f, nnz: %f, mask: %f]' % (
                    i, loss_D_tot.item(), loss_G.item(), tot_loss[0].item(), tot_loss[1].item(), loss_GAN.item(), loss_pixel.item(), loss_lr_pixel.item(), loss_hist.item(), loss_nnz.item(), loss_mask.item()))
            if batches_done % opt.sample_interval == 0 and not opt.sample_interval == -1:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.factor)
                img_grid = torch.cat((imgs_hr, imgs_lr, generated[0]), -1)
                save_image(img_grid, os.path.join(opt.root, image_dir, "%d.png" % batches_done), nrow=1, normalize=False)
                if opt.plot_grad:
                    # plot gradients
                    plot_grad_flow(generator.named_parameters(), os.path.join(image_dir, 'grad_%d.png' % batches_done))

            if ((validation_interval != np.inf and (batches_done+1) % validation_interval == 0) or (
                    validation_interval == np.inf and (batches_done+1) % (total_batches//opt.n_validations) == 0)) and opt.validation_path is not None:
                print('Validation')
                output_path = opt.output_path if 'output_path' in dir(opt) else None
                val_results = calculate_metrics(opt.validation_path, opt.dataset_type, generator, device, output_path, opt.batch_size,
                                                opt.n_cpu, opt.bins, opt.hr_height, opt.hr_width, opt.factor, pre=opt.pre_factor, amount=opt.N//10, thres=opt.E_thres, N=opt.n_hardest,noise_factor=opt.noise_factor)
                val_results['epoch'] = epoch
                val_results['batch'] = batches_done
                # If necessary lower the learning rate
                try:
                    hrl1val = val_results['metrics']['hr_l1']['mean']
                    scheduler_G.step(hrl1val)
                    scheduler_D.call('step', hrl1val)
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
                eval_result = distribution(opt.validation_path, opt.dataset_type, generator, device, os.path.join(image_dir, '%d_hist.png' % batches_done),
                                           30, 0, 30, opt.hr_height, opt.hr_width, opt.factor, opt.N, pre=opt.pre_factor, thres=opt.E_thres, N=opt.n_hardest,
                                           mode=opt.eval_modes, noise_factor=opt.noise_factor, nth_jet_eval_mode=opt.nth_jet_eval_mode, split_eval=opt.split_eval)

                if opt.split_eval:
                    eval_split_raw = []
                    eval_split_raw.append(float(np.abs(eval_result['hitogram'])))
                    tmp = []
                    for key in eval_result:
                        if key == 'hitogram':
                            continue
                        tmp.append(eval_result[key])
                    eval_split_raw.append(float(np.mean(np.abs(tmp))))
                    if 'eval_split' in info:
                        info['eval_split'].append(eval_split_raw)
                    else:
                        info['eval_split'] = [eval_split_raw]
      
                    if ((eval_split_raw[0] - best_eval_split[0])/best_eval_split[0] + (eval_split_raw[1] - best_eval_split[1])/best_eval_split[1]) < 0:
                        best_eval_split = eval_split_raw
                        if opt.smart_save:
                            if not opt.emd_save:
                                try:
                                    info['saved_split'][epoch] = [batches_done, best_eval_split]
                                except KeyError:
                                    info['saved_split'] = {epoch: [batches_done, best_eval_split]}
                                try:
                                    info['saved_batch'][epoch] = [batches_done, float(np.mean(np.abs(best_eval_split)))]
                                except KeyError:
                                    info['saved_batch'] = {epoch: [batches_done, float(np.mean(np.abs(best_eval_split)))]}
                                save_weights(epoch)
                        

                mean_grid = torch.cat((generated[0].mean(0)[None, ...], ground_truth[0].mean(0)[None, ...]), -1)
                save_image(mean_grid, os.path.join(opt.root, image_dir, "%d_mean.png" % batches_done), nrow=1, normalize=False)

                if  (wait('hit') and (batches_done - batches_trained) > 20000 and (batches_done - batches_trained) < 50000 and opt.lambda_hit > 0):
                    #hit_f = plot_hist2d(gen_hit.cpu().detach(), target.cpu().detach())
                    #hit_f.savefig(os.path.join(opt.root, image_dir, "%d_batchhito.png" % batches_done))
                    genhit_ls.append(gen_hit.cpu().detach().numpy())  # save hitogram arrays and corresponding batch nr in list, plot them later
                    gthit_ls.append(target.cpu().detach().numpy())
                    batch_ls.append(batches_done)

                if (wait('hit') and (batches_done - batches_trained) > 50000 and opt.lambda_hit > 0):
                    hit_f = plot_hist2d(gen_hit.cpu().detach(), target.cpu().detach(), vmin=vmin, vmax=vmax)
                    hit_f.savefig(os.path.join(opt.root, image_dir, "%d_batchhito.png" % batches_done))
                
                if not opt.split_eval:
                    if eval_result is not None:
                        eval_result_mean = float(np.mean(np.abs(eval_result)))
                        if 'eval_results' in info:
                            info['eval_results'].append(eval_result)
                        else:
                            info['eval_results'] = [eval_result]
                        if eval_result_mean < best_eval_result:
                            best_eval_result = eval_result_mean
                            if opt.smart_save:
                                if not opt.emd_save:
                                    try:
                                        info['saved_batch'][epoch] = [batches_done, best_eval_result]
                                    except KeyError:
                                        info['saved_batch'] = {epoch: [batches_done, best_eval_result]}
                                    save_weights(epoch)
                if opt.emd_save and opt.smart_save:
                    val_results = calculate_metrics(opt.validation_path, opt.dataset_type, generator, device, None, opt.batch_size,
                                                    opt.n_cpu, opt.bins, opt.hr_height, opt.hr_width, opt.factor, pre=opt.pre_factor, amount=opt.N, thres=opt.E_thres, N=opt.n_hardest)
                    val_results['epoch'] = epoch
                    val_results['batch'] = batches_done
                    try:
                        info['validation'].append(val_results)
                    except KeyError:
                        info['validation'] = [val_results]
                    emd_result = val_results['emd']['mean']
                    if emd_result < best_emd_result:
                        best_emd_result = emd_result
                        try:
                            info['saved_batch'][epoch] = [batches_done, best_emd_result]
                        except KeyError:
                            info['saved_batch'] = {epoch: [batches_done, best_emd_result]}
                        save_weights(epoch)

                save_info()
                generator.train()

            # Save model checkpoints
            if (checkpoint_interval != np.inf and (batches_done+1) % checkpoint_interval == 0) or (
                    checkpoint_interval == np.inf and (batches_done+1) % (total_batches//opt.n_checkpoints) == 0):
                if not opt.smart_save:
                    save_weights(epoch)

            # Reset second generator
            if (opt.second_discr_reset_interval > 0 and (batches_done+1) % opt.second_discr_reset_interval == 0):
                for k in range(2):
                    if opt.uniform_reset:
                        SecondDiscriminators[k].apply(uniform_reset)
                    else:
                        SecondDiscriminators[k].apply(weight_reset)

            if ((batches_done - batches_trained +1) == 50000 and opt.lambda_hit > 0): # plot all hitos collected so far and determine vmin, vmax
                vmin = (min(np.concatenate(gthit_ls, axis=None)) + min(np.concatenate(genhit_ls, axis=None))) / 2
                vmax = (max(np.concatenate(gthit_ls, axis=None)) + max(np.concatenate(genhit_ls, axis=None))) / 2
                for sr,gt,batch in zip(genhit_ls, gthit_ls, batch_ls):
                    hit_f = plot_hist2d(sr, gt, vmin= vmin, vmax= vmax)
                    hit_f.savefig(os.path.join(opt.root, image_dir, "%d_batchhito.png" % batch))                   

            if (opt.save_late > 0 and (batches_done - batches_trained +1) == opt.save_late):
                torch.save(generator.state_dict(), os.path.join(opt.root, opt.model_path, "%sgenerator_%s_ep%i.pth" % (model_name, 'late_save', epoch)))
                for k in range(2):
                    if lambdas[k] > 0:
                        torch.save(Discriminators[k].state_dict(), os.path.join(opt.root, opt.model_path, "%sdiscriminator%s_%s_ep%i.pth" % (model_name, ['', '_pow'][k], 'late_save', epoch)))


            if batches_done == total_batches:
                save_info()
                return info
        info['epochs'] += 1
        save_info()


if __name__ == "__main__":
    print('pytorch version:', torch.__version__)
    print('GPU:', torch.cuda.get_device_name('cuda'))
    print('cuda version:', torch.version.cuda)
    print('cudnn version:', torch.backends.cudnn.version())
    try:
        gpu = get_gpu_index()
        num_gpus = torch.cuda.device_count()
        if gpu >= num_gpus:
            gpu = np.random.randint(num_gpus)
    except Exception as e:
        print(e)

    print('running on gpu index {}'.format(gpu))
    opt = get_parser()
    try:
        train(opt)
    except RuntimeError as e:
        if 'cuda' in str(e).lower():
            os.system('nvidia-smi > nsmi.txt')
            raise RuntimeError(open('nsmi.txt', 'r').read(), e)
        else:
            raise RuntimeError(e)
