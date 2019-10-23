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

def toUInt(x):
    return np.squeeze(x*255/x.max()).astype(np.uint8)
    
def save_numpy(array,path):
    Image.fromarray(toUInt(array)).save(path)

def calcualte_metrics(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        MAX = opt.max
    except AttributeError:
        MAX = 80  # TODO get a theoretical estimate for the max number
    try:
        crop_border = opt.crop_border
    except AttributeError:
        crop_border = 4
    save_ims = 'output_path' in opt._asdict()
    dataset = JetDataset(opt.dataset_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu
    )
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device).eval()
    generator.load_state_dict(torch.load(opt.checkpoint_model))
    psnr, ssim, lr_similarity, hr_similarity = [], [], [], []
    l1_criterion = nn.L1Loss()
    pool = SumPool2d()
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"]
        imgs_hr_np = imgs_hr.permute(0, 2, 3, 1).numpy()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach().cpu()
        gen_hr_np = gen_hr.permute(0, 2, 3, 1).numpy()
        for j in range(len(imgs_hr)):
            ground_truth, generated = imgs_hr_np[j], gen_hr_np[j]
            #save generated hr image 
            if save_ims:
                save_numpy(generated, os.path.join(opt.output_path,'%s.png'%(i*opt.batch_size+j)))

            # crop
            if ground_truth.ndim == 3:
                ground_truth = ground_truth[crop_border:-crop_border, crop_border:-crop_border, :]
                generated = generated[crop_border:-crop_border, crop_border:-crop_border, :]
            elif ground_truth.ndim == 2:
                ground_truth = ground_truth[crop_border:-crop_border, crop_border:-crop_border]
                generated = generated[crop_border:-crop_border, crop_border:-crop_border]
            else:
                raise ValueError(
                    'Wrong image dimension: {}. Should be 2 or 3.'.format(ground_truth.ndim))
            psnr.append(calculate_psnr(ground_truth, generated, MAX))
            ssim.append(calculate_ssim(ground_truth, generated, MAX))
            
        # compare downsampled generated image with lr ground_truth using l1 loss
        with torch.no_grad():
            #low resolution image L1 metric
            gen_lr = pool(imgs['hr'])
            l1_loss = l1_criterion(gen_lr,imgs_lr.cpu())
            lr_similarity.append(l1_loss.item())
            #high resolution image L1 metric
            hr_similarity.append(l1_criterion(gen_hr, imgs_hr).item())

    results = {}
    for metric_name, metric_values in zip(['psnr', 'ssim', 'lr_l1', 'hr_l1'], [psnr, ssim, lr_similarity, hr_similarity]):
        results[metric_name] = np.mean(metric_values)
        results[metric_name+'_std'] = np.std(metric_values)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image")
    parser.add_argument("--output_path", type=str, default='images/outputs', help="Path where output will be saved")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
    opt = vars(parser.parse_args())

    arguments = {**opt, **{key: default_dict[key] for key in default_dict if key not in opt}}

    opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
    # print(opt)
    print(calcualte_metrics(opt))
