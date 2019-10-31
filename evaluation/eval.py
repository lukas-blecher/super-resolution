import sys
import os
# add home directory to pythonpath
if os.path.split(os.path.abspath('.'))[1] == 'super-resolution':
    sys.path.insert(0, '.')
else:
    sys.path.insert(0, '..')
from datasets import *
#from evaluation.PSNR_SSIM_metric import calculate_ssim
#from evaluation.PSNR_SSIM_metric import calculate_psnr
from torch.utils.data import DataLoader
from models import GeneratorRRDB
from options.default import default_dict
import argparse
import json
from collections import namedtuple
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


def toUInt(x):
    return np.squeeze(x*255/x.max()).astype(np.uint8)


def save_numpy(array, path):
    Image.fromarray(toUInt(array)).save(path)


class HHDist:
    '''Highest value distribution'''

    def __init__(self):
        self.pred = []
        self.gt = []

    def append_gt(self, img):
        self.gt.append(img.max().item())

    def append_pred(self, img):
        self.pred.append(img.max().item())

    def append_pair(self, pred, gt):
        self.append_gt(gt)
        self.append_pred(pred)

    def get_range(self):
        mi = min(self.pred) 
        if min(self.gt) < mi:
            mi = min(self.gt)
        ma = max(self.pred)
        if max(self.gt) > ma:
            ma = max(self.gt)
        return (mi, ma)

    def histogram_gt(self, bins=10):
        return np.histogram(self.gt, bins, range=self.get_range())

    def histogram_pred(self, bins=10):
        return np.histogram(self.pred, bins, range=self.get_range())


def call_func(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dopt = dir(opt)
    output_path = opt.output_path if 'output_path' in dopt else None
    bins = opt.bins if 'bins' in dopt else 10
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model))
    if opt.histogram:
        return highest_energy_distribution(opt.dataset_path, generator, device, output_path, opt.batch_size, opt.n_cpu, bins)
    else:
        return calculate_metrics(opt.dataset_path, generator, device, output_path, opt.batch_size, opt.n_cpu, bins)


def calculate_metrics(dataset_path, generator, device, output_path=None, batch_size=4, n_cpu=0, bins=10, amount=None):
    generator.eval()
    if 'h5' in dataset_path:
        dataset = EventDataset(dataset_path, amount)
    else:
        dataset = EventDatasetText(dataset_path, amount)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_cpu
    )

    energy_dist, lr_similarity, hr_similarity = [], [], []
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    pool = SumPool2d()
    for _, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"]
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach().cpu()

        # compare downsampled generated image with lr ground_truth using l1 loss
        with torch.no_grad():
            # low resolution image L1 metric
            gen_lr = pool(gen_hr)
            l1_loss = l1_criterion(gen_lr, imgs_lr.cpu())
            lr_similarity.append(l1_loss.item())
            # high resolution image L1 metric
            hr_similarity.append(l1_criterion(gen_hr, imgs_hr).item())

            # energy distribution
            gen_nnz = gen_hr[gen_hr > 0]
            real_nnz = imgs_hr[imgs_hr > 0]
            t_min = torch.min(torch.cat((gen_nnz, real_nnz), 0)).item()
            t_max = torch.max(torch.cat((gen_nnz, real_nnz), 0)).item()
            gen_hist = torch.histc(gen_nnz, bins, min=t_min, max=t_max).float()
            real_hist = torch.histc(real_nnz, bins, min=t_min, max=t_max).float()
            energy_dist.append(l2_criterion(gen_hist, real_hist).item())

    results = {}
    for metric_name, metric_values in zip(['lr_l1', 'hr_l1', 'energy distribution'], [lr_similarity, hr_similarity, energy_dist]):
        results[metric_name] = {'mean': np.mean(metric_values), 'std': np.std(metric_values)}

    return results


def to_hist(data, bins):
    '''nearest neighbor interpolation for 1d numpy arrays'''
    hist = np.zeros(2*len(data))
    hist[::2] = data.copy()
    hist[1::2] = data.copy()
    x = np.vstack((bins, bins)).flatten('F')
    return x[1:-1], hist


def highest_energy_distribution(dataset_path, generator, device, output_path=None, batch_size=4, n_cpu=0, bins=10, amount=200):
    generator.eval()
    if 'h5' in dataset_path:
        dataset = EventDataset(dataset_path, amount)
    else:
        dataset = EventDatasetText(dataset_path, amount)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_cpu
    )

    hhd=HHDist()
    print('collecting data from %s' % dataset_path)
    for _, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"]
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach()
        hhd.append_pair(gen_hr, imgs_hr)

    plt.figure()
    plt.plot(*to_hist(*hhd.histogram_gt(bins)),label="ground truth")
    plt.plot(*to_hist(*hhd.histogram_pred(bins)),'--',label="model prediction")
    plt.title('highest energy distribution')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()
    return hhd

def hline(newline=False, n=100):
    print('_'*n) if not newline else print('_'*n, '\n')


def clean_labels(label):
    '''attempt to reduce legend size'''
    if 'lambda' in label:
        label = '$\\%s{%s}$' % (label[:7], label[7:])
    else:
        label = label.replace('residual', 'res')
        label = label.replace('_', ' ')
    return label


def evaluate_results(file):
    '''
    plot results from the hyperparameter search
        `file` should the path to the file containing the results
    '''
    with open(file) as f:
        results = json.load(f)

    def ts(l): return (len(l)-7)//8+2
    tmax = ts(max(results.keys()))
    # print constant arguments
    hline()
    for key, value in results.items():
        if key != 'results':
            print(key, '\t'*(2*tmax-ts(key)), value)
    hline(True)

    hyper_set = results['results']
    assert len(hyper_set) > 0
    p0 = hyper_set[0]['metrics'][0]
    num_metrics = len(p0)-2
    M = int(np.sqrt(num_metrics))
    N = num_metrics//M
    f, ax = plt.subplots(M, N)
    ax = ax.flatten()
    # iterate over every metric that was measured
    for m, (key, value) in enumerate(p0.items()):
        if key in ('epoch', 'batch'):
            m -= 1
            continue
        splt = ax[m]
        splt.set_title(key)
        splt.set_xlabel('iterations')
        # iterate over every set of hyperparameters that was investigated
        for h in range(len(hyper_set)):
            checkpoints = hyper_set[h]['metrics']
            label = ''
            for k, v in hyper_set[h].items():
                if k not in ('epoch', 'batch', 'metrics'):
                    vstr = str(v)
                    if len(vstr) > 6:
                        vstr = '%.4f' % v
                    label += '%s=%s ' % (clean_labels(k), vstr)
            iterations = []  # will contain batch number
            y = []
            y_err = []
            # iterate over every point in time that was measured
            for i in range(len(checkpoints)):
                p = checkpoints[i]  # point in time
                iterations.append(p['batch'])
                y.append(p[key]['mean'])
                y_err.append(p[key]['std'])
            _, caps, bars = splt.errorbar(iterations, y, yerr=y_err, label=label, capsize=1.5)
            # loop through bars and caps and set the alpha value
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            splt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to image")
    parser.add_argument("--output_path", type=str, default='images/outputs', help="Path where output will be saved")
    parser.add_argument("--checkpoint_model", type=str, default=None, help="Path to checkpoint model")
    parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
    parser.add_argument("-r", "--hyper_results", type=str, default=None, help="if used, show hyperparameter search results")
    parser.add_argument("--histogram", action="store_true", default=True, help="whether to show energy distribution histogram")
    parser.add_argument("--bins", type=int, default=10, help="number of bins in the histogram")

    opt = vars(parser.parse_args())
    if opt['hyper_results'] is not None:
        evaluate_results(opt['hyper_results'])
    else:
        if opt['dataset_path'] is None or opt['checkpoint_model'] is None:
            raise ValueError("For evaluation dataset_path and checkpoint_model are required")
        arguments = {**opt, **{key: default_dict[key] for key in default_dict if key not in opt}}
        opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
        # print(opt)
        print(call_func(opt))
