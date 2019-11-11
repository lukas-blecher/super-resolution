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
from options.default import *
import argparse
import json
from collections import namedtuple
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm.auto import tqdm 

def toUInt(x):
    return np.squeeze(x*255/x.max()).astype(np.uint8)


def save_numpy(array, path):
    Image.fromarray(toUInt(array)).save(path)


class MultHist:
    '''A class to collect data for any number of histograms
    modes: 'max' collects the maximum value for each image, 'min' collects the minimum value, 'mean' collects the mean value, 'nnz' saves the amount of nonzero values,
           'sum' collects the total energy for each image, 'meannnz' saves the mean energy for each image disregarding the empty pixels
    '''

    def __init__(self, num, mode='max'):
        self.num = num
        self.list = [[] for _ in range(num)]
        self.mode = mode
        self.thres = 0.002

    def append(self, *argv):
        assert len(argv) == self.num
        for i,L in enumerate(argv):
            Ln = L.detach().cpu().numpy()
            if self.mode == 'max':
                self.list[i].extend(list(Ln.max((1,2,3))))
            elif self.mode == 'min':
                self.list[i].extend(list(Ln.min((1,2,3))))
            elif self.mode == 'mean':
                self.list[i].extend(list(Ln.mean((1,2,3))))
            elif self.mode == 'nnz':
                self.list[i].extend(list((L>=self.thres).sum(1).sum(1).sum(1).detach().cpu().numpy()))
            elif self.mode == 'sum':
                self.list[i].extend(list(Ln.sum((1,2,3))))
            elif self.mode == 'meannnz':
                for j in range(len(Ln)):
                    self.list[i].append(Ln[j][Ln[j]>self.thres].mean())
        
    def get_range(self):
        mins = [min(self.list[i]) for i in range(self.num)]
        maxs = [max(self.list[i]) for i in range(self.num)]
        return min(mins), max(maxs)

    def histogram(self, L, bins=10):
        return np.histogram(L, bins, range=self.get_range())



def call_func(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dopt = dir(opt)
    output_path = opt.output_path if 'output_path' in dopt else None
    bins = opt.bins if 'bins' in dopt else default.bins

    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(np.log2(opt.factor))).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model))
    args = [opt.dataset_path, opt.dataset_type, generator, device, output_path, opt.batch_size, opt.n_cpu, bins, opt.hr_height, opt.hr_width, opt.factor, opt.batch_size]
    if opt.histogram:
        return distribution(*args, mode=opt.histogram)
    else:
        return calculate_metrics(*args)


def calculate_metrics(dataset_path, dataset_type, generator, device, output_path=None, batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=None):
    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu
    )

    energy_dist, lr_similarity, hr_similarity, nnz = [], [], [], []
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss()
    pool = SumPool2d(factor)
    for _, imgs in enumerate(dataloader):
        with torch.no_grad():
            # Configure model input
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"]
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr).detach().cpu()

            # compare downsampled generated image with lr ground_truth using l1 loss
        
            # low resolution image L1 metric
            gen_lr = pool(gen_hr)
            l1_loss = l1_criterion(gen_lr, imgs_lr.cpu())
            lr_similarity.extend(list(l1_loss.numpy().mean((1,2,3))))

            # high resolution image L1 metric
            hr_similarity.extend(list(l1_criterion(gen_hr, imgs_hr).numpy().mean((1,2,3))))

            
            # energy distribution
            for i in range(len(imgs_lr)):                
                gen_nnz = gen_hr[i][gen_hr[i] > 0]
                if len(gen_nnz) > 0:
                    real_nnz = imgs_hr[i][imgs_hr[i] > 0]
                    t_min = torch.min(torch.cat((gen_nnz, real_nnz), 0)).item()
                    t_max = torch.max(torch.cat((gen_nnz, real_nnz), 0)).item()
                    gen_hist = torch.histc(gen_nnz, bins, min=t_min, max=t_max).float()
                    real_hist = torch.histc(real_nnz, bins, min=t_min, max=t_max).float()
                    energy_dist.append(l2_criterion(gen_hist, real_hist).item())

                # non-zero pixels
                real_amount_nnz = (imgs_hr.numpy()>0).sum((1,2,3))
                pred_amount_nnz = (gen_hr.numpy()>0).sum((1,2,3))
                nnz.extend(np.abs(real_amount_nnz-pred_amount_nnz))

    results = {}
    for metric_name, metric_values in zip(['hr_l1', 'lr_l1', 'energy distribution', 'non-zero'], [lr_similarity, hr_similarity, energy_dist, nnz]):
        results[metric_name] = {'mean': float(np.mean(metric_values)), 'std': float(np.std(metric_values))}

    return results


def to_hist(data, bins):
    '''nearest neighbor interpolation for 1d numpy arrays'''
    hist = np.zeros(2*len(data))
    hist[::2] = data.copy()
    hist[1::2] = data.copy()
    x = np.vstack((bins, bins)).flatten('F')
    return x[1:-1], hist


def distribution(dataset_path, dataset_type, generator, device, output_path=None, 
                                batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=5000, mode='max'):
    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu
    )

    hhd = MultHist(2 if "mean" in mode else 3, mode)
    print('collecting data from %s' % dataset_path)
    for _, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Configure model input
        imgs_lr = imgs["lr"].to(device)
        imgs_hr = imgs["hr"]
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr).detach()
        if "mean" in mode:
            hhd.append(gen_hr, imgs_hr)
        else:
            hhd.append(gen_hr, imgs_hr, imgs_lr)

    plt.figure()
    for i, (ls, lab) in enumerate(zip(['-','--','-.'], ["model prediction", "ground truth", "low resolution input"])):
        if "mean" in mode and i == 2:
            continue
        x,y = to_hist(*hhd.histogram(hhd.list[i], bins))
        plt.plot(x, y, ls, label=lab)
        std = np.sqrt(y)
        std[y==0] = 0
        plt.fill_between(x, y+std, y-std ,alpha=.2)
    
    
    #plt.title('Highest energy distribution' if mode == 'max' else r'Amount of nonzero pixels $\geq 2\cdot 10^{-2}$')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    plt.show()

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
        if key not in ('results','validation','binedges'):
            print(key, '\t'*(2*tmax-ts(key)), value)
    hline()
    val=False

    if 'results' in results:
        hyper_set = results['results']
        num_lines = len(hyper_set)
        assert num_lines > 0
        for i in range(num_lines):
            if 'metrics' in hyper_set[i].keys():
                p0 = hyper_set[i]['metrics'][0]
                break
    else: #plot validation data from info.json generated by esrgan.py
        hyper_set = results['validation']
        num_lines = 1
        p0 = hyper_set[0]
        val=True

    max_lines_per_plot = 6
    num_metrics = len(p0)-2
    N = num_metrics #int(np.sqrt(num_metrics))
    M = int(num_lines%max_lines_per_plot!=0)+num_lines//max_lines_per_plot
    f, ax = plt.subplots(M, N)
    ax = ax.flatten()
    max_lines_per_plot = num_lines//M
    set_indices=[i if i < num_lines else num_lines for i in range(0,num_lines+max_lines_per_plot,max_lines_per_plot)]
    # iterate over every metric that was measured
    for l in range(M):
        for m, (key, value) in enumerate(p0.items()):
            if key in ('epoch', 'batch'):
                m -= 1
                continue
            splt = ax[m+N*l]
            if l==0:
                splt.set_title(key)
            if l ==M-1:
                splt.set_xlabel('iterations')
            # iterate over every set of hyperparameters that was investigated
            for h in range(set_indices[l],set_indices[l+1]):
                if not 'metrics' in hyper_set[h].keys() and not val:
                    continue
                checkpoints = hyper_set if val else hyper_set[h]['metrics'] 
                label = ''
                if not val:
                    for k, v in hyper_set[h].items():
                        if k not in ('epoch', 'batch', 'metrics'):
                            vstr = str(v)
                            if len(vstr) > 6:
                                vstr = '%.4f' % v if v > 1e-3 else '%.2e' % v
                            label += '%s=%s ' % (clean_labels(k), vstr)
                iterations = []  # will contain batch number
                y = []
                y_err = []
                # iterate over every point in time that was measured
                for i in range(len(checkpoints)):
                    p = checkpoints[i] # point in time
                    iterations.append(p['batch'])
                    y.append(p[key]['mean'])
                    y_err.append(p[key]['std'])
                _, caps, bars = splt.errorbar(iterations, y, yerr=y_err, label=label, capsize=1.5)
                # loop through bars and caps and set the alpha value
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
        if not val:
            splt.legend(loc=(1, 0))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to image")
    parser.add_argument("--output_path", type=str, default='images/outputs', help="Path where output will be saved")
    parser.add_argument("--checkpoint_model", type=str, default=None, help="Path to checkpoint model")
    parser.add_argument("--residual_blocks", type=int, default=10, help="Number of residual blocks in G")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during evaluation")
    parser.add_argument("--factor", type=int, default=default_dict['factor'], help="factor to super resolve (multiple of 2)")
    parser.add_argument("--hr_height", type=int, default=default_dict['hr_height'], help="input image height")
    parser.add_argument("--hr_width", type=int, default=default_dict['hr_width'], help="input image width")
    parser.add_argument("-r", "--hyper_results", type=str, default=None, help="if used, show hyperparameter search results")
    parser.add_argument("--histogram", choices=['max','nnz', 'min', 'mean', 'sum', 'meannnz'], default=None, help="what histogram to show if any")
    parser.add_argument("--bins", type=int, default=30, help="number of bins in the histogram")

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
