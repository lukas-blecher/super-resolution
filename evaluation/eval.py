import sys
import os
# add home directory to pythonpath
if os.path.split(os.path.abspath('.'))[1] == 'super-resolution':
    sys.path.insert(0, '.')
else:
    sys.path.insert(0, '..')
from datasets import *
from utils import *
from torch.utils.data import DataLoader
from models import GeneratorRRDB, NaiveGenerator
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
from scipy.special import legendre
import scipy.ndimage as ndimage

show = False

total_entries = 0
top_veto_real = 0
top_veto_gen = 0
w_veto_real = 0
w_veto_gen = 0
gpu = 0


def FWM(arr, l=1, j=2, etarange=1., phirange=1., power=.5):  # arr: input image batch, l: # of FWM to take, j: up to which const to consider
    img = np.squeeze(arr)
    bins = img.shape[1]
    leg = legendre(l)
    idx = np.argsort(img.reshape(len(img), -1))[:, -j:][:, ::-1]
    etabin, phibin = np.array(np.unravel_index(idx, img.shape)[1:])
    eta = etabin*2*etarange/bins-etarange
    phi = phibin*2*phirange/bins-phirange
    theta = 2*np.arctan(np.exp(-eta))  # formula for theta(eta)
    ptabs = np.abs(img[np.repeat(np.arange(len(img)), j).reshape(-1, j), etabin, phibin])**power

    ptsum = np.sum(ptabs, axis=1)
    dtheta = theta[:, :, None]-theta[:, None, :]
    stheta = theta[:, :, None]+theta[:, None, :]
    dphi = phi[:, :, None]-phi[:, None, :]
    cos = 0.5*(np.cos(dtheta)-np.cos(stheta))*np.cos(dphi) + 0.5*(np.cos(dtheta)+np.cos(stheta))  # from Anja's bach.
    term = ptabs[:, :, None]*ptabs[:, None, :]/(ptsum[:, None, None]**2)*leg(cos)
    return term.sum((1, 2)).tolist()


def get_nth_hardest(arr, n=1):
    temp = np.squeeze(arr)
    ls = []
    for i in range(temp.shape[0]):
        tm = temp[i, :, :]
        st = np.sort(tm, axis=None)
        ls.append(st[-n])
    return ls


def get_const_ratio(arr, n=1, m=2):
    temp = np.squeeze(arr)
    ls = []
    for i in range(temp.shape[0]):
        tm = temp[i, :, :]
        st = np.sort(tm, axis=None)
        if st[-m] == 0:
            raise ValueError('DIVIDED BY ZERO')
        ls.append(st[-n]/st[-m])
    return ls


def w_hist(info, i=0):
    from pyjet import cluster
    dtype = np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')])
    w_hist = []

    global total_entries
    global top_veto_real
    global top_veto_gen
    global w_veto_real
    global w_veto_gen

    # extract w masses for each picture in batch:
    for k in range(len(info)):
        total_entries += 1
        jet = np.array(info[k], dtype=dtype)
        sequence = cluster(jet, R=0.8, p=1)  # use kt-alg with a jet radius of 0.8 (same as top sample generation)
        jet = sequence.inclusive_jets()
        jet_2subs = sequence.exclusive_jets(2)  # go back in clustering history to where there have been 2 subjets
        # check if top mass is reasonable:
        if (jet[0].mass > 140. and jet[0].mass < 200.):
            w_hist.append(float(jet_2subs[0].mass))  # assume W coincides with hardest subjet, not always true but negligible
        else:
            if i == 0:
                top_veto_gen += 1
            if i == 1:
                top_veto_real += 1

    # throw away unreasonable W masses
    w_hist_cut = [entry for entry in w_hist if entry > 50. and entry < 110.]

    outliers = len(w_hist) - len(w_hist_cut)
    if i == 0:
        w_veto_gen += outliers
    if i == 1:
        w_veto_real += outliers

    print(outliers, i)

    return w_hist_cut


class extract_const:
    def __init__(self, img_array, etarange=1., phirange=1.):  # ranges are fixed for current images
        self.img_array = np.squeeze(img_array)
        self.bins = self.img_array.shape[1]
        self.etarange = etarange
        self.phirange = phirange
        self.batchsize = self.img_array.shape[0]
        self.ls = [[] for i in range(self.batchsize)]
      #  self.out=[() for i in range(self.batchsize)]
        for i in range(0, self.img_array.shape[0]):

            for j in range(0, self.img_array.shape[1]):

                for k in range(0, self.img_array.shape[2]):

                    if self.img_array[i, j, k] != 0:
                        eta = self.calc_eta(etabin=j)
                        phi = self.calc_phi(phibin=k)
                        pt = self.img_array[i, j, k]
                        self.ls[i].append((pt, eta, phi, 0))
       # undo binning

    def calc_eta(self, etabin=0):
        eta_new = float(etabin*2*self.etarange/self.bins-self.etarange)
        return eta_new

    def calc_phi(self, phibin=0):
        phi_new = float(phibin*2*self.phirange/self.bins-self.phirange)
        return phi_new


def delta_r(arr, n=1, m=2, etarange=1., phirange=1.):
    img = np.squeeze(arr)
    bins = arr.shape[1]
    ls = []
    for i in range(img.shape[0]):
        tmp = img[i]
        n_args = np.where(tmp == np.sort(tmp.flatten())[-n])
        m_args = np.where(tmp == np.sort(tmp.flatten())[-m])
        n_etabin, n_phibin = n_args
        m_etabin, m_phibin = m_args
        eta_n = float(n_etabin*2*etarange/bins-etarange)
        eta_m = float(m_etabin*2*etarange/bins-etarange)
        phi_n = float(n_phibin*2*phirange/bins-phirange)
        phi_m = float(m_phibin*2*phirange/bins-phirange)
        deta = eta_n - eta_m
        dphi = phi_n - phi_m
        dr = np.sqrt(deta**2 + dphi**2)
        ls.append(dr)
    return ls


class MultHist:
    '''A class to collect data for any number of histograms
    modes:  'max' collects the maximum value for each image, 'min' collects the minimum value, 'mean' collects the mean value, 'nnz' saves the amount of nonzero values,
            'sum' collects the total energy for each image, 'meannnz' saves the mean energy for each image disregarding the empty pixels, 'wmass' extracts a prediction 
                  for the w mass from each image, taken to be the hardest subjet in the clustering history when there have been 2 subjets; unreasonable top and w mass predictions are vetoed,
            'E' will extract the total energy distribution
            'E_n' plots the distribution of the n-th hardest constituent NOTE: start from 1 , not 0
            'R_nm' plots the ratio of the nth and mth hardest consts, eg R_25 for ratio of second and fifth hardest NOTE: atm n AND m have to be smaller 10 ie 1 digit
            'deltaR_nm' plots deltaR distribution for nth and mth hardest constituents
            'hitogram' returns the gives insight in how the constituents are distributed in the super resolved image
            'FWM_i_j' plots the fox wolfram moments for the ith hardest constituent with respect to the 1...j hardest.
            'meanimg' returns the mean of the hr and sr images
 '''

    def __init__(self, num, mode='max', factor=None, **kwargs):
        self.num = num
        self.list = [[] for _ in range(num)]
        self.mode = mode
        self.title, self.xlabel, self.ylabel = '', r'Energy [\sqrt{GeV}]', 'Entries'
        self.thres = 0.002
        self.inpl = '0'
        self.ratio = '0'
        if 'E_' in self.mode:
            self.inpl = self.mode[2:]
            self.title = 'Energy of the ' + num_to_str(int(self.inpl), latex=(kwargs['pdf'] if 'pdf' in kwargs else 0)) + 'hardest constituent'
        elif 'deltaR_' in self.mode:
            self.dr = self.mode[7:]
            self.dr1 = self.mode[7]
            self.dr2 = self.mode[8]
        elif 'R_' in self.mode:
            self.ratio = self.mode[2:]
            self.harder = self.mode[2]
            self.softer = self.mode[3]
        elif self.mode == 'hitogram':
            self.raster = SumRaster(factor)
        elif self.mode == 'meanimg':
            self.preprocess = False
            try:
                if opt.preprocessing:
                    self.preprocess = True
            except NameError:
                self.preprocess = False
            self.meanimg = MeanImage(factor, preprocess=self.preprocess)
        elif 'FWM' in self.mode:
            self.l, self.j = [int(s) for s in self.mode[4:].split('_')]

    def append(self, *argv):
        assert len(argv) == self.num

        for i, L in enumerate(argv):
            Ln = L.detach().cpu().numpy()
            if (Ln != Ln).any():
                # check if the image contains Nans
                continue
            try:
                if self.mode == 'max':
                    self.list[i].extend(list(Ln.max((1, 2, 3))))
                elif self.mode == 'min':
                    self.list[i].extend(list(Ln.min((1, 2, 3))))
                elif self.mode == 'mean':
                    self.list[i].extend(list(Ln.mean((1, 2, 3))))
                elif self.mode == 'nnz':
                    self.list[i].extend(list((Ln > self.thres).sum((1, 2, 3))))
                elif self.mode == 'sum':
                    self.list[i].extend(list(Ln.sum((1, 2, 3))))
                elif self.mode == 'meannnz':
                    for j in range(len(Ln)):
                        nnz = Ln[j][Ln[j] > self.thres]
                        if len(nnz) == 0:
                            self.list[i].append(0)
                            continue
                        self.list[i].append(np.nan_to_num(nnz.mean(), 0))
                elif self.mode == 'wmass':
                    self.list[i].extend(w_hist(extract_const(Ln).ls, i))
                elif self.mode == 'E':
                    nnz = Ln[Ln > self.thres]
                    if len(nnz) == 0:
                        self.list[i].append(0)
                        continue
                    self.list[i].extend(list(nnz.flatten()))
                elif 'E_' in self.mode:
                    self.list[i].extend(get_nth_hardest(Ln, n=int(self.inpl)))
                elif 'deltaR_' in self.mode:
                    self.list[i].extend(delta_r(Ln, int(self.dr1), int(self.dr2)))
                elif 'R_' in self.mode:
                    self.list[i].extend(get_const_ratio(Ln, n=int(self.harder), m=int(self.softer)))
                elif 'FWM_' in self.mode:
                    self.list[i].extend(FWM(Ln, l=self.l, j=self.j))

            except Exception as e:
                print('Exception while adding to MultHist with mode %s' % self.mode, e)

        if self.mode == 'hitogram':
            self.raster.add(*[T.detach().cpu() for T in argv])
        elif self.mode == 'meanimg':
            self.meanimg.add(*[T.detach().cpu().numpy() for T in argv])

    def get_range(self):
        mins = [min(self.list[i]) for i in range(self.num)]
        maxs = [max(self.list[i]) for i in range(self.num)]
        return min(mins), max(maxs)

    def max(self, threshold=.9, power=.3):
        '''Function introduced for total energy distribution'''
        MAX = 0
        for i in range(self.num):
            pl = np.array(self.list[i])**power
            c, b = np.histogram(pl, 100)
            e_max = b[(np.cumsum(c) > len(pl)*threshold).argmax()]
            if e_max > MAX:
                MAX = e_max
        return MAX

    def histogram(self, L, bins=10, auto_range=True):
        if auto_range:
            if self.mode == 'E' or ('R' in self.mode and 'deltaR' not in self.mode):
                power = .5
                return np.histogram(np.array(L)**power, bins, range=(self.get_range()[0], self.max(power=power)))

            else:
                return np.histogram(L, bins, range=self.get_range())
        else:
            return np.histogram(L, bins)



class MultModeHist:
    def __init__(self, modes, num='standard', factor=default.factor, **kwargs):
        self.modes = modes
        self.standard_nums = {'max': 3, 'min': 3, 'nnz': 3, 'mean': 2, 'meannnz': 2, 'wmass': 2, 'E': 2, 'hitogram': 2, 'meanimg': 2}
        self.hist = []
        self.nums = [num] * len(self.modes) if num != 'standard' else [self.standard_nums[mode] if '_' not in mode else 3 for mode in self.modes]
        for i in range(len(self.modes)):
            self.hist.append(MultHist(self.nums[i], modes[i], factor, **kwargs))

    def append(self, *argv):
        for i in range(len(self.hist)):
            self.hist[i].append(*argv[:self.nums[i]])

    def __getitem__(self, item):
        return self.hist[item]


def call_func(opt):
    global gpu
    device = torch.device('cuda:%i' % gpu if torch.cuda.is_available() else 'cpu')
    dopt = dir(opt)
    output_path = opt.output_path if 'output_path' in dopt else None
    bins = opt.bins if 'bins' in dopt else default.bins

    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(np.log2(opt.factor)), power=opt.scaling_power, res_scale=opt.res_scale).to(device)
    generator.load_state_dict(torch.load(opt.checkpoint_model, map_location=device))
    if opt.E_thres:
        generator.thres = opt.E_thres
    if 'naive_generator' in dopt:
        if opt.naive_generator:
            generator = NaiveGenerator(int(np.log2(opt.factor)))
    args = [opt.dataset_path, opt.dataset_type, generator, device, output_path, opt.batch_size, opt.n_cpu, bins, opt.hr_height, opt.hr_width,
            opt.factor, opt.amount, opt.pre_factor, opt.E_thres, opt.n_hardest]
    if opt.histogram:
        return distribution(*args, **opt.kwargs)
    else:
        return calculate_metrics(*args)


def calculate_metrics(dataset_path, dataset_type, generator, device, output_path=None, batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=None, pre=1, thres=None, N=None):
    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount, pre, thres, N)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu
    )

    emd, lr_similarity, hr_similarity = [], [], []
    l1_criterion = nn.L1Loss(reduction='none')
    pool = SumPool2d(factor)
    Ethres = thres if thres is not None else 0
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
            lr_similarity.extend(list(l1_loss.numpy().mean((1, 2, 3))))

            # high resolution image L1 metric
            hr_similarity.extend(list(l1_criterion(gen_hr, imgs_hr).numpy().mean((1, 2, 3))))

            # Energy moving distance
            emd.extend(get_emd(gen_hr.numpy(), imgs_hr.numpy(), thres=Ethres))

    results = {}
    for metric_name, metric_values in zip(['hr_l1', 'lr_l1', 'emd'], [lr_similarity, hr_similarity, emd]):
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
                 batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=5000, pre=1, thres=None, N=None, mode='max', **kwargs):

    statement = Wrapper(output_path)
    pdf=False
    if output_path:
        if 'pdf' in kwargs and kwargs['pdf']:
            pdf=True
            from matplotlib.backends.backend_pdf import PdfPages
            statement = PdfPages(output_path.replace('.png', '')+'.pdf')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=(kwargs['fontsize'] if 'fontsize' in kwargs else 12))
            plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount, pre, thres, N)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    modes = mode if type(mode) is list else [mode]
    hhd = MultModeHist(modes, factor=factor, pdf=pdf)
    print('collecting data from %s' % dataset_path)
    for _, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            # Configure model input
            imgs_lr = imgs["lr"].to(device)
            imgs_hr = imgs["hr"]
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr).detach()
            hhd.append(gen_hr, imgs_hr, imgs_lr)
    if 'wmass' in modes:
        print('total entries: ', total_entries)
        print('top veto real / gen:', top_veto_real, top_veto_gen)
        print('w veto real / gen: ', w_veto_real, w_veto_gen)
        entries_gen = len(hhd[0].list[0])
        entries_real = len(hhd[0].list[1])
        print('hist entries real / gen: ', entries_real, entries_gen)
    global show
    total_kld = []
    kldiv = nn.KLDivLoss(reduction='sum')
    with statement as output:
        for m in range(len(modes)):
            # check for hitogram and mean image
            if modes[m] in ('hitogram', 'meanimg'):
                if modes[m] == 'hitogram':
                    sr, hr = hhd[m].raster.get_hist()
                    f = plot_hist2d(sr, hr)
                elif modes[m] == 'meanimg':
                    sr, hr = hhd[m].meanimg.get_hist()
                    f = plot_mean(hhd[m].meanimg)
                sr, hr = .1+torch.Tensor(sr)[None, None, ...], .1+torch.Tensor(hr)[None, None, ...]
                if output_path:
                    if type(output) == str:
                        f.savefig((output_path+modes[m]).replace(".png", ""))
                    else:
                        plt.savefig(output, format='pdf')

                total_kld.append(float(kldiv((sr/sr.sum()).log(), hr/(sr.sum()))))
                continue

            plt.figure()
            bin_entries = []
            for i, (ls, lab) in enumerate(zip(['-', '--', '-.'], ["model prediction", "ground truth", "low resolution input"])):
                if hhd.nums[m] == i:
                    continue
                try:
                    entries, binedges = hhd[m].histogram(hhd[m].list[i], bins)
                    if i < 2:
                        bin_entries.append(entries)
                except ValueError:
                    print('auto range failed for %s' % modes[m])
                    print(hhd[m].list[i])
                    entries, binedges = hhd[m].histogram(hhd[m].list[i], bins, auto_range=False)
                x, y = to_hist(entries, binedges)
                plt.plot(x, y, ls, label=lab)
                std = np.sqrt(y)
                std[y == 0] = 0
                plt.fill_between(x, y+std, y-std, alpha=.2)
            if hhd.nums[m] >= 2 and len(bin_entries) == 2:
                KLDiv = KLD_hist(torch.Tensor(binedges))
                total_kld.append(float(KLDiv(torch.Tensor(bin_entries[0]), torch.Tensor(bin_entries[1])).item()))
            plt.title(hhd[m].title)
            plt.xlabel(hhd[m].xlabel)
            plt.ylabel(hhd[m].ylabel)
            plt.legend()
            if output_path:
                if type(output) == str:
                    out_path = output_path + ('_' + modes[m] if len(modes) > 1 else '')
                    plt.savefig(out_path.replace('.png', ''))
                else:
                    plt.savefig(output, format='pdf')
            if show:
                plt.show()
        plt.close('all')
    return total_kld


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
        if key not in ('results', 'validation', 'binedges', 'loss', 'eval_results'):
            print(key, '\t'*(2*tmax-ts(key)), value)
    hline()
    val = False

    if 'results' in results:
        hyper_set = results['results']
        num_lines = len(hyper_set)
        assert num_lines > 0
        for i in range(num_lines):
            if 'metrics' in hyper_set[i].keys():
                p0 = hyper_set[i]['metrics'][0]
                break
    else:  # plot validation data from info.json generated by esrgan.py
        hyper_set = results['validation']
        num_lines = 1
        p0 = hyper_set[0]
        val = True

    max_lines_per_plot = 6
    num_metrics = len(p0)-2
    N = num_metrics  # int(np.sqrt(num_metrics))
    M = int(num_lines % max_lines_per_plot != 0)+num_lines//max_lines_per_plot
    f, ax = plt.subplots(M, N, sharex=True)
    for m in range(1, M):
        for n in range(N):
            ax[0, n]._shared_y_axes.join(ax[0, n], ax[m, n])
    ax = ax.flatten()
    max_lines_per_plot = num_lines//M
    set_indices = [i if i < num_lines else num_lines for i in range(0, num_lines+max_lines_per_plot, max_lines_per_plot)]
    # iterate over every metric that was measured
    for l in range(M):
        for m, (key, value) in enumerate(p0.items()):
            if key in ('epoch', 'batch'):
                m -= 1
                continue
            splt = ax[m+N*l]
            if l == 0:
                splt.set_title(key)
            if l == M-1:
                splt.set_xlabel('iterations')
            # iterate over every set of hyperparameters that was investigated
            for h in range(set_indices[l], set_indices[l+1]):
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
                    p = checkpoints[i]  # point in time
                    iterations.append(p['batch'])
                    y.append(p[key]['mean'])
                    y_err.append(p[key]['std'])
                _, caps, bars = splt.errorbar(iterations, y, yerr=y_err, label=label, capsize=1.5)
                # loop through bars and caps and set the alpha value
                [bar.set_alpha(0.5) for bar in bars]
                [cap.set_alpha(0.5) for cap in caps]
        if not val:
            splt.legend(loc=(1, 0))
    global show
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path", type=str, default=None, help="Path to image")
    parser.add_argument("-o", "--output_path", type=str, default='images/outputs', help="Path where output will be saved")
    parser.add_argument("-m", "--checkpoint_model", type=str, default=None, help="Path to checkpoint model")
    parser.add_argument("--residual_blocks", type=int, default=10, help="Number of residual blocks in G")
    parser.add_argument("-b", "--batch_size", type=int, default=30, help="Batch size during evaluation")
    parser.add_argument("-f", "--factor", type=int, default=default_dict['factor'], help="factor to super resolve (multiple of 2)")
    parser.add_argument("--pre_factor", type=int, default=1, help="factor to downsample images before giving it to the model")
    parser.add_argument("-N", "--amount", type=int, default=None, help="amount of test samples to use. Standard: All")
    parser.add_argument("--hr_height", type=int, default=default_dict['hr_height'], help="input image height")
    parser.add_argument("--hr_width", type=int, default=default_dict['hr_width'], help="input image width")
    parser.add_argument("--hw", type=int, default=None, nargs='+', help="specify image height and width at once")
    parser.add_argument("-r", "--hyper_results", type=str, default=None, help="if used, show hyperparameter search results")
    parser.add_argument("--histogram", nargs="+", default=None, help="what histogram to show if any")
    parser.add_argument("--bins", type=int, default=30, help="number of bins in the histogram")
    parser.add_argument("--naive_generator", action="store_true", help="use a naive upsampler")
    parser.add_argument("--no_show", action="store_false", help="don't show figure")
    parser.add_argument("--scaling_power", type=float, default=1, help="power to which to raise the input image pixelwise")
    parser.add_argument("--n_hardest", type=int, default=None, help="how many of the hardest constituents should be in the ground truth")
    parser.add_argument("--E_thres", type=float, default=None, help="Energy threshold for the ground truth and the generator")
    parser.add_argument("--res_scale", type=float, default=default.res_scale, help="Residual weighting factor")
    parser.add_argument("--preprocessing", action="store_true", help="preprocess pictures used for meanimg")
    parser.add_argument("--pdf", type=str_to_bool, default=True, help="wheter to save the figures as pdf files and tex files")
    parser.add_argument("--fontsize", default=12, type=float)
    opt = parser.parse_args()
    if opt.hw is not None and len(opt.hw) == 2:
        opt.hr_height, opt.hr_width = opt.hw
    opt = vars(opt)
    opt['kwargs'] = {'pdf': opt['pdf'], 'mode': opt['histogram'], 'fontsize': opt['fontsize']}
    try:
        gpu = get_gpu_index()
        num_gpus = torch.cuda.device_count()
        if gpu >= num_gpus:
            gpu = np.random.randint(num_gpus)
        print('running on gpu index {}'.format(gpu))
    except Exception:
        pass
    show = opt['no_show']
    if opt['hyper_results'] is not None:
        evaluate_results(opt['hyper_results'])
    else:
        if opt['dataset_path'] is None or (opt['checkpoint_model'] is None and not opt['naive_generator']):
            raise ValueError("For evaluation dataset_path and checkpoint_model are required")
        arguments = {**opt, **{key: default_dict[key] for key in default_dict if key not in opt}}
        opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
        # print(opt)

        out = call_func(opt)
        if out is not None:
            print(out)
