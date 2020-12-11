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
import matplotlib.patches as mpatches
from scipy.stats import wasserstein_distance

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

show = False
normh = False
savehito = False
split_meanimg = False

total_entries = 0
top_veto_real = 0
top_veto_gen = 0
w_veto_real = 0
w_veto_gen = 0
gpu = 0

def JetMass(arr, etarange=1., phirange=1.):
    img = np.squeeze(arr)
    bins = img.shape[1]
    jetMassList = []

    # loop over batch, etabins, phibins
    for pic in range(img.shape[0]):
        ls_E, ls_px, ls_py, ls_pz = [], [], [], []
        for etabin in range(bins):
            for phibin in range(bins):
                if img[pic, etabin, phibin] != 0:
                    #get eta, phi, pt
                    eta = etabin*2*etarange/bins-etarange + etarange/bins
                    phi = phibin*2*phirange/bins-phirange + phirange/bins
                    pt = img[pic, etabin, phibin]

                    # convert to (E, px, py, pz)
                    E, px, py, pz = PtEtaPhiM_to_EPxPyPz(pt, eta, phi, 0)
                    ls_E.append(E)
                    ls_px.append(px)
                    ls_py.append(py)
                    ls_pz.append(pz)

        #calculate jet mass as sum over four momenta squared
        E_jet = sum(ls_E)
        px_jet = sum(ls_px)
        py_jet = sum(ls_py)
        pz_jet = sum(ls_pz)
        M_2 = E_jet**2 - px_jet**2 - py_jet**2 - pz_jet**2
        jetMassList.append(np.sqrt(np.abs(M_2)))

    return jetMassList


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
        # self.out=[() for i in range(self.batchsize)]
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
        eta_new = float(etabin*2*self.etarange/self.bins-self.etarange + self.etarange/self.bins)
        return eta_new

    def calc_phi(self, phibin=0):
        phi_new = float(phibin*2*self.phirange/self.bins-self.phirange + self.etarange/self.bins)
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
        eta_n = float(n_etabin*2*etarange/bins-etarange + etarange/bins)
        eta_m = float(m_etabin*2*etarange/bins-etarange + etarange/bins)
        phi_n = float(n_phibin*2*phirange/bins-phirange+phirange/bins)
        phi_m = float(m_phibin*2*phirange/bins-phirange+phirange/bins)
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
            'E' will extract the total energy nntion
            'E_n' plots the distribution of the n-th hardest constituent NOTE: start from 1 , not 0
                'corr_' in front will plot the correlation between HR and SR (also a slice plot will be made)
                '_lr' after will plot the correlation between all combinations of HR, LR and SR
            'R_nm' plots the ratio of the nth and mth hardest consts, eg R_25 for ratio of second and fifth hardest NOTE: atm n AND m have to be smaller 10 ie 1 digit
            'deltaR_nm' plots deltaR distribution for nth and mth hardest constituents
            'hitogram' returns the gives insight in how the constituents are distributed in the super resolved image
            'FWM_i_j' plots the fox wolfram moments for the ith hardest constituent with respect to the 1...j hardest.
            'meanimg' returns the mean of the hr and sr images
            'nsubj_i_j' plots ratio of the N-subjettiness for N=i and N'=j
            'jetmass' plot high level invariant mass of images
            'w_pf' plots the jet observable w_pf ("girth")
            'C_0_n' plots two point pT correlator to power n
            
    additions:
            'corr_' will plot the correlations between HR and SR in a 2d histogram aswell as in a slice plot
                '_lr' will additionally plot the LR - HR/SR correlations in a 2d histogram        
            'ratio_' plots |E_sr-E_gt|/E_gt for the nth hardest constituent
 '''

    def __init__(self, num, mode='max', factor=None, **kwargs):
        self.num = num
        self.list = [[] for _ in range(num)]
        self.mode = mode
        self.thres = 0.002
        self.display_ratio = 1
        if 'threshold' in kwargs:
            self.thres = float(kwargs['threshold'])
        self.power = 1 
        if self.mode == 'E' or ('R' in self.mode and 'deltaR' not in self.mode) or 'E_' == self.mode[:2]:
            self.power = .5
            self.display_ratio=0.98
        if 'power' in kwargs:
            if kwargs['power'] is not None:
                self.power = kwargs['power']
        latex = (kwargs['pdf'] if 'pdf' in kwargs else 0)
        self.title, self.xlabel, self.ylabel = '', 'E [GeV]', 'Entries'
        if self.power==.5 and latex:
            self.xlabel = r'E [$\sqrt{\text{GeV}}$]' 
        elif self.power!=1:
            self.xlabel = 'E [GeV$^{%s}$]'%self.power
        
        if 'E_' in self.mode:
            self.inpl = self.mode[2:]
            self.title = 'Energy of the ' + num_to_str(int(self.inpl), thres=1, latex=latex) + 'hardest constituent'
        elif 'deltaR_' in self.mode:
            self.dr = self.mode[7:]
            self.dr1 = self.mode[7]
            self.dr2 = self.mode[8]
        elif 'R_' in self.mode:
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
            energy = False
            if 'meanenergy' in kwargs:
                energy = kwargs['meanenergy']
            self.meanimg = MeanImage(factor, preprocess=self.preprocess, threshold=self.thres, energy=energy)
        elif 'FWM' in self.mode:
            self.l, self.j = [int(s) for s in self.mode[4:].split('_')]
            self.title = num_to_str(int(self.l), thres=0, latex=latex) + 'Fox Wolfram Moment ('+num_to_str(int(self.j), thres=0, latex=latex)+'constituents)'
        elif 'nsubj' in self.mode:
            self.n, self.m = [int(s) for s in self.mode.split('nsubj_')[1].split('_')]
            self.xlabel = r'$\tau_{%i}/\tau_{%i}$' % (self.n, self.m)
            self.title = "Ratio of N-subjettiness for "+"\n"+"$N=%i$ and $N'=%i$" % (self.n, self.m)
            self.display_ratio=0.92
            self.power=0.999
        elif self.mode == 'meannnz':
            self.title = 'Mean energy per constituent'
            self.display_ratio=.95
        elif self.mode == 'nnz':
            self.xlabel = 'Constituents'
            self.title = 'Number of constituents'
        elif self.mode == 'jetmass':
            self.title = 'Invariant Jet Mass'
            self.xlabel = r'$m_{jet} [\text{GeV}]$'
        elif self.mode == 'w_pf':
            self.xlabel = r'$w_{PF}$'
            self.title = r'$w_{PF}$'
            self.display_ratio=0.92
            self.power=0.999
        elif 'C_0_' in self.mode:
            self.n = int(self.mode.split('C_0_')[1])
            print('n for 2p correlator: ', self.n)
            self.xlabel = r'$C_{0.%i}$' % (self.n)
            self.title = self.xlabel
            self.display_ratio=0.92
            self.power=0.999

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
                elif 'nsubj' in self.mode:
                    np.seterr(divide='raise')
                    try:
                        event=[img2event(Ln[k], thres=self.thres) for k in range(len(Ln))]
                        self.list[i].extend([nsubjettiness(event[k], self.n)/nsubjettiness(event[k], self.m) for k in range(len(Ln)) if (nsubjettiness(event[k], self.n) is not None and nsubjettiness(event[k], self.m) is not None)])
                    except FloatingPointError:
                        continue

                elif self.mode == 'jetmass':
                    self.list[i].extend(JetMass(Ln))
                elif self.mode == 'w_pf':
                    event=[img2event(Ln[k], thres=self.thres) for k in range(len(Ln))]
                    self.list[i].extend([w_pf(event[k]) for k in range(len(Ln))])
                elif 'C_0_' in self.mode:
                    event=[img2event(Ln[k], thres=self.thres) for k in range(len(Ln))]
                    self.list[i].extend([C_0_n(event[k], self.n) for k in range(len(Ln))])

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

    def max(self, threshold=None):
        '''Function introduced for total energy distribution'''
        MAX = 0
        if threshold is None:
            threshold=self.display_ratio
        if threshold < 1:
            for i in range(self.num):
                pl = np.array(self.list[i])**self.power
                c, b = np.histogram(pl, 100)
                e_max = b[(np.cumsum(c) > len(pl)*threshold).argmax()]
                if e_max > MAX:
                    MAX = e_max
        else:
            MAX = max([max(self.list[i]) for i in range(self.num)])**self.power
        return MAX

    def histogram(self, L, bins=10, auto_range=True, threshold=None):
        if auto_range:
            if self.power != 1:
                return np.histogram(np.array(L)**self.power, bins, range=(self.get_range()[0]**self.power, self.max(threshold=threshold)))
            else:
                return np.histogram(L, bins, range=self.get_range())
        else:
            return np.histogram(L, bins)


class MultModeHist:
    def __init__(self, modes, num='standard', factor=default.factor, **kwargs):
        self.modes = modes
        self.standard_nums = {'max': 3, 'min': 3, 'nnz': 4, 'mean': 2, 'meannnz': 4, 'wmass': 2, 'E': 2, 'hitogram': 2, 'meanimg': 2, 'jetmass': 4}
        self.hist = []
        self.nums = [num] * len(self.modes) if num != 'standard' else [self.standard_nums[self.rmAdditions(mode)]
                                                                       if '_' not in self.rmAdditions(mode) else 4 for mode in self.modes]
        for i in range(len(self.modes)):
            self.hist.append(MultHist(self.nums[i], self.rmAdditions(modes[i]), factor, **kwargs))

    def rmAdditions(self, mode):
        return mode.replace('corr_', '').replace('_lr', '').replace('ratio_', '')

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

    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=int(np.log2(opt.factor)), res_scale=opt.res_scale, use_transposed_conv=opt.use_transposed_conv, fully_tconv_upsample=opt.fully_transposed_conv, num_final_layer_res=opt.num_final_res_blocks).to(device)
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


def calculate_metrics(dataset_path, dataset_type, generator, device, output_path=None, batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=None, pre=1, thres=None, N=None,noise_factor=None):
    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount, pre, thres, N,noise_factor=noise_factor)
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


def distribution(dataset_path, dataset_type, generator, device, output_path=None,
                 batch_size=4, n_cpu=0, bins=10, hr_height=40, hr_width=40, factor=2, amount=5000, pre=1, thres=None, N=None, mode='max',noise_factor=None, **kwargs):

    save_hhd = kwargs['save_hhd']
    load_hhd = kwargs['load_hhd']
    wasserstein = kwargs['wasserstein']
    statement = Wrapper(output_path)
    pdf = False
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if 'pdf' in kwargs and kwargs['pdf']:
            pdf = True
            from matplotlib.backends.backend_pdf import PdfPages
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif', size=(kwargs['fontsize'] if 'fontsize' in kwargs else 12))
            plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
            statement = PdfPages(output_path.replace('.png', '').replace('.pdf', '')+'.pdf')
    title,legend = True,True
    if 'title' in kwargs:
        title = kwargs['title']
    if 'legend' in kwargs:
        legend = kwargs['legend']
    generator.eval()
    dataset = get_dataset(dataset_type, dataset_path, hr_height, hr_width, factor, amount, pre, thres, N,noise_factor=noise_factor)
    if not load_hhd:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_cpu
        )
    modes = mode if type(mode) is list else [mode]
    pool = SumPool2d(factor)
    if not load_hhd:
        hhd = MultModeHist(modes, factor=factor, **kwargs)
        print('collecting data from %s' % dataset_path)
        for _, imgs in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                # Configure model input
                imgs_lr = imgs["lr"].to(device)
                imgs_hr = imgs["hr"]
                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr).detach()
                hhd.append(gen_hr, imgs_hr, imgs_lr, pool(gen_hr))
    
    ##############################
    if save_hhd and not load_hhd:
        hhd_save_str = output_path + '_hhd.pkl'
        with open(hhd_save_str, 'wb') as output:
            pickle.dump(hhd, output, pickle.HIGHEST_PROTOCOL)

    if load_hhd and not save_hhd:
        hhd_load_str = output_path + '_hhd.pkl'
        hhd = pickle.load( open( hhd_load_str, "rb" ) )

    if load_hhd and save_hhd:
        hhd_load_str = output_path + '_hhd.pkl'
        hhd_save_str = output_path + '_hhd.pkl'
        hhd = pickle.load( open( hhd_load_str, "rb" ) )
        hhd_cp = hhd
        '''
        bad fix, but changes to hhd go here...
        '''
        for m in range(len(modes)):
            if 'E_' in modes[m]:
                hhd_cp[m].xlabel = r'E [$\sqrt{\text{GeV}}$]'
            if modes[m] == 'jetmass':
                hhd_cp[m].xlabel =  r'$m_{jet} [\text{GeV}]$'
        
        with open(hhd_save_str, 'wb') as output:
            pickle.dump(hhd_cp, output, pickle.HIGHEST_PROTOCOL)
    
        print('done')
        exit()

    ##############################

    if 'wmass' in modes:
        print('total entries: ', total_entries)
        print('top veto real / gen:', top_veto_real, top_veto_gen)
        print('w veto real / gen: ', w_veto_real, w_veto_gen)
        entries_gen = len(hhd[0].list[0])
        entries_real = len(hhd[0].list[1])
        print('hist entries real / gen: ', entries_real, entries_gen)
    global show
    global normh
    global savehito
    global split_meanimg
    total_kld = []
    kld_dict = {}
    return_kld_dict = False
    if 'split_eval' in kwargs:
        if kwargs['split_eval']:
            return_kld_dict = True

    kldiv = nn.KLDivLoss(reduction='sum')

    if 'nth_jet_eval_mode' in kwargs:
        nth_jet_eval_mode = kwargs['nth_jet_eval_mode']
    else:
        nth_jet_eval_mode = 'hr'

    with statement as output:
        for m in range(len(modes)):
            # check for hitogram and mean image
            if modes[m] in ('hitogram', 'meanimg'):
                if modes[m] == 'hitogram':
                    sr, hr = hhd[m].raster.get_hist()
                    if savehito:
                        np.save((output_path+modes[m]+'_sr').replace(".npy", ""), sr)
                        np.save((output_path+modes[m]+'_hr').replace(".npy", ""), hr)
                        exit()
                    if normh:
                        gtmax = float(np.max(hr))
                        gtmin = float(np.min(hr))
                        plt.rc('font', size=20)
                        f = plot_hist2d(sr/gtmax, hr/gtmax, vmax=1.*2, vmin=gtmin/gtmax*0.5)
                    else:
                        plt.rc('font', size=20)
                        f = plot_hist2d(sr, hr)
                    #f.tight_layout()
                    if show:
                        plt.show()
                elif modes[m] == 'meanimg':
                    sr, hr = hhd[m].meanimg.get_hist()
                    f = plot_mean(hhd[m].meanimg)
                    if split_meanimg:
                        plt.rc('font', size=14)
                        f1 = plot_mean2(hhd[m].meanimg, mode=0)
                        f2 = plot_mean2(hhd[m].meanimg, mode=1)
                    #f.tight_layout()                    
                    if show:
                        plt.show()
                sr, hr = .1+torch.Tensor(sr)[None, None, ...], .1+torch.Tensor(hr)[None, None, ...]
                if output_path:
                    if not pdf:
                        f.savefig((output_path+modes[m]).replace(".png", ""))
                    else:
                        if modes[m] == 'meanimg' and split_meanimg:
                            f1.savefig(output, format='pdf',bbox_inches='tight')
                            f2.savefig(output, format='pdf',bbox_inches='tight')

                        else:
                            plt.savefig(output, format='pdf',bbox_inches='tight')
                        
                            

                total_kld.append(float(kldiv((sr/sr.sum()).log(), hr/(sr.sum()))))
                kld_dict[modes[m]] = float(kldiv((sr/sr.sum()).log(), hr/(sr.sum())))
                continue
            plt.rc('font', size=20)
            plt.rc('legend', fontsize=14)
            plt.rc('axes', labelsize=22)
            plt.rcParams['legend.title_fontsize'] = 14
            p = hhd[m].power
            unit = '[GeV$^{%s}$]' % p if p != 1 else '[GeV]'
            if p==0.5 and pdf:
                unit = r'[$\sqrt{\text{GeV}}$]' 
            plt.figure(figsize=(6.4, 4.2))
            bin_entries = []
            #for i, (ls, lab) in enumerate(zip(['-', '--', '-.','dotted'], ["model prediction", "ground truth", "low resolution input","downsampled output"])):
            for i, (ls, lab) in enumerate(zip([('black','--'), ('black','-'), ('#E50000','-'),('#E50000','--')], ["SR", "HR", "LR",r"$\mathrm{LR_{gen}}$"])):
                if hhd.nums[m] == i:
                    continue
                try:
                    try:
                        entries, binedges = hhd[m].histogram(hhd[m].list[i], bins)
                    except IndexError:
                        continue
                    if nth_jet_eval_mode == 'hr':
                        if i < 2:
                            bin_entries.append(entries)
                    elif nth_jet_eval_mode == 'lr':
                        if i > 1:
                            bin_entries.append(entries)
                    else:
                        bin_entries.append(entries)
                except ValueError as e:
                    print('auto range failed for %s' % modes[m])
                    print(e)
                    entries, binedges = hhd[m].histogram(hhd[m].list[i], bins, auto_range=False)
                x, y = to_hist(entries, binedges)
                plt.plot(x, y, color=ls[0], linestyle=ls[1], label=lab)
                std = np.sqrt(y)
                std[y == 0] = 0
                plt.fill_between(x, y+std, y-std, alpha=.2, color=ls[0])
            ######################
            # calculate wasserstein distance
            if wasserstein:
                try:
                    sr_list=np.sort(hhd[m].list[0], axis=0)
                    hr_list=np.sort(hhd[m].list[1], axis=0)
                    lr_list=np.sort(hhd[m].list[2], axis=0)
                    lrgen_list=np.sort(hhd[m].list[3], axis=0)

                    wasserDist_SR_HR = wasserstein_distance(sr_list, hr_list)
                    wasserDist_LR_LRgen = wasserstein_distance(lr_list, lrgen_list)

                    
                except IndexError:
                    print('Mode: {} does not list all 4 distributions'.format(m))

            ######################
            if nth_jet_eval_mode=='hr' or nth_jet_eval_mode=='lr':
                if hhd.nums[m] >= 2 and len(bin_entries) == 2:
                    KLDiv = KLD_hist(torch.Tensor(binedges))
                    total_kld.append(float(KLDiv(torch.Tensor(bin_entries[0]), torch.Tensor(bin_entries[1])).item()))
                    kld_dict[modes[m]] = float(KLDiv(torch.Tensor(bin_entries[0]), torch.Tensor(bin_entries[1])).item())
            else:
                if hhd.nums[m] >= 2 and len(bin_entries) == 4:
                    KLDiv = KLD_hist(torch.Tensor(binedges))
                    curr_kl = float(KLDiv(torch.Tensor(bin_entries[0]), torch.Tensor(bin_entries[1])).item())
                    curr_kl += float(KLDiv(torch.Tensor(bin_entries[2]), torch.Tensor(bin_entries[3])).item())
                    total_kld.append(curr_kl)
                    kld_dict[modes[m]] = curr_kl
            if title:
                plt.title(hhd[m].title)
            plt.xlabel(hhd[m].xlabel)
            plt.ylabel(hhd[m].ylabel)
            if legend: 
                #handles, labels = plt.gca().get_legend_handles_labels()
                #if 'E_' in modes[m]:
                #    patch = mpatches.Patch(visible=False,color='none', label=(num_to_str(int(modes[m][2:])) + 'hardest'))
                #    handles.append(patch)
                #plt.legend(handles=handles)
                if 'E_' in modes[m]:
                    if modes[m] == 'E_1':
                        plt.legend(title='hardest pixel')
                    else:
                        plt.legend(title=(num_to_str(int(modes[m][2:])) + 'hardest'))
                else:
                    plt.legend()
            plt.tight_layout(pad=0.5)
            if output_path:
                if not pdf:
                    out_path = output_path + ('_' + hhd.rmAdditions(modes[m]))
                    plt.savefig(out_path.replace('.png', ''))
                else:
                    plt.savefig(output, format='pdf')
            if show:
                plt.show()
            ####
            # include display of wasserstein dist here
            # ...
            ####
            plt.close()
            # plot correlations if necessary.
            if 'corr_' in modes[m]:
                xs,ys = [0,0,1,3,1,0], [1,2,3,2,2,3]
                for i in range(1+5*int('_lr' in modes[m])):
                    xi, yi = xs[i], ys[i]
                    lab = ['Ground truth','Generated','Generated','Ground truth']
                    short = ['HR','SR','LR','LR']
                    kw = {'title': hhd[m].title,
                          'xlabel': '%s %s' %(lab[xi], short[xi]),
                          'ylabel': '%s %s' %(lab[yi], short[yi]),
                          'unit' : unit, 'show_title': title}
                    f,(M,x,y) = plot_corr(hhd[m].list[xi], hhd[m].list[yi], bins=binedges, power=p, **kw)
                    if output_path:
                        if not pdf:
                            out_path = output_path + ('_' + modes[m]+short[xi]+'_'+short[yi])
                            f.savefig(out_path.replace('.png', ''))
                        else:
                            f.savefig(output, format='pdf')
                    if show:
                        f.show()
                    plt.close()
                    if 'slices' in kwargs:
                        kw['slices'] = kwargs['slices']
                    if i==0:
                        f=slice_plot(M.T,x,y,**kw)
                        if output_path:
                            if not pdf:
                                out_path = output_path + ('_' + modes[m]+'_slice')
                                f.savefig(out_path.replace('.png', ''))
                            else:
                                f.savefig(output, format='pdf')
                        if show:
                            f.show()
            if 'ratio_' in modes[m]:
                plt.figure()
                f=plt.gcf()
                if title:
                    plt.title('Ratio: '+hhd[m].title+ '\n' + '$|GT-PR|/GT$')
                plt.ylabel('Entries')
                plt.xlabel('Ratio')
                ratio = MultHist(len(hhd[m].list)//2, mode='')
                for i in range(ratio.num):
                    gt,pr=[0,3][i],[1,2][i]
                    ratio.list[i]=(np.abs(np.array(hhd[m].list[gt])-np.array(hhd[m].list[pr]))/np.array(hhd[m].list[gt]))
                for i in range(len(hhd[m].list)//2):                    
                    try:
                        try:
                            entries, binedges = ratio.histogram(ratio.list[i], np.logspace(-4,np.log10(2), bins), threshold=0.85)
                        except IndexError:
                            continue
                    except ValueError as e:
                        entries, binedges = ratio.histogram(ratio.list[i], np.logspace(-4,np.log10(2), bins), auto_range=False, threshold=0.85)
                    x, y = to_hist(entries, binedges)
                    plt.plot(x, y, linestyle=['-','--'][i], label=['HR','LR'][i])
                    plt.xscale('log')
                    plt.yscale('log')
                    std = np.sqrt(np.clip(y,0,None))
                    std[y == 0] = 0
                    plt.fill_between(x, y+std, y-std, alpha=.2)
                plt.legend()
                plt.tight_layout()
                if output_path:
                    if not pdf:
                        out_path = output_path + ('_' + modes[m]+'_ratio')
                        f.savefig(out_path.replace('.png', ''))
                    else:
                        f.savefig(output, format='pdf')
                if show:
                    f.show()
                plt.close()

        plt.close('all')
    if return_kld_dict:
        return kld_dict
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


def evaluate_results(file, **kwargs):
    '''
    plot results from the hyperparameter search
        `file` should the path to the file containing the results
    '''
    with open(file) as f:
        results = json.load(f)
    def pretty_key(key):
        if key == 'hr_l1':
            return 'HR L$_1$'
        elif key == 'lr_l1':
            return 'LR L$_1$'
        elif key == 'emd':
            return 'EMD'
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
                if 'title' in kwargs and kwargs['title']:
                    splt.set_title(pretty_key(key))
            if l == M-1:
                splt.set_xlabel('iterations [1k]')
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
                y, y_err = np.array(y), np.array(y_err)
                best_iter=y.argmin()
                print(key,iterations[best_iter], y[best_iter], '+/-', y_err[best_iter])
                iterations=np.array(iterations)*1e-3
                splt.plot(iterations, y, linestyle='-', lw=1)
                splt.fill_between(iterations, y+y_err, y-y_err, alpha=.2)
                #_, caps, bars = splt.errorbar(iterations, y, yerr=y_err, label=label, capsize=1.5)
                # loop through bars and caps and set the alpha value
                #[bar.set_alpha(0.5) for bar in bars]
                #[cap.set_alpha(0.5) for cap in caps]
        if not val:
            splt.legend(loc=(1, 0))
    global show
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--dataset_path", type=str, default=None, help="Path to image")
    parser.add_argument("--dataset_type", choices=['h5', 'txt', 'jet', 'spjet', 'hrlrjet'], default=default.dataset_type, help="how is the dataset saved")
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
    parser.add_argument("--power", type=float, default=None, help="power to which to raise the Energy in eval plots")
    parser.add_argument("--n_hardest", type=int, default=None, help="how many of the hardest constituents should be in the ground truth")
    parser.add_argument("--E_thres", type=float, default=None, help="Energy threshold for the ground truth and the generator")
    parser.add_argument("--res_scale", type=float, default=default.res_scale, help="Residual weighting factor")
    parser.add_argument("--preprocessing", action="store_true", help="preprocess pictures used for meanimg")
    parser.add_argument("--pdf", type=str_to_bool, default=True, help="wheter to save the figures as pdf files and tex files")
    parser.add_argument("--fontsize", default=12, type=float)
    parser.add_argument("--slices",type=int, default=5, help='how many slices in slice plot')
    parser.add_argument("--legend", type=str_to_bool, default=True, help="Plot the legend or not")
    parser.add_argument("--title", type=str_to_bool, default=True, help="Plot the title or not")
    parser.add_argument("--thres", type=float, default=0.002, help="Threshold for entry in SR to count as constituent")
    parser.add_argument("--meanenergy", type=str_to_bool, default=False, help="Plot the average jet image with energy instead of counts")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--normhito", action="store_true", help="divide all hitogram entries by hardest gt pixel")
    parser.add_argument("--savehito", action="store_true", help="save hito as np array")
    parser.add_argument("--use_transposed_conv", type=str_to_bool, default=False, help="Whether to use transposed convolutions in upsampling")
    parser.add_argument("--fully_transposed_conv", type=str_to_bool, default=False, help="Whether to ONLY use transposed convolutions in upsampling")   
    parser.add_argument("--num_final_res_blocks", type=int, default=0, help="Whether to add res blocks AFTER upsampling")
    parser.add_argument("--split_meanimg", action='store_true', help='save the mean image for SR / HR separately')
    parser.add_argument('--save_hhd', action='store_true', help='save the MultiModeHist for reuse')
    parser.add_argument('--load_hhd', action='store_true', help='load the MultiModeHist for reuse')
    parser.add_argument('--wasserstein', action='store_true', help='calculate wasserstein distance between distributions')

    opt = parser.parse_args()
    if opt.hw is not None and len(opt.hw) == 2:
        opt.hr_height, opt.hr_width = opt.hw
    opt = vars(opt)
    opt['kwargs'] = {'pdf': opt['pdf'], 'mode': opt['histogram'], 'fontsize': opt['fontsize'], 'threshold': opt['thres'],
                     'power': opt['power'], 'slices': opt['slices'], 'legend': opt['legend'], 'title': opt['title'], 'meanenergy': opt['meanenergy']}
    if opt['save_hhd']:
        opt['kwargs']['save_hhd'] = True
    else:
        opt['kwargs']['save_hhd'] = False
    
    if opt['load_hhd']:
        opt['kwargs']['load_hhd'] = True
    else:
        opt['kwargs']['load_hhd'] = False
    if opt['wasserstein']:
        opt['kwargs']['wasserstein'] = True
    else:
        opt['kwargs']['wasserstein'] = False


    if opt['gpu'] is None:
        try:
            gpu = get_gpu_index()
            num_gpus = torch.cuda.device_count()
            if gpu >= num_gpus:
                gpu = np.random.randint(num_gpus)
            print('running on gpu index {}'.format(gpu))
        except Exception:
            pass
    else:
        gpu = opt['gpu']
    show = opt['no_show']
    normh = opt['normhito']
    savehito = opt['savehito']
    split_meanimg = opt['split_meanimg']
    if opt['hyper_results'] is not None:
        evaluate_results(opt['hyper_results'], **opt['kwargs'])
    else:
        if opt['dataset_path'] is None or (opt['checkpoint_model'] is None and not opt['naive_generator']):
            raise ValueError("For evaluation dataset_path and checkpoint_model are required")
        arguments = {**opt, **{key: default_dict[key] for key in default_dict if key not in opt}}
        opt = namedtuple("Namespace", arguments.keys())(*arguments.values())
        # print(opt)

        out = call_func(opt)
        if out is not None:
            print(out)
