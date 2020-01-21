import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def toUInt(x):
    return np.squeeze(x*255/x.max()).astype(np.uint8)


def save_numpy(array, path):
    Image.fromarray(toUInt(array)).save(path)


def str_to_bool(value):
    if value is None:
        return None
    elif value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


class pointerList:
    def __init__(self, *argv):
        self.inds = []
        for i in range(len(argv)):
            self.inds.append(i)
            setattr(self, chr(i+97), argv[i])

    def __getitem__(self, i):
        if i not in self.inds:
            raise IndexError("index out of range")
        return getattr(self, chr(i+97))

    def get(self, i):
        return self[self.inds[i]]

    def __setitem__(self, i, val):
        if i not in self.inds:
            self.inds.append(i)
        setattr(self, chr(i+97), val)

    def __len__(self):
        return len(self.inds)

    def append(self, val):
        if len(self.inds) > 0:
            self.inds.append(self.inds[-1]+1)
        else:
            self.inds.append(0)
        self[self.inds[-1]] = val

    def __str__(self):
        s = ''
        for i in self.inds:
            s += '%i: %s, ' % (i, str(self[i]))
        return s[:-2]

    def call(self, foo, arg=None):
        for i in self.inds:
            if arg is None:
                getattr(self[i], foo)()
            else:
                getattr(self[i], foo)(arg)


class KLD_hist(nn.Module):
    def __init__(self, binedges):
        super(KLD_hist, self).__init__()
        binsizes = binedges[1:]-binedges[:-1]
        self.binsizes = binsizes.float()
        self.binmean = self.binsizes.mean()
        self.kldiv = nn.KLDivLoss(reduction='sum')

    def to(self, device):
        self.kldiv = self.kldiv.to(device)
        self.binsizes = self.binsizes.to(device)
        return self

    def forward(self, q_entries, p_entries):
        # compute sums
        N_p, N_q = p_entries.sum().float(), q_entries.sum().float()
        # convert p and q to probabilies
        p_entries = p_entries*self.binsizes/N_p
        # add epsilon to Q because kld is not defined if it is zero and P is not
        q_entries += 1e-6
        # convert Q to log probability
        q_entries = (q_entries*self.binsizes/N_q).log()

        return self.kldiv(q_entries, p_entries)/self.binmean


def plot_grad_flow(named_parameters, path=None):
    '''By RoshanRane https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6'''

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.figure(figsize=(len(ave_grads)//5, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c", label='max-gradient')
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b", label='mean-gradient')
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(-.5, len(ave_grads)-.5)
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path)

# Adapted from https://github.com/dfdazac/wassdistance :
# Originally adapted from https://github.com/gpeyre/SinkhornAutoDiff


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(B, N, P_1, D_1)`, :math:`(B, N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return self

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        #print(x_points, y_points)
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False, device=self.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False, device=self.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def cut_smaller(t, threshold=0):
    '''
    Cuts a tensor in the 2nd dimension in a way that all values over :math:`threshold` are still in the tensor
    at the cost that some values smaller still remain in the other batches
    Args:
        t (Tensor): Tensor to be cut
        threshold (float): Value to cut at
    Shape: 
        - Input: :math:`(B, N)`
        - Output: :math:`(B, N')`
    '''
    maximum = 0
    for i in range(len(t)):
        gi = t[i][t[i] > threshold]
        maximum = maximum if len(gi) < maximum else len(gi)
    return t[:, -maximum:]


def softgreater(x, val, sigma=5000, delta=0):
    # differentiable verions of torch.where(x>val)
    return torch.sigmoid(sigma * (x-val+delta))

def get_hitogram(t, factor, threshold=.1, sig=80):
    return torch.sigmoid(sig*(torch.cat(torch.split(torch.cat(torch.split(t,factor,-2)),factor,-1))-threshold)).mean((0,1))

def nnz_mask(x, sigma=5e4):
    return torch.sigmoid(sigma*x)


class SumRaster:
    def __init__(self, factor, height=None, width=None, threshold=.1):
        self.width, self.height = width, height
        self.factor = factor
        self.threshold = threshold
        if width is None:
            self.width = height
        self.sr, self.hr = [np.zeros((factor, factor)) for _ in range(2)]

    def add(self, SR, HR):
        if self.height is None:
            self.height = HR.shape[-2]
        if self.width is None:
            self.width = HR.shape[-1]
        self.sr += (np.array(np.split(np.array(np.split(SR.numpy(), self.height//self.factor, -2)),
                                      self.width//self.factor, -1)) > self.threshold).sum((0, 1, 2, 3))
        self.hr += (np.array(np.split(np.array(np.split(HR.numpy(), self.height//self.factor, -2)),
                                      self.width//self.factor, -1)) > self.threshold).sum((0, 1, 2, 3))

    def get_hist(self):
        return self.sr, self.hr


def make_hist(raster, threshold=.1):
    'Takes a Raster class and returns two histograms each of the shape (factor, factor).'
    raster.reset()
    sr, hr = [torch.zeros(raster.factor, raster.factor) for _ in range(2)]
    for s, h in raster:
        sr += (s > threshold)
        hr += (h > threshold)
    return sr, hr


def plot_hist2d(sr, hr, cmap='viridis'):
    vmin = min([hr.min().item(), sr.min().item()])
    vmax = max([hr.max().item(), sr.max().item()])
    f, ax = plt.subplots(1, 2)
    ax = ax.flatten()
    ax[0].imshow(sr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title('prediction')
    ax[0].axis('off')
    gt = ax[1].imshow(hr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title('ground truth')
    ax[1].axis('off')
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.25, 0.05, 0.5])
    f.colorbar(gt, cax=cbar_ax)
    return f


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
