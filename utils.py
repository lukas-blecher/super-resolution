import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def toUInt(x):
    return np.squeeze(x*255/x.max()).astype(np.uint8)


def save_numpy(array, path):
    Image.fromarray(toUInt(array)).save(path)


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
        self.binsizes = binsizes
        self.binmean = binsizes.mean()
        self.kldiv = nn.KLDivLoss(reduction='sum')

    def to(self, device):
        self.kldiv = self.kldiv.to(device)
        self.binsizes = self.binsizes.to(device)
        return self

    def forward(self, q_entries, p_entries):
        # convert p and q to probabilies
        p_entries = p_entries*self.binsizes/p_entries.sum()
        # add epsilon to Q because kld is not defined if it is zero and P is not
        q_entries += 1e-8
        # convert Q to log probability
        q_entries = (q_entries*self.binsizes/q_entries.sum()).log()

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
