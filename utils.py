import torch
import torch.nn as nn


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


def KLD_hist(p_entries, q_entries, binedges):
    q_entries+=1e-6
    binsizes = binedges[1:]-binedges[:-1]
    logp=torch.log(p_entries/torch.sum(p_entries))
    logp[logp==-float('inf')]=0
    kl = len(binsizes)/(len(p_entries)*binsizes.mean())*torch.sum(p_entries*binsizes*(logp-torch.log(q_entries/torch.sum(q_entries))))
    kl[kl!=kl]=0
    return kl
