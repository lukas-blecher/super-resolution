import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import glob

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_losses(j, file, ax):
    with open(file) as f:
        info = json.load(f)
    if not 'loss' in info:
        raise KeyError('No Loss in the info file.')
    loss = info['loss']
    try:
        warmup = info['warmup_batches']
    except KeyError:
        warmup = info['argument']['warmup_batches']
    if len(loss['g_loss']) == len(loss['d_loss']):
        warmup = 0
    name = os.path.basename(file).replace('_info.json', '')
    batches = np.arange(0, len(loss['g_loss']))

    for i, k in enumerate(loss.keys()):
        ax[i].set_title(k)
        try:
            start = 0 if len(loss[k]) == len(batches) else warmup
            ax[i].plot(batches[start:], loss[k], color=colors[j % len(colors)], label=name)
        except ValueError:
            start = 0 if len(loss[k]) == len(batches) else warmup//info['argument']['report_freq']
            ax[i].plot(batches[start:], loss[k], color=colors[j % len(colors)], label=name)

        if i in range(N-a, N):
            ax[i].set_xlabel('iterations')
        if i == 0:
            ax[i].legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True, help='info file with loss')
    args = parser.parse_args()
    if len(args.file)==1:
        args.file = glob.glob(args.file[0])
    with open(args.file[0]) as f:
        info = json.load(f)
    if not 'loss' in info:
        raise KeyError('No Loss in the info file.')
    N = len(info['loss'].keys())
    a = int(np.sqrt(N))
    b = int(np.ceil(N/a))
    fig, ax = plt.subplots(b, a, sharex=True)
    ax = ax.flatten()
    for i in range(a*b-1, N-1, -1):
        fig.delaxes(ax[i])
    for i, f in enumerate(args.file):
        plot_losses(i, f, ax)

    plt.tight_layout()
    plt.show()
