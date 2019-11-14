import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_losses(j, file, ax):
    with open(file) as f:
        info = json.load(f)
    if not 'loss' in info:
        raise KeyError('No Loss in the info file.')
    loss = info['loss']
    warmup = info['warmup_batches']
    if len(loss['g_loss']) == len(loss['d_loss']):
        warmup = 0
    name = os.path.basename(file).replace('_info.json', '')
    ax = ax.flatten()
    batches = np.arange(0, len(loss['g_loss']))

    for i, k in enumerate(loss.keys()):
        ax[i].set_title(k)
        start = 0 if len(loss[k]) == len(batches) else warmup
        ax[i].plot(batches[start:], loss[k], color=colors[j % len(colors)], label=name)
        if i in range(N-a, N):
            ax[i].set_xlabel('iterations')
        if i == 0:
            ax[i].legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True, help='info file with loss')

    N = 8
    a = int(np.sqrt(N))
    b = N//a
    f, ax = plt.subplots(b, a, sharex=True)

    args = parser.parse_args()
    for i, f in enumerate(args.file):
        plot_losses(i, f, ax)
    plt.tight_layout()
    plt.show()
