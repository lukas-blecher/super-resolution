import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import glob

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_eval_mean(file):
    with open(file) as f:
        info = json.load(f)
    results = np.array(info['eval_results'])
    eval_modes = info['argument']['eval_modes']
    interval = info['argument']['evaluation_interval']
    ev_mean = []
    batches = []
    name = os.path.basename(file).replace('_info.json', '')
    for i in range(len(results)):
        ev_mean.append(float(np.mean(np.abs(results[i]))))
        batches.append((i+1)*interval)
    ev_mean = np.array(ev_mean)
    batches = np.array(batches)
    plt.plot(batches, ev_mean, color='#1f77b4', alpha=1.0, label = name, marker='o', markerfacecolor='#ff7f0e')
    plt.xlabel('batches done')
    plt.ylabel('mean')
    plt.title('mean eval results for '+name)
    plt.legend()
    

def plot_losses(j, file, ax, alpha=1):
    with open(file) as f:
        info = json.load(f)
    if not 'loss' in info:
        raise KeyError('No Loss in the info file.')
    loss = info['loss']
    try:
        warmup = info['warmup_batches']
    except KeyError:
        warmup = info['argument']['warmup_batches']
    try:
        if len(loss['g_loss']) == len(loss['d_loss']):
            warmup = 0
    except KeyError:
        if len(loss['g_loss']) == len(loss['d_loss_def']):
            warmup = 0

    name = os.path.basename(file).replace('_info.json', '')
    batches = np.arange(0, len(loss['g_loss']))

    for i, k in enumerate(loss.keys()):
        ax[i].set_title(k)
        try:
            start = 0 if len(loss[k]) == len(batches) else warmup
            ax[i].plot(batches[start:], np.array(loss[k]), color=colors[j % len(colors)], label=name, alpha=alpha)
        except ValueError:
            start = 0 if len(loss[k]) == len(batches) else warmup//info['argument']['report_freq']
            ax[i].plot(batches[start:], loss[k], color=colors[j % len(colors)], label=name, alpha=alpha)

        if i in range(N-a, N):
            ax[i].set_xlabel('iterations')
        if i == 0:
            ax[i].legend()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True, help='info file with loss')
    parser.add_argument('-m', '--mode', choices=['loss', 'evalmean'], default='loss', help='what to plot')
    parser.add_argument('--show', action="store_true", help='whether to show plots or save as pdf')
    args = parser.parse_args()
    files=[]
    name = os.path.basename(args.file[0]).replace('_info.json', '')
    if args.mode == 'loss':
        for f in args.file:
            files.extend(glob.glob(f))
        with open(files[0]) as f:
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
        for i, f in enumerate(files):
            plot_losses(i, f, ax, 1/np.sqrt(len(files)))
    
    elif args.mode == 'evalmean':
        plt.figure(figsize=(12, 8))
        plot_eval_mean(args.file[0])

    plt.tight_layout()
    if args.show:
        plt.show()
    elif args.mode == 'loss':
        plt.savefig(name+'_loss.pdf')
    elif args.mode == 'evalmean':
        plt.savefig(name+'_evalmean.pdf')