import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import glob

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

keys=['d_loss_def', 'd_loss_pow', 'd2_loss_def', 'd2_loss_pow', 'g_loss', 'def_loss', 'pow_loss', 'adv_loss', 'pixel_loss', 'lr_loss', 'hist_loss', 'nnz_loss', 'mask_loss', 'wasser_loss', 'hit_loss']
lambdas=[None,None,None,None,None,'lambda_pix','lambda_pow','lambda_adv','lambda_hr','lambda_lr','lambda_hist','lambda_nnz','lambda_mask','lambda_wasser','lambda_hit']
loss_to_lambda = {}
for i, key in enumerate(keys):
    loss_to_lambda[key] = lambdas[i]

def compare_loss(ax, info_files, loss_name, include_lambda, min_batch, max_batch, avr_int=None):
    infos = []
    losses = []
    report_frequency = []
    min_val, max_val = 0, 0
    for i in range(len(info_files)):
        with open(info_files[i]) as f:
            infos.append(json.load(f))
        report_frequency.append(infos[i]['argument']['report_freq'])
        losses.append(infos[i]['loss'][loss_name])
        if include_lambda:
            if loss_to_lambda[loss_name] is not None:
                for jj in range(len(losses[i])):
                    losses[i][jj] *= infos[i]['argument'][loss_to_lambda[loss_name]]
                                
    min_entry = int(min_batch / report_frequency[0])
    max_entry = int(max_batch / report_frequency[0])
    entry_list = np.arange(min_entry, max_entry+1)
    min_val = min(losses[0][min_entry:max_entry])
    max_val = max(losses[0][min_entry:max_entry])
    for j in range(len(info_files)):
        currmin_val = min(losses[j][min_entry:max_entry])
        currmax_val = max(losses[j][min_entry:max_entry])
        if currmin_val < min_val:
            min_val = currmin_val
        if currmax_val > max_val:
            max_val = currmax_val
    if include_lambda and (loss_to_lambda[loss_name] is not None):
        print('minimal value: {}, maximal value: {}, lambda: {}'.format(min_val,max_val,infos[0]['argument'][loss_to_lambda[loss_name]]))
    else:
        print('minimal value: {}, maximal value: {}'.format(min_val,max_val))
    for k in range(len(info_files)):
        if include_lambda and (loss_to_lambda[loss_name] is not None) and (k==len(info_files)-1):
            ax[k].set_title(os.path.basename(info_files[k])+', lambda: {}'.format(infos[0]['argument'][loss_to_lambda[loss_name]]))
        else:
            ax[k].set_title(os.path.basename(info_files[k]))
        ax[k].set_xlabel('batches x {}'.format(report_frequency[0]))
        ax[k].set_ylim([min_val-(0.03*min_val), max_val+(0.03*max_val)])
        ax[k].plot(entry_list, np.array(losses[k][min_entry:max_entry+1]), color=colors[0], label=loss_name, alpha=1,zorder=0)
        if avr_int is not None:
            x_points,y_points,y_errs = plot_mean(entry_list, np.array(losses[k][min_entry:max_entry+1]), avr_int)
            ax[k].errorbar(x_points,y_points,y_errs,fmt='ro',linewidth=2.0,ecolor='red',zorder=10,label=str(avr_int))
        if k == 0:
            ax[k].legend()

def plot_mean(x,y,avr_int):
    x_points = []
    y_points = []
    y_errs = []
    assert len(x)==len(y), 'err'
    assert avr_int > 0 and avr_int < 1, 'err2'
    plotLen = len(x)
    foldLen = int(plotLen * avr_int)
    numFolds = int(1/avr_int)
    if numFolds*foldLen > plotLen:
        numFolds = numFolds - 1
    for fold in range(1, numFolds+1):
        x_points.append(fold*foldLen+x[0])
        y_arr = y[(fold-1)*foldLen:fold*foldLen]
        y_points.append(np.mean(y_arr))
        y_errs.append(np.std(y_arr))
    print(x_points)
    print(y_points)
    print(y_errs)
    return x_points, y_points, y_errs

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
    print(eval_modes)
    

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
    parser.add_argument('-m', '--mode', choices=['loss', 'evalmean','compare_loss'], default='loss', help='what to plot')
    parser.add_argument('--show', action="store_true", help='whether to show plots or save as pdf')
    parser.add_argument('--include_lambda',action="store_true",help='include lambda params in loss plot')
    parser.add_argument('--min_batch', type=int, default=0, help='batch to start the plotting')
    parser.add_argument('--max_batch', type=int, default=50000, help='batch to stop the plotting')
    parser.add_argument('--interval', type=float, default=0.1, help='relative fraction of batches to include in mean plot')
    parser.add_argument('--out', type=str, default='./',help='output path')
    args = parser.parse_args()
    
    if  args.mode == 'compare_loss':
        if not os.path.isdir(args.out):
            os.mkdir(args.out)
        for loss in keys:
            fig, ax = plt.subplots(1, len(args.file), sharex=True, figsize=(12,6))
            ax = ax.flatten()
            compare_loss(ax, args.file, loss, include_lambda=args.include_lambda, min_batch=args.min_batch, max_batch=args.max_batch,avr_int=args.interval)
            plt.tight_layout()
            if args.show:
                plt.show()
            else:
                basestr = ''
                for i in range(len(args.file)):
                    basestr += os.path.basename(args.file[i].replace('_info.json', ''))
                    basestr += '_'
                outpath = os.path.join(args.out, 'compare_'+basestr+loss+'.pdf')
                plt.savefig(outpath)
        
    else:
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