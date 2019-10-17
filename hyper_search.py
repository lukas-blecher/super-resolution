
import torch
import numpy as np
import json
import time
import sys
import os
import os.path as path
import argparse
from collections import namedtuple
from sklearn.model_selection import ParameterGrid

from test_on_image import test_image
from options.default import *
from evaluation.PSNR_SSIM_metric import get_metrics
from esrgan import train
'''
The goal of this script is to find the best hyperparameters for the model.
A grid search is performed over all the possible hyperparameter combinations that are defined in the argument 'options'.
A fixed amount of checkpoints are saved during training and evaluated on two metrics. The results are saved in a json file.
'''


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=1500, help='number of batches to run')
    parser.add_argument('--options', type=str, required=True, help='path to options file (json)')
    parser.add_argument('--arguments', type=str, default=None, help='path to arguments file (json)')
    parser.add_argument('--checkpoints', type=int, default=7, help='number of checkpoints')
    parser.add_argument('--root', type=str, default='', help='path to root directory')
    parser.add_argument('--test_images', type=str, required=True, help='path to test images')
    parser.add_argument('--output_path', type=str, default='images/outputs', help='path to output images folder')
    parser.add_argument('--results', type=str, default='results/hyper_search_results.json', help='path to json file containing hyper_search results')
    '''
    options file should be a json file containing a dictionary where the keys are the parameter names in esrgan.py and the values
    are another dictionary. The keys are 'type' and 'value'. 'type' can be one of 'range' or 'discrete'.
        'range' should conteain the start, end and number of samples, e.g. [1,4,10] for start=1, end=4 and 10 samples in total.
        'discrete' should contain a list of discrete numbers that should be checked for the parameter.
    '''
    args = parser.parse_args()
    print(args)
    results, args_dict = {}, {}
    os.makedirs(os.path.split(args.results)[0], exist_ok=True)
    with open(args.options) as options_file:
        options = json.load(options_file)
    # convert list to linspace
    for k, d in zip(options.keys(), options.values()):
        x = d['value']
        # convert linspace
        if d['type'] == 'range':
            assert len(x) == 3
            options[k] = np.linspace(x[0], x[1], x[2])
        elif d['type'] == 'discrete':
            options[k] = x

    if not args.arguments is None:
        with open(args.arguments) as f:
            args_dict = json.load(f)
            results = {key: args_dict[key] for key in options if key not in options}

    # perform grid search
    grid = ParameterGrid(options)
    print('Performing grid search for %i different sets of hyperparameters. Each trained for %i batches.' % (
        len(grid), args.batches))
    info = []
    for hyperparameters in grid:
        print('testing hyperparameters: %s' % str(hyperparameters))
        info.append(hyperparameters.copy())
        # new name for each hyperparameter set
        model_name = '-'.join(str(round(x, 4)).replace('.', '_') for x in hyperparameters.values())

        arguments = merge_two_dicts(hyperparameters, args_dict)  # {**hyperparameters, **args_dict}
        additional_arguments = {key: default_dict[key] for key in default_dict if key not in arguments}
        arguments = merge_two_dicts(arguments, additional_arguments)  # {**arguments, **additional_arguments}
        arguments['n_batches'] = args.batches
        arguments['name'] = model_name
        arguments['n_checkpoints'] = args.checkpoints
        arguments_ntuple = namedtuple("arguments", arguments.keys())(*arguments.values())
        train(arguments_ntuple)

        # evaluate results
        print('testing parameters')
        metric_results = {'psnr': [], 'ssim': [], 'psnr_std': [], 'ssim_std': []}
        model_path = args_dict['model_path'] if 'model_path' in args_dict else default.model_path
        for i in range(args.checkpoints):
            model_name_i = os.path.join(args.root, model_path, "%s_generator_%d.pth" % (model_name, i+1))
            outpath = os.path.join(args.output_path, model_name)
            test_dict = arguments.copy()
            test_dict['checkpoint_model'] = model_name_i
            test_dict['image_path'] = args.test_images
            test_dict['output_path'] = outpath
            test_dict['downsample'] = True
            test_image(namedtuple("opt", test_dict.keys())(*test_dict.values()))
            psnr, psnr_std, ssim, ssim_std = get_metrics(args.test_images, outpath)
            metric_results['psnr'].append(psnr)
            metric_results['psnr_std'].append(psnr_std)
            metric_results['ssim'].append(ssim)
            metric_results['ssim_std'].append(ssim_std)
        info[-1]['metrics'] = metric_results
        print(info)
    results['results'] = info

    with open(args.results, 'w') as outfile:
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
