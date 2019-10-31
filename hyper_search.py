
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
from esrgan import train
'''
The goal of this script is to find the best hyperparameters for the model.
A grid search is performed over all the possible hyperparameter combinations that are defined in the argument 'options'.
A fixed amount of checkpoints are saved during training and evaluated on two metrics. The results are saved in a json file.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=1500, help='number of batches to run')
    parser.add_argument('--options', type=str, required=True, help='path to options file (json)')
    parser.add_argument('--arguments', type=str, default=None, help='path to constant arguments file (json)')
    parser.add_argument('--checkpoints', type=int, default=7, help='number of checkpoints')
    parser.add_argument('--root', type=str, default='', help='path to root directory')
    parser.add_argument('--test_images', type=str, required=True, help='path to test images')
    parser.add_argument('--output_path', type=str, default='images/outputs', help='path to output images folder')
    parser.add_argument('--results', type=str, default='results/hyper_search_results.json', help='path to json file containing hyper_search results')
    parser.add_argument('--save_results', action='store_true', help='whether also to save results as images or to only save the metrics')
    parser.add_argument('--save_checkpoints', action='store_true', help='whether to save checkpoints or not')
    '''
    options file should be a json file containing a dictionary where the keys are the parameter names in esrgan.py and the values
    are another dictionary. The keys are 'type', 'value' and 'coupled. 'type' can be one of 'range' or 'discrete'.
        'range' should conteain the start, end and number of samples, e.g. [1,4,10] for start=1, end=4 and 10 samples in total.
        'logrange' should contain the start, end and number of samples just as in 'range'. The difference is that the steps in between are logarithmically increasing.
        'discrete' should contain a list of discrete numbers that should be checked for the parameter.
        'coupled' should contain a list of the same length as the coupled parameter that needs to be saved under 'parameter'. (only compatible with 'discrete')
            This option can be used if you want to scale a parameter accordingly to another specified hyperparameter
    '''
    args = parser.parse_args()
    print(args)
    results, args_dict = {}, {}
    os.makedirs(os.path.split(args.results)[0], exist_ok=True)
    with open(args.options) as options_file:
        options = json.load(options_file)
    # check what type of hyperparameter we have
    coupled, parameters = {}, {}
    for k, d in zip(options.keys(), options.values()):
        x = d['value']
        # convert linspace
        if d['type'] == 'range':
            assert len(x) == 3
            parameters[k] = np.linspace(x[0], x[1], x[2])
        if d['type'] == 'logrange':
            assert len(x) == 3
            parameters[k] = logrange(x[0], x[1], x[2])
        elif d['type'] == 'discrete':
            parameters[k] = x
        elif d['type'] == 'coupled':
            # the hyperparameter is coupled to another hyperparameter from the type `distcrete`
            coupled_param = d['parameter']
            # check if lenght of values for both hyperparameters are the same
            assert len(x) == len(options[coupled_param]['value'])
            coupled[k] = {}
            coupled[k]['value'] = x
            coupled[k]['coupled'] = coupled_param

    if not args.arguments is None:
        with open(args.arguments) as f:
            args_dict = json.load(f)
            args_dict = {key: args_dict[key] for key in args_dict if key not in options}
            results = args_dict.copy()

    results['validation_path'] = args.test_images
    # perform grid search
    grid = ParameterGrid(parameters)
    print('Performing grid search for %i different sets of hyperparameters. Each trained for %i batches.' % (len(grid), args.batches))
    info = []
    for hyperparameters in grid:
        # add any coupled parameters to the dictionary
        for c in coupled:
            # parameter c is coupled to:
            c_param = coupled[c]['coupled']
            # get right value for c
            hyperparameters[c] = coupled[c]['value'][parameters[c_param].index(hyperparameters[c_param])]

        print('checking hyperparameters: %s' % str(hyperparameters))
        info.append(hyperparameters.copy())
        # new name for each hyperparameter set
        model_name = '-'.join(str(round(x, 4)).replace('.', '_') for x in hyperparameters.values())
        arguments = {**hyperparameters, **args_dict}
        additional_arguments = {key: default_dict[key] for key in default_dict if key not in arguments}
        arguments = {**arguments, **additional_arguments}
        arguments['validation_path'] = args.test_images
        arguments['n_batches'] = args.batches
        arguments['name'] = model_name
        arguments['n_checkpoints'] = args.checkpoints
        arguments['save'] = args.save_checkpoints
        if args.save_results:
            arguments['output_path'] = args.output_path
        metric_results = []
        arguments['metric_results'] = metric_results
        arguments_ntuple = namedtuple("arguments", arguments.keys())(*arguments.values())
        try:
            torch.cuda.empty_cache()
            info[-1]['metrics'] = train(arguments_ntuple)
        except RuntimeError as e:
            print('RuntimeError: %s' % e)
            continue
        results['results'] = info
        with open(args.results, 'w') as outfile:
            json.dump(results, outfile)

def logrange(a,b,c):
    return np.logspace(np.log(a)/np.log(10),np.log(b)/np.log(10),c)

if __name__ == '__main__':
    main()
