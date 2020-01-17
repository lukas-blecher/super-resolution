
import torch
import numpy as np
import json
import time
import sys
import os
import os.path as path
import argparse
import warnings
from collections import namedtuple
from sklearn.model_selection import ParameterGrid
from options.default import *
from utils import str_to_bool, get_gpu_index
from esrgan import train
from evaluation.eval import distribution
'''
The goal of this script is to find the best hyperparameters for the model.
A grid search is performed over all the possible hyperparameter combinations that are defined in the argument 'options'.
A fixed amount of checkpoints are saved during training and evaluated on two metrics. The results are saved in a json file.
'''
gpu = 0


def grid_search(args):
    '''
    options file should be a json file containing a dictionary where the keys are the parameter names in esrgan.py and the values
    are another dictionary. The keys are 'type', 'value' and 'coupled. 'type' can be one of 'range' or 'discrete'.
        'range' should conteain the start, end and number of samples, e.g. [1,4,10] for start=1, end=4 and 10 samples in total.
        'logrange' should contain the start, end and number of samples just as in 'range'. The difference is that the steps in between are logarithmically increasing.
        'discrete' should contain a list of discrete numbers that should be checked for the parameter.
        'coupled' should contain a list of the same length as the coupled parameter that needs to be saved under 'parameter'. (only compatible with 'discrete')
            This option can be used if you want to scale a parameter accordingly to another specified hyperparameter
    For all types there is the option 'include_zero' which should be a boolean. If true the value 0 will be be added to the list. Usefull for the range types.
    '''
    global gpu
    results, args_dict = {}, {}
    os.makedirs(os.path.split(args.results)[0], exist_ok=True)
    with open(args.options) as options_file:
        options = json.load(options_file)
    # check what type of hyperparameter we have
    coupled, parameters = {}, {}
    for k, d in zip(options.keys(), options.values()):
        x = d['value']
        use_zero = False
        if 'include_zero' in d:
            use_zero = d['include_zero']
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
        if use_zero:
            parameters[k] = [0, *list(parameters[k])]

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
        model_name = '-'.join((str(round(x, 4)) if x > 1e-4 else '%.3e' % x).replace('.', '_') for x in hyperparameters.values())
        arguments = {**hyperparameters, **args_dict}
        additional_arguments = {key: default_dict[key] for key in default_dict if key not in arguments}
        arguments = {**arguments, **additional_arguments}
        arguments['validation_path'] = args.test_images
        arguments['n_batches'] = args.batches
        arguments['name'] = model_name
        arguments['n_validations'] = args.checkpoints
        arguments['n_histograms'] = 1 if args.histograms else -1
        #arguments['save'] = args.save_checkpoints
        if args.save_results:
            arguments['image_path'] = args.output_path
        metric_results = []
        arguments['metric_results'] = metric_results
        arguments_ntuple = namedtuple("arguments", arguments.keys())(*arguments.values())
        try:
            torch.cuda.empty_cache()
            info[-1]['metrics'] = train(arguments_ntuple, gpu=gpu)['validation']
        except RuntimeError as e:
            print('RuntimeError: %s' % e)
            continue
        results['results'] = info
        with open(args.results, 'w') as outfile:
            json.dump(results, outfile)


def random_search(args):
    global gpu
    results, args_dict = {}, {}
    os.makedirs(os.path.split(args.results)[0], exist_ok=True)
    with open(args.options) as options_file:
        options = json.load(options_file)
    parameters = {}
    for k, d in zip(options.keys(), options.values()):
        x = d['value']
        if d['type'] == 'range':
            assert len(x) >= 2
            parameters[k] = [x[0], x[1]]
        else:
            warnings.warn('Every other type will be ignored. Expected "range" got %s' % d['type'])

    if not args.arguments is None:
        with open(args.arguments) as f:
            args_dict = json.load(f)
            args_dict = {key: args_dict[key] for key in args_dict if key not in options}
            results = args_dict.copy()

    results['validation_path'] = args.test_images
    # perform random search
    print('Performing grid search for %i different sets of hyperparameters. Each trained for %i batches.' % (args.amount, args.batches))
    info = []
    for i in range(args.amount):
        hyperparameters = {key: sample(parameters[key]) for key in parameters}
        print('checking hyperparameters: %s' % str(hyperparameters))
        info.append(hyperparameters.copy())
        # new name for each hyperparameter set
        model_name = '-'.join((str(round(x, 4)) if x > 1e-4 else '%.3e' % x).replace('.', '_') for x in hyperparameters.values())
        arguments = {**hyperparameters, **args_dict}
        additional_arguments = {key: default_dict[key] for key in default_dict if key not in arguments}
        arguments = {**arguments, **additional_arguments}
        arguments['testset_path'] = args.test_images
        arguments['n_batches'] = args.batches
        arguments['name'] = model_name
        arguments['n_validations'] = args.checkpoints
        arguments['n_histograms'] = 1 if args.histograms else -1
        if args.save_results:
            arguments['image_path'] = args.output_path
        metric_results = []
        arguments['metric_results'] = metric_results
        arguments_ntuple = namedtuple("arguments", arguments.keys())(*arguments.values())
        try:
            torch.cuda.empty_cache()
            eval_results = train(arguments_ntuple, gpu=gpu)['eval_results']
        except RuntimeError as e:
            print('RuntimeError: %s' % e)
            continue
        aresults = np.array(eval_results)
        info[-1]['best_mean'] = float(aresults.mean(1).min())
        info[-1]['eval_results'] = eval_results
        results['results'] = info
        with open(args.results, 'w') as outfile:
            json.dump(results, outfile)

    minkld = float('inf')
    best_ind = None
    for i in range(len(info)):
        if info[i]['best_mean'] < minkld:
            minkld = info[i]['best_mean']
            best_ind = i
    if best_ind is not None:
        results['best_set'] = info[best_ind]
    with open(args.results, 'w') as outfile:
        json.dump(results, outfile)


def logrange(a, b, c):
    return np.logspace(np.log(a)/np.log(10), np.log(b)/np.log(10), c)


def sample(list):
    return list[0]+np.random.random()*(list[1]-list[0])


if __name__ == '__main__':
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
    parser.add_argument('--histograms', action='store_true', help='whether to also compute histograms at the end of the training')
    parser.add_argument('--random', type=str_to_bool, default=False, help='if true the parameters are selected randomly from the given ranges')
    parser.add_argument('--amount', type=int, default=6, help='amount of points to check in the hyperparameter space if used with random')
    args = parser.parse_args()
    print(args)
    try:
        gpu=get_gpu_index()        
        num_gpus=torch.cuda.device_count()
        if gpu >= num_gpus:
            gpu=np.random.randint(num_gpus)
        print('running on gpu index {}'.format(gpu))
    except Exception:
        pass
    if args.random:
        random_search(args)
    else:
        grid_search(args)
