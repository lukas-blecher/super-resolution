
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

'''
The goal of this script is to find the best hyperparameters for the model.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=1500,
                        help='number of batches to run')
    parser.add_argument('--options', type=str, required=True,
                        help='path to options file (json)')
    parser.add_argument('--arguments', type=str, default=None,
                        help='path to arguments file (json)')
    parser.add_argument('--checkpoints', type=int,
                        default=7, help='number of checkpoints')
    parser.add_argument('--root', type=str, default='',
                        help='path to root directory')
    '''
    options file should be a json file containing a dictionary where the keys are the parameter names in esrgan.py and the values
    are another dictionary. The keys are 'type' and 'value'. 'type' can be one of 'range' or 'discrete'.
        'range' should conteain the start, end and number of samples, e.g. [1,4,10] for start=1, end=4 and 10 samples in total.
        'discrete' should contain a list of discrete numbers that should be checked for the parameter.
    '''
    args = parser.parse_args()
    print(args)
    constant_args = ''
    if not args.arguments is None:
        with open(args.arguments) as f:
            args_dict = json.load(f)
            for key in args_dict:
                constant_args += ' --%s %s' % (key, args_dict[key])

    with open(args.options) as options_file:
        options = json.load(options_file)
    # convert list to linspace
    for k, d in zip(options.keys(), options.values()):
        x = d['value']
        # convert linspace
        if d['type'] == 'range':
            options[k] = np.linspace(x[0], x[1], x[2])
        elif d['type'] == 'discrete':
            options[k] = x

    # perform grid search
    grid = ParameterGrid(options)
    print('Performing grid search for %i different sets of hyperparameters. Each trained for %i batches.' % (
        len(grid), args.batches))
    for hyperparameters in grid:
        # new name for each hyperparameter set
        arguments = ['--name '+'-'.join(str(round(x, 4)).replace('.', '_')
                                        for x in hyperparameters.values())]
        for i in hyperparameters:
            arguments.append('--%s %s' % (i, hyperparameters[i]))
        arguments = ' '.join(arguments)
        print(os.path.join(args.root, "esrgan.py "+arguments+constant_args))
        #os.system('python3 '+os.path.join(args.root, "esrgan.py"+arguments+constant_args))
    #options = namedtuple("options", options.keys())(*options.values())

    # evaluate results


if __name__ == '__main__':
    main()
