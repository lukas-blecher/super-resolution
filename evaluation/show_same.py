import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import glob


def hyper2float(s):
    s = s.replace('_training', '').replace('_', '.')
    try:
        s = float(s)
    except:
        pass
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='file name/pattern')
    parser.add_argument('-i', '--dir', type=str, required=True, help='folder with subdirs')
    args = parser.parse_args()
    subdirs = sorted(glob.glob(os.path.join(args.dir, '*')))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    look_in_subdirs = len(subdirs) != 0  # check if we want to look in one folder for the pattern or in the subfolders if they exist
    if look_in_subdirs:
        names = [hyper2float(os.path.basename(subdirs[i])) for i in range(len(subdirs))]
        try:
            perm = np.argsort(names)
            if type(names[0]) is str:
                digs=[int(''.join(filter(lambda x: x.isdigit(), s))) for s in names]
                perm = np.argsort(digs)
            subdirs = np.array(subdirs)[perm]
            names = np.array(names)[perm]
        except:
            pass
        N = len(subdirs)
        a = int(np.sqrt(N))
        b = int(np.ceil(N/a))
        f, ax = plt.subplots(a, b, sharex=True, sharey=True, figsize=(b*5, a*5))
        axf = ax.flatten()
        for i in range(len(subdirs)):
            axi = axf[i]
            file = glob.glob(os.path.join(subdirs[i], args.file))
            if len(file) != 1:
                print('There are %s files matching "%s" in %s' % ('multiple' if len(file) > 1 else 'no', args.file, subdirs[i]))
                f.delaxes(axf[i])
                continue
            file = file[0]
            name = names[i]
            axi.set_title(name)
            axi.imshow(plt.imread(file), interpolation='bilinear')
            axi.axis('off')
    else:
        files = glob.glob(os.path.join(args.dir, args.file))
        names = [os.path.basename(files[i]) for i in range(len(files))]
        try:
            perm = np.argsort(names)
            if type(names[0]) is str:
                digs=[int(''.join(filter(lambda x: x.isdigit(), s))) for s in names]
                perm = np.argsort(digs)
            files = np.array(files)[perm]
            names = np.array(names)[perm]
        except:
            pass
        N = len(files)
        a = int(np.sqrt(N))
        b = int(np.ceil(N/a))
        f, ax = plt.subplots(a, b, sharex=True, sharey=True, figsize=(b*5, a*5))
        axf = ax.flatten()
        for i in range(len(files)):
            axi = axf[i]
            name = names[i]
            axi.set_title(name)
            axi.imshow(plt.imread(files[i]), interpolation='bilinear')
            axi.axis('off')
    for i in range(a*b-1, N-1, -1):
        f.delaxes(axf[i])

    plt.tight_layout()
    plt.show()
