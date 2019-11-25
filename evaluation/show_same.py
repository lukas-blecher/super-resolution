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
    names = [hyper2float(os.path.basename(subdirs[i])) for i in range(len(subdirs))]
    try:
        perm = np.argsort(names)
        subdirs = np.array(subdirs)[perm]
        names = np.array(names)[perm]
    except:
        pass
    N = len(subdirs)
    a = int(np.sqrt(N))
    b = int(np.ceil(N/a))
    f, ax = plt.subplots(a, b, sharex=True, sharey=True)
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
        axi.imshow(plt.imread(file))
        axi.axis('off')
    for i in range(a*b-1, N-1, -1):
        f.delaxes(axf[i])

    # plt.tight_layout()
    plt.show()
