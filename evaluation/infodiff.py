import numpy as np
import json
import argparse
import os
import glob
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def round_list(it, digits=5):
    n = []
    for x in list(it):
        n.append(round(x, digits))
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True, help='info file with loss')
    args = parser.parse_args()
    if len(args.file) == 1:
        args.file = glob.glob(args.file[0])
    assert len(args.file) >= 2
    d = None
    for i, f in enumerate(args.file):
        with open(f) as fs:
            info = json.load(fs)

        if d is None:
            d = {k: [] for k in info['argument'].keys()}
            d['saved_batch'] = []
        for k in d.keys():
            try:
                d[k].append(info['argument'][k])
            except KeyError:
                pass
        d['saved_batch'].append(round_list(list(info['saved_batch'].values())[-1]))
    delkeys = []
    for k in d.keys():
        same = True
        v0 = None
        for v in d[k]:
            if v0 is None:
                v0 = v
            if v != v0:
                same = False
                break
            v0 = v
        if same:
            delkeys.append(k)
    for k in delkeys:
        del d[k]
    df = pd.DataFrame(d)

    print(df)
