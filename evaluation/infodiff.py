import numpy as np
import json
import argparse
import os
import glob
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


def round_list(it, digits=5):
    n = []
    for x in list(it):
        n.append(round(x, digits))
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True, help='info file with loss')
    parser.add_argument('-d', '--delkeys', type=str, nargs='+', default=[], help='which keys not to show')
    args = parser.parse_args()
    files=[]
    for f in args.file:
        files.extend(glob.glob(f))
    files = list(set(files))
    assert len(files) >= 2
    d = None
    for i, f in enumerate(files):
        with open(f) as fs:
            try:
                info = json.load(fs)
            except UnicodeDecodeError:
                print('Error:', f)
                continue
        if d is None:
            d = {k: [] for k in info['argument'].keys()}
            d['saved_batch'] = []
        other_keys=list(set(list(d.keys())+list(info['argument'].keys())))
        for k in info['argument'].keys():
            del other_keys[other_keys.index(k)]
            try:
                d[k].append(info['argument'][k])
            except KeyError as e:
                # print(e, f)
                d[k] = [None]*i + [info['argument'][k]]
        for k in other_keys:
            if k in ('saved_batch'):
                continue
            d[k].append(None)
        try:
            it=list(info['saved_batch'].values())
        except:
            print(f)
            continue
        if type(it[0])==list:
            it=it[-1]
        else:
            print(f,it)
            continue
        d['saved_batch'].append([list(info['saved_batch'].keys())[-1],*round_list(it)])
    delkeys = args.delkeys.copy()
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

    for k in ['validation_path','testset_path','eval_modes','dataset_path']:
        if k not in delkeys:
            delkeys.append(k)
    for k in delkeys:
        del d[k]
    try:
        #df = pd.DataFrame(d)
        df=pd.DataFrame.from_dict(d, orient='index').T
        df.sort_values('name', inplace=True)    
        print(df)
    except ValueError as e:
        print(e)
        print(d)
        
