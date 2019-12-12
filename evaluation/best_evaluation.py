import numpy as np
import json
import argparse
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='*info.json', help='file name/pattern (default: *info.json)')
    parser.add_argument('-i', '--dir', type=str, default='.', help='folder with info files')
    args = parser.parse_args()
    paths = glob.glob(os.path.join(args.dir, args.file))
    infos = []
    best = [float('inf'), 0]
    for i in range(len(paths)):
        with open(paths[i], 'r') as f:
            infos.append(json.load(f))
        meani = float(np.array(infos[i]['eval_results']).mean(1).min())
        print('%s:\t%.2f' % (infos[i]['argument']['name'], meani))
        if meani < best[0]:
            best = [meani, i]
    print("best eval result with score of %.2f at batch %i of %i (mean=%.2f)" % (best[0], -1+(best[1])*infos[best[1]]['argument']
                                                                                 ['evaluation_interval'], infos[best[1]]['batches_done'], np.array(infos[best[1]]['eval_results']).mean(1).mean()))
    print(infos[best[1]]['argument'])
