import numpy as np
import json
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, default='*info.json', help='file name/pattern (default: *info.json)')
parser.add_argument('-i', '--dir', type=str, default='.', help='folder with info files')
args = parser.parse_args()
paths = glob.glob(os.path.join(args.dir, args.file))

for i in range(len(paths)):
    with open(paths[i], 'r') as f:
        info = json.load(f)
        results = np.array(info['saved_batch'])
        print(os.path.basename(paths[i]))
        print(results)
        print('-------------------------')