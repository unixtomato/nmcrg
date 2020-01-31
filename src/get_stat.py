import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pbs-jobdir', type=str, default='plots')
parser.add_argument('--ncpus', type=int, default=1)
parser.add_argument('--input-width', type=int, default=32)
args = parser.parse_args()

l = args.input_width
method = 'weight'

datas = []
for i in range(1, args.ncpus+1):
    datas.append(np.loadtxt("{}/ops_l{}_{}_{}.out".format(args.pbs_jobdir, l, method, i)))
datas = np.array(datas)

res = [np.mean(datas, axis=0), (np.std(datas, axis=0)/np.sqrt(args.ncpus-1))]
for r in range(res[0].shape[0]):
    print(res[0][r], res[1][r])
