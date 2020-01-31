import os
import sys
import pickle
import subprocess
import shutil
import random

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='{}/plots/log'.format(os.getcwd()))
parser.add_argument('--epoch', type=int, default=-1)
parser.add_argument('--ncpus', type=int, default=10)
args = parser.parse_args()

dd = []
d = {}
with open(args.log, 'r') as f:
    for line in f:
        if not line == '\n':
            tok = line.split()
            d[tok[0]] = tok[1]
        else:
            dd.append(d)
            d = {}

print('log file is {}'.format(args.log), file=sys.stderr)

for d in dd: 

    epoch = int(d['epochs']) - 1 if args.epoch == -1 else args.epoch

    mcrg_folder = '{}/mcrg_at_epoch_{}'.format(d['plot_folder'], epoch)

    if os.path.isdir(mcrg_folder):
        print('{} exists removing'.format(mcrg_folder), file=sys.stderr)
        shutil.rmtree(mcrg_folder)

    os.makedirs(mcrg_folder)


    randint = random.randint(0, 100000)

    params = np.array(pickle.load(open('{}/models_at_epoch_{}.pkl'.format(d['plot_folder'], epoch), 'rb')))
    
    np.array(params[0].flatten()).astype(np.float32).tofile('{}/weight.dat'.format(mcrg_folder))
    
    input_width = int(d['input_width'])
    nr = int(np.log2(input_width)) - 1 
    str_arr = ''
    for i in range(nr):
        str_arr += 'int s%d[%d][%d]; ' % (i, input_width//2**i, input_width//2**i)
    str_arr += 'void *ptrs[] = {'
    for i in range(nr):
        str_arr += 's%d,' % (i) 
    str_arr = str_arr[:-1]
    str_arr += '};'

    exe_name = 'cal.{}.{}'.format(randint, epoch)
    subprocess.call(['g++', '-std=c++11', '-O2',
                     '-DL=%d' % input_width,
                     '-DNR=%d' % nr, 
                     '-DCASCADE=%s' % str_arr,
                     '-DLW=%s' % d['filter_width'],
                     'src/cal_ops.cpp', '-o', exe_name])

    command = 'seq {} | xargs -I{{}} -n 1 -P {} ./{} {} {} {{}} '.format(
                args.ncpus, args.ncpus, exe_name, mcrg_folder, 'ops.txt')
    os.system(command)
    os.remove(exe_name)
    
    nc = sum(1 for line in open('src/ops.txt'))
    command = ('seq {} | xargs -I{{}} -n 1 -P {} python src/mcrg.py '
               '--pbs-jobdir {} --cpu {{}} --input-width {} --n-reno {} --n-coup {} ').format(
                args.ncpus, args.ncpus, mcrg_folder, input_width, nr, nc)
    os.system(command)

    res_folder = d['plot_folder']+'/res'
    if not os.path.isdir(res_folder):
        os.makedirs(res_folder)

    command = 'python src/get_stat.py --pbs-jobdir {} --ncpus {} --input-width {} 1> {}'.format(
                mcrg_folder, args.ncpus, input_width, '{}/{}.out'.format(res_folder, epoch))
    os.system(command)



