import argparse
import sys
import subprocess
import os
import shutil

import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general/data
    parser.add_argument('--input-width', type=int, default=32, metavar='N',
                        help='linear dimension of model')
    parser.add_argument('--n-train', type=int, default=10000, metavar='N',
                        help='number of training samples')

    parser.add_argument('--data-folder', type=str, default='{}/datasets'.format(os.getcwd()), metavar='S',
                        help='folder that store training data set')

    args = parser.parse_args()


    # generate data
    if not os.path.isdir(args.data_folder):
        os.makedirs(args.data_folder)

    if not os.path.exists(args.data_folder + '/ising_l%d.dat' % args.input_width):
        print('... data not found; compiling', file=sys.stderr)

        try:
            subprocess.call(['g++', '-std=c++11', '-O2',
                             '-DL=%d' % args.input_width,
                             '-DNCONF=%d' % args.n_train, 
                             'src/wolff.cpp'])
        except OSError:
            print('... compilation failed', file=sys.stderr)
            sys.exit()
        else:
            print('... compilation successiful', file=sys.stderr)

        subprocess.call(['./a.out', args.data_folder])
        print('... data generated', file=sys.stderr)
        os.remove('a.out')
    else:
        print('... data exists', file=sys.stderr)


if __name__ == '__main__':
    main()
