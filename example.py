import argparse
import sys
import subprocess
import os
import shutil

import numpy as np
import pickle

from src.rbm import test_rbm

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general/data
    parser.add_argument('--input-width', type=int, default=32, metavar='N',
                        help='linear dimension of model')

    # RBM related
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--n-gibbs-steps', type=int, default=3, metavar='N',
                        help='number of Gibbs updates per weights update')
    parser.add_argument('--filter-width', type=int, default=8, metavar='N',
                        help='linear dimension of filter')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='F',
                        help='learning rate of ADAM')


    parser.add_argument('--data-folder', type=str, default='{}/datasets'.format(os.getcwd()), metavar='S',
                        help='folder that store training data set')
    parser.add_argument('--plot-folder', type=str, default='{}/plots'.format(os.getcwd()), metavar='S')

    args = parser.parse_args()


    if os.path.isdir(args.plot_folder):
        shutil.rmtree(args.plot_folder)
    os.makedirs(args.plot_folder)

    # logging
    logfile = '{}/log'.format(args.plot_folder)
    with open(logfile, 'a') as f:
        for arg, value in vars(args).items():
            f.write('{:20}{}\n'.format(arg, value))
        f.write('{:20}{}\n'.format('plot_folder', args.plot_folder))
        f.write('\n')


    # training
    test_rbm(
        data_folder=args.data_folder,
        dataset='ising_l%d.dat' % args.input_width,
        input_width=args.input_width,
        plot_folder=args.plot_folder,
        filter_width=args.filter_width,
        batch_size=args.batch_size,
        gibbs_steps=args.n_gibbs_steps,
        training_epochs=args.epochs,
        starter_learning_rate=args.learning_rate
    )



if __name__ == '__main__':
    main()
