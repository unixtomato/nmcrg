import numpy as np
import timeit
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pbs-jobdir', type=str, default='plots')
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--input-width', type=int, default=32)
parser.add_argument('--n-reno', type=int, default=4)
parser.add_argument('--n-coup', type=int, default=4)

args = parser.parse_args()



def transformation_matrix(s):
    n_coup = s.shape[-1] # number of couplings

    A = []; B = []
    for i in range(n_coup):
        for j in range(n_coup):
            A.append(np.mean(s[:,1,i] * s[:,1,j]) - np.mean(s[:,1,i]) * np.mean(s[:,1,j]))
            B.append(np.mean(s[:,1,i] * s[:,0,j]) - np.mean(s[:,1,i]) * np.mean(s[:,0,j]))

    T = np.linalg.solve(np.array(A).reshape(n_coup, n_coup), np.array(B).reshape(n_coup, n_coup))
    W, V = np.linalg.eig(T)
    return np.sort(np.real(W))


def mcrg():

    n_coup = args.n_coup
    method = "weight"

    l = args.input_width
    n_reno = args.n_reno

    start_time = timeit.default_timer()

    # read in operators (correlation function)
    data = np.fromfile("{}/ops_l{}_{}_{}.dat".format(args.pbs_jobdir, l, method, args.cpu), dtype=np.int32)
    data = data.reshape(-1, n_reno, n_coup) / l**2


    vals = []


    for r in range(n_reno-1):
        for i in range(0, n_coup):
            res = transformation_matrix(data[:, r:r+2, :i+1])

            # calculaute exponents of interest
            res = res[-1:]
            res = np.log(res)/np.log(2)
            vals.append(res)

    np.savetxt("{}/ops_l{}_{}_{}.out".format(args.pbs_jobdir, l, method, args.cpu), np.array(vals))

    end_time = timeit.default_timer()
    #print("took", (end_time - start_time), "seconds", file=sys.stderr)



if __name__ == '__main__':
    mcrg()
