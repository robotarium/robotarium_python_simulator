from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import itertools
import numpy as np
from scipy.special import comb

options['show_progress'] = False

def create_single_integrator_barrier_certificate(barrier_gain=10000,safety_radius=0.06):

    def f(dxi, x):
        _,N = np.shape(dxi)

        num_constraints = int(comb(N, 2))
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)

        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - safety_radius*safety_radius
                A[count, (2*i, (2*i+1))] = -2*error
                A[count, (2*j, (2*j+1))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        H = 2*np.identity(2*N)
        f = -2*np.reshape(dxi, (2*N, -1))
        result = qp(sparse(matrix(H)), matrix(f), sparse(matrix(A)), matrix(b))['x']



        return np.reshape(result, (2, -1))

    return f
