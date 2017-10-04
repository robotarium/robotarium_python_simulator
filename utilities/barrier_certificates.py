from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import itertools
import numpy as np
from scipy.special import comb

from . import transformations

options['show_progress'] = False

def create_single_integrator_barrier_certificate(number_of_agents, barrier_gain=10000,safety_radius=0.08):
    """ TODO: comment
    """

    N = number_of_agents
    num_constraints = int(comb(N, 2))
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = 2*np.identity(2*N)

    def f(dxi, x):

        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
                A[count, (2*i, (2*i+1))] = -2*error
                A[count, (2*j, (2*j+1))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        f = -2*np.reshape(dxi, 2*N, order='F')

        result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_unicycle_barrier_certificate(number_of_agents, barrier_gain=8000, safety_radius=0.08, projection_distance=0.05, magnitude_limit=0.08):
    """ TODO: comment
    """
    N = number_of_agents
    num_constraints = int(comb(N, 2))
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = 2*np.identity(2*N)

    si_barrier_cert = create_single_integrator_barrier_certificate(N, barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance)

    dyn, to_si_states = create_single_integrator_to_unicycle(projection_distance=projection_distance)

    def f(dxu, states):
        """Barrier certificate function."""

        x_si = to_si_states(states)
        return dyn(si_barrier_cert(unicycle_to_single_integrator(dxu), states))

    return f
