from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

import itertools
import numpy as np
from scipy.special import comb

from rps.utilities.transformations import *
import timeit
import time

# Disable output of CVXOPT
options['show_progress'] = False

def create_single_integrator_barrier_certificate(number_of_agents, barrier_gain=100, safety_radius=0.15, magnitude_limit=0.35):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    number_of_agents: int (number of agents.  should be a constant)
    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)

    -> function (the barrier certificate function)
    """

    # Initialize some variables for computational savings
    N = number_of_agents
    num_constraints = int(comb(N, 2))
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = sparse(matrix(2*np.identity(2*N)))

    def f(dxi, x):
        """ TODO: comment"""
        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1))] = -2*error
                A[count, (2*j, (2*j+1))] = 2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_unicycle_barrier_certificate(number_of_agents, barrier_gain=80, safety_radius=0.15, projection_distance=0.03, magnitude_limit=0.4):
    """ TODO: Creates a unicycle barrier cetifcate to avoid collisions.  For
    optimization purposes, this function returns another function.

    number_of_agents: int
    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """
    N = number_of_agents
    num_constraints = int(comb(N, 2))
    A = np.zeros((num_constraints, 2*N))
    b = np.zeros(num_constraints)
    H = 2*np.identity(2*N)

    si_barrier_cert = create_single_integrator_barrier_certificate(N, barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance)

    dyn, to_si_states = create_single_integrator_to_unicycle(projection_distance=projection_distance)

    def f(dxu, states):
        """TODO: comment """

        x_si = to_si_states(states)
        return dyn(si_barrier_cert(unicycle_to_single_integrator(dxu, states), x_si), states)

    return f
