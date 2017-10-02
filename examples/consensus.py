import robotarium
import utilities.graph as graph
import utilities.transformations as transformations
from utilities.barrier_certificates import *

import numpy as np

# Instantiate Robotarium object
N = 10
r = robotarium.Robotarium(number_of_agents=N)

si_to_uni_dyn = transformations.create_single_integrator_to_unicycle2()

L = graph.cycle_GL(N)

si_barrier_cert = create_single_integrator_barrier_certificate()

xybound = [-0.2, 0.2, -0.5, 0.5]
p_theta = range(1, 2*N, 2)
p_circ = np.hstack([xybound[1]*np.cos(p_theta), xybound[1]*np.cos(p_theta+np.pi), xybound[3]*np.sin(p_theta), xybound[3]*np.sin(p_theta+np.pi)])

flag = 0
x_goal = p_circ[:, :N]

for k in range(1000):

    # Do stuff
    x = r.get_poses()
    x_si = x[:2, :]
    si_velocities = np.zeros((2, N))

    if(np.linalg.norm(x_goal - x_si) < 0.08):
        flag = 1-flag

    if(flag == 0):
        x_goal = p_circ[:, :N]
    else:
        x_goal = p_circ[:, N:]


    for i in range(N):
        j = graph.topological_neighbors(L, i)
        si_velocities[:, i] = np.sum(x_si[:, j] - x_si[:, i, None], 1)

    si_velocities = si_barrier_cert(si_velocities, x_si)

    r.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x))
    r.step()
