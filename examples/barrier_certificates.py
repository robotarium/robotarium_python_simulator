import robotarium
import utilities.graph as graph
import utilities.transformations as transformations
from utilities.barrier_certificates import *
from utilities.controllers import *

import numpy as np

# Instantiate Robotarium object
N = 10
r = robotarium.Robotarium(number_of_agents=N)

L = graph.cycle_GL(N)

si_barrier_cert = create_single_integrator_barrier_certificate(N)

xybound = [-0.5, 0.5, -0.2, 0.2]
p_theta = 2*np.pi*(np.arange(0, 2*N, 2)/(2*N))
p_circ = np.vstack([
            np.hstack([xybound[1]*np.cos(p_theta), xybound[1]*np.cos(p_theta+np.pi)]),
            np.hstack([xybound[3]*np.sin(p_theta), xybound[3]*np.sin(p_theta+np.pi)])
            ])

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


    dxi = single_integrator_position_controller(x_si, x_goal, gain=1, magnitude_limit=1)

    dxi = si_barrier_cert(dxi, x_si)
    dxu = transformations.single_integrator_to_unicycle2(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()
