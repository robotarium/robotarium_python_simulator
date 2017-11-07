import robotarium
from utilities import graph
from utilities import transformations
from utilities.barrier_certificates import *

# Array representing the geometric distances betwen the agents.  In this case,
# the agents try to form a Rectangle
L = np.array([
    [3, -1, 0, -1, 0, -1],
    [-1, 3, -1, 0, -1, 0],
    [0, -1, 3, -1, 0, -1],
    [-1, 0 , -1, 3, -1, 0],
    [0, -1, 0, -1, 3, -1],
    [-1, 0, -1, 0, -1, 3]
])

# Some gains for this experiment.  These aren't incredibly relevant.
d = 0.2
ddiag = np.sqrt(5)*d
formation_control_gain = 4

# Weight matrix to control inter-agent distances
weights = np.array([
    [0, d, 0, d, 0, ddiag],
    [d, 0, d, 0, d, 0],
    [0, d, 0, ddiag, 0, d],
    [d, 0, ddiag, 0, d, 0],
    [0, d, 0, d, 0, d],
    [ddiag, 0, d, 0, d, 0]
])

# Experiment constants
iterations = 2000
N = 6

r = robotarium.Robotairum(number_of_agents=6)

si_barrier_cert = create_single_integrator_barrier_certificate(N)


for k in range(iterations):

    x = r.get_poses()
    dxi = np.zeros((2, N))

    for i in range(N):
        for j in graph.topological_neighbors(L, i):
            error = x[:2, j] - x[:2, i]
            dxi[:, i] += formation_control_gain*(np.power(np.linalg.norm(error), 2)- np.power(weights[i, j], 2)) * error

    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = transformations.single_integrator_to_unicycle2(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()
