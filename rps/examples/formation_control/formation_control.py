#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

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
d = 0.3
ddiag = np.sqrt(5)*d
formation_control_gain = 10

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

#Limit maximum linear speed of any robot
magnitude_limit = 0.15

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics()

for k in range(iterations):

    # Get the poses of the robots
    x = r.get_poses()

    # Initialize a velocity vector
    dxi = np.zeros((2, N))

    for i in range(N):
        for j in topological_neighbors(L, i):
            # Perform a weighted consensus to make the rectangular shape
            error = x[:2, j] - x[:2, i]
            dxi[:, i] += formation_control_gain*(np.power(np.linalg.norm(error), 2)- np.power(weights[i, j], 2)) * error

    #Keep single integrator control vectors under specified magnitude
    # Threshold control inputs
    norms = np.linalg.norm(dxi, 2, 0)
    idxs_to_normalize = (norms > magnitude_limit)
    dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

    # Make sure that the robots don't collide
    dxi = si_barrier_cert(dxi, x[:2, :])

    # Transform the single-integrator dynamcis to unicycle dynamics
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities of the robots
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()