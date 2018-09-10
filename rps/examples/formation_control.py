import rps.robotarium as robotarium
from rps.utilities import graph
from rps.utilities import transformations
from rps.utilities.barrier_certificates import *

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

r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=True, update_time=1)

si_barrier_cert = create_single_integrator_barrier_certificate(N)


for k in range(iterations):

    # Get the poses of the robots
    x = r.get_poses()

    # Initialize a velocity vector
    dxi = np.zeros((2, N))

    for i in range(N):
        for j in graph.topological_neighbors(L, i):
            # Perform a weighted consensus to make the rectangular shape
            error = x[:2, j] - x[:2, i]
            dxi[:, i] += formation_control_gain*(np.power(np.linalg.norm(error), 2)- np.power(weights[i, j], 2)) * error

    # Make sure that the robots don't collide
    dxi = si_barrier_cert(dxi, x[:2, :])

    # Transform the single-integrator dynamcis to unicycle dynamics
    dxu = transformations.single_integrator_to_unicycle2(dxi, x)

    # Set the velocities of the robots
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

# Always call this at the end of your scripts!! It will acccelerate your execution time.
r.call_at_scripts_end()
