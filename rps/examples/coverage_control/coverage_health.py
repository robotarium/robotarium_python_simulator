import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import pandas as pd

# Instantiate Robotarium object
N = 5
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# How many iterations do we want (about N*0.033 seconds)
iterations = 1000

# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Generated a connected graph Laplacian (for a cylce graph).
L = lineGL(N)

x_min = -1.5
x_max = 1.5
y_min = -1
y_max = 1
res = 0.05

 
cost = []
wij = []
pose = []
cwi = []
dxi = []
ui = []
dist_robots = np.zeros(N)
cumulative_dist = []
covergence = 0

# Robot configuration
sensor_type = {1: [1,2,3], 2: [3,4,5]}
healths = [1,1,0.5,1,1]
mobility = [1,1,1,1,1]
sensor_weights = np.zeros(N)


for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]
  
    c_v = np.zeros((N,2)) # centroid vector
    w_v = np.zeros(N) # weight vector
    
    
    power_cost = 0
    for ix in np.arange(x_min,x_max,res):
        for iy in np.arange(y_min,y_max,res):
            importance_value = 1
            distances = np.zeros(N)
            for robots in range(N):
                distances[robots] = np.square(ix - current_x[robots]) + np.square(iy - current_y[robots]) - healths[robots]
            # a list of N distances from the current point to each robot
            min_index = np.argmin(distances) # get the index of the robot that is closest to the current point
            c_v[min_index][0] += ix * importance_value
            c_v[min_index][1] += iy * importance_value
            w_v[min_index] += 1
            power_cost += distances[min_index] * importance_value * (res ** 2) / 2

    wij.append(w_v)
    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))
    gain = 1
    for robots in range(N):
       c_x = 0
       c_y = 0
       if not w_v[robots] == 0:
          c_x = c_v[robots][0] / w_v[robots]
          c_y = c_v[robots][1] / w_v[robots]  
          si_velocities[:, robots] = gain * np.array([(c_x - current_x[robots][0]), (c_y - current_y[robots][0])])


    # If convergence is reached
    if len(cost) > 1 and abs(power_cost - cost[-1]) < 0.00001: 
        convergence = k
    cost.append(power_cost)
    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

# Save the data to csv file
df = pd.DataFrame(cost)
df.to_csv('cost_case5.csv', index=False)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()