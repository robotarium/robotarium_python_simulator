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
L = completeGL(N)

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
sensor_types = {1: [0, 1, 2], 2: [2, 3, 4]}  # Sensor type 1: robots 1, 2, 3; Sensor type 2: robots 3, 4, 5
importance = {
    1: {0: 1, 1: 1, 2: 1},  # Importance for sensor type 1 (robots 0, 1, 2)
    2: {2: 1, 3: 1, 4: 1}   # Importance for sensor type 2 (robots 2, 3, 4)
}
for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]
  
    c_v = np.zeros((N,2)) # centroid vector
    w_v = np.zeros(N) # weight vector
    
    
    locational_cost = 0
    for ix in np.arange(x_min,x_max,res):
        for iy in np.arange(y_min,y_max,res):
            for sensor_type, robots in sensor_types.items():
                distances = np.zeros(N)
                
                for robot in robots:
                    distances[robot] = np.sqrt(np.square(ix - current_x[robot]) + np.square(iy - current_y[robot])) 
                # a list of N distances from the current point to each robot
                #print(distances)
                #min_index = np.argmin(distances) # get the index of the robot that is closest to the current point
              
                # Find the closest robot within the current sensor type
                min_index = robots[np.argmin([distances[r] for r in robots])]

                # Get importance value for the closest robot
                importance_value = importance[sensor_type].get(min_index, 1.0)

                c_v[min_index][0] += ix * importance_value
                c_v[min_index][1] += iy * importance_value
                w_v[min_index] += importance_value
                locational_cost += distances[min_index] * importance_value * (res ** 2)

    wij.append(w_v)
    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))
    for robots in range(N):
       c_x = 0
       c_y = 0
       if not w_v[robots] == 0:
          c_x = c_v[robots][0] / w_v[robots]
          c_y = c_v[robots][1] / w_v[robots] 
                        
          si_velocities[:, robots] = 1 * [(c_x - current_x[robots][0]), (c_y - current_y[robots][0] )]

    # If convergence is reached
    if len(cost) > 1 and abs(locational_cost - cost[-1]) < 0.0001: 
        convergence = k
    cost.append(locational_cost)
   
    # Use the barrier certificate to avoid collisions
    #si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

# Save the data to csv file
df = pd.DataFrame(cost)
df.to_csv('cost_case1.csv', index=False)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()