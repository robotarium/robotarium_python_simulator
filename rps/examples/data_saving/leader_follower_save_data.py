'''
 Same script as leader_follower.m but with additional data saving. Two
 data sets will be saved, one saving the distance between connected robots
 through time, and another with the distance between the leader and goal
 location when the goal is "reached". They will each be saved as .npy files
 and human readable csv .txt files.

 Sean Wilson
 10/2019
'''

#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

#Other Imports
import numpy as np
import time

# Experiment Constants
iterations = 5000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
N=4 #Number of robots to use, this must stay 4 unless the Laplacian is changed.

waypoints = np.array([[-1, -1, 1, 1],[0.8, -0.8, -0.8, 0.8]]) #Waypoints the leader moves to.
close_enough = 0.03; #How close the leader must get to the waypoint to move to the next one.

# Preallocate data saving
robot_distance = np.zeros((5,iterations)) #Saving 4 inter-robot distances and time
goal_distance = np.empty((0,2))
start_time = time.time()

# Create the desired Laplacian
followers = -completeGL(N-1)
L = np.zeros((N,N))
L[1:N,1:N] = followers
L[1,1] = L[1,1] + 1
L[1,0] = -1

# Find connections
[rows,cols] = np.where(L==1)

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2,N))

#Initialize leader state
state = 0

#Limit maximum linear speed of any robot
magnitude_limit = 0.15

# Create gains for our formation control algorithm
formation_control_gain = 10
desired_distance = 0.3

# Initial Conditions to Avoid Barrier Use in the Beginning.
initial_conditions = np.array([[0, 0.5, 0.3, -0.1],[0.5, 0.5, 0.2, 0],[0, 0, 0, 0]])

# Instantiate the Robotarium object with these parameters
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=False)

# Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
# Single-integrator -> unicycle dynamics mapping
si_to_uni_dyn ,uni_to_si_states = create_si_to_uni_mapping()
# Single-integrator barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
# Single-integrator position controller
leader_controller = create_si_position_controller(velocity_magnitude_limit=0.1)

for t in range(iterations):

	# Get the most recent pose information from the Robotarium. The time delay is
	# approximately 0.033s
	x = r.get_poses()
	xi = uni_to_si_states(x)

	#Algorithm

	#Followers
	for i in range(1,N):
		# Zero velocities and get the topological neighbors of agent i
		dxi[:,[i]]=np.zeros((2,1))
		neighbors = topological_neighbors(L,i)

		for j in neighbors:
			dxi[:,[i]] += formation_control_gain*(np.power(np.linalg.norm(x[:2,[j]]-x[:2,[i]]), 2)-np.power(desired_distance, 2))*(x[:2,[j]]-x[:2,[i]])

	#Leader
	waypoint = waypoints[:,state].reshape((2,1))

	dxi[:,[0]] = leader_controller(x[:2,[0]], waypoint)
	if np.linalg.norm(x[:2,[0]] - waypoint) < close_enough:
		state = (state + 1)%4
		goal_distance = np.append(goal_distance, np.array([[np.linalg.norm(xi[:,[0]] - waypoint)],[time.time()-start_time]]))


	#Keep single integrator control vectors under specified magnitude
	# Threshold control inputs
	norms = np.linalg.norm(dxi, 2, 0)
	idxs_to_normalize = (norms > magnitude_limit)
	dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

	#Use barriers and convert single-integrator to unicycle commands
	dxi = si_barrier_cert(dxi, x[:2,:])
	dxu = si_to_uni_dyn(dxi,x)

	# Set the velocities of agents 1,...,N to dxu
	r.set_velocities(np.arange(N), dxu)

	# Compute data to be saved and stored in matrix
	# Distance between connected robots
	robot_distance[0,t] = np.linalg.norm(xi[:,[0]]-xi[:,[1]])
	robot_distance[4,t] = time.time() - start_time
	for j in range(1,int(len(rows)/2)+1):
		robot_distance[j,t] = np.linalg.norm(xi[:,[rows[j]]]-xi[:,[cols[j]]])

	# Iterate the simulation
	r.step()

#Save Data Locally as Numpy
np.save('goal_distance_data', goal_distance)
np.save('inter_robot_distance_data', robot_distance)

#Save Data Locally as CSV Text File
np.savetxt('goal_distance_data.txt', goal_distance, delimiter=',')
np.savetxt('inter_robot_distance_data.txt', robot_distance.T, delimiter=',')

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
