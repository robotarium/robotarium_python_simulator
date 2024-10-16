import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# How many robots we want to use in the simulation
N = 10
# Instantiate the Robotarium object with these parameters
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# How many times should the robots form the circle?
num_cycles=2
count = -1 # How many times have they formed the circle? (starts at -1 since initial formation will increment the count)

# Default Barrier Parameters
safety_radius = 0.17

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

# Initial plots
x=r.get_poses()
g = r.axes.scatter(x[0,:], x[1,:], s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=7)
#g = r.axes.plot(x[0,:], x[1,:], markersize=safety_radius_marker_size, linestyle='none', marker='o', markerfacecolor='none', markeredgecolor=color[CM],linewidth=7)
r.step()

# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate()

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# This portion of the code generates points on a circle enscribed in a 6x6 square
# that's centered on the origin.  The robots switch positions on the circle.
xybound = np.array([-1.2, 1.2, -1, 1])
p_theta = 2*np.pi*(np.arange(0, 2*N, 2)/(2*N))
p_circ = np.vstack([
            np.hstack([xybound[1]*np.cos(p_theta), xybound[1]*np.cos(p_theta+np.pi)]),
            np.hstack([xybound[3]*np.sin(p_theta), xybound[3]*np.sin(p_theta+np.pi)])
            ])

# These variables are so we can tell when the robots should switch positions
# on the circle.
flag = 0
x_goal = p_circ[:, :N]

# Perform the simulation for a certain number of iterations
while(1):

    # Get the poses of the agents that we want
    x = r.get_poses()

    # Update Plotted Visualization
    g.set_offsets(x[:2,:].T)
    # This updates the marker sizes if the figure window size is changed. 
    # This should be removed when submitting to the Robotarium.
    g.set_sizes([determine_marker_size(r,safety_radius)])

    # We're planning on using the single-integrator to unciycle mapping,
    # so our single-integrator states are the projected point.
    x_si = uni_to_si_states(x)

    # Initialize a velocities variable
    si_velocities = np.zeros((2, N))

    # Check if all the agents are close enough to the goals
    if(np.linalg.norm(x_goal - x_si) < 0.05):
        flag = 1-flag
        count += 1

    if count == num_cycles:
        break

    # Switch goals depending on the state of flag (goals switch to opposite
    # sides of the circle)
    if(flag == 0):
        x_goal = p_circ[:, :N]
    else:
        x_goal = p_circ[:, N:]

    # Use a position controller to drive to the goal position
    dxi = si_position_controller(x_si,x_goal)

    # Use the barrier certificates to make sure that the agents don't collide
    dxi = si_barrier_cert(dxi, x_si)

    # Use the second single-integrator-to-unicycle mapping to map to unicycle
    # dynamics
    dxu = si_to_uni_dyn(dxi, x)

    # Set the velocities of agents 1,...,N to dxu
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

finished_caption = "All robots safely reached \n their destination"
finished_label = r.axes.text(0,0,finished_caption,fontsize=font_height_points, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=20)
r.step()
time.sleep(5)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
