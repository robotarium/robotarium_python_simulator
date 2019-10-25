import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 5
initial_conditions = np.array(np.mat('1 0.5 -0.5 0 0.28; 0.8 -0.3 -0.75 0.1 0.34; 0 0 0 0 0'))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

# Define goal points by removing orientation from poses
goal_points = generate_initial_conditions(N, width=r.boundaries[2]-2*r.robot_diameter, height = r.boundaries[3]-2*r.robot_diameter, spacing=0.5)

# Create unicycle pose controller
unicycle_pose_controller = create_hybrid_unicycle_pose_controller()

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate()

# define x initially
x = r.get_poses()

# Plotting Parameters
CM = np.random.rand(N,3) # Random Colors
goal_marker_size_m = 0.2
robot_marker_size_m = 0.15
marker_size_goal = determine_marker_size(r,goal_marker_size_m)
marker_size_robot = determine_marker_size(r, robot_marker_size_m)
font_size = determine_font_size(r,0.1)
line_width = 5

# Create Goal Point Markers
#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Arrow for desired orientation
goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.02, length_includes_head=True, color = CM[ii,:], zorder=-2)
for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]
robot_markers = [r.axes.scatter(x[0,ii], x[1,ii], s=marker_size_robot, marker='o', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width) 
for ii in range(goal_points.shape[1])]

r.step()

# While the number of robots at the required poses is less
# than N...
while (np.size(at_pose(x, goal_points)) != N):

    # Get poses of agents
    x = r.get_poses()

    #Update Plot
    # Update Robot Marker Plotted Visualization
    for i in range(x.shape[1]):
        robot_markers[i].set_offsets(x[:2,i].T)
        # This updates the marker sizes if the figure window size is changed. 
        # This should be removed when submitting to the Robotarium.
        robot_markers[i].set_sizes([determine_marker_size(r, robot_marker_size_m)])

    for j in range(goal_points.shape[1]):
        goal_markers[j].set_sizes([determine_marker_size(r, goal_marker_size_m)])

    # Create unicycle control inputs
    dxu = unicycle_pose_controller(x, goal_points)

    # Create safe control inputs (i.e., no collisions)
    dxu = uni_barrier_cert(dxu, x)

    # Set the velocities
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
