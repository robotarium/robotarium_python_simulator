# GO_TO_POSE_WITH_PLOTTING
# Drives N robots to randomized goal poses within the Robotarium arena.
# When all robots reach their goals, new goals are chosen and the process
# repeats for the duration of the simulation.
#
# Visualization includes:
#   - Colored circle markers tracking each robot's position
#   - Colored square markers and orientation arrows at each goal
#   - Per-robot position readouts
#   - Iteration count and elapsed time display
#
# Sean Wilson / Python port
# 07/2019

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_pose_parking_controller_clf
from rps.utilities.misc import generate_random_poses, create_at_pose

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N          = 6
iterations = 5000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_conditions = generate_random_poses(N, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                           initial_conditions=initial_conditions,
                           sim_in_real_time=True)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary()
controller  = create_pose_parking_controller_clf(position_error=0.05, rotation_error=0.2)
init_checker = create_at_pose(position_error=0.05, rotation_error=0.2)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
arena_width  = (r.BOUNDARIES[1] - r.BOUNDARIES[0]) - 3 * r.ROBOT_DIAMETER
arena_height = (r.BOUNDARIES[3] - r.BOUNDARIES[2]) - 3 * r.ROBOT_DIAMETER

goal_poses = generate_random_poses(N, width=arena_width, height=arena_height, spacing=0.5)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def determine_marker_size(robotarium_instance, marker_size_meters):
    """Marker size in points matching a physical width in metres."""
    fig  = robotarium_instance._fig
    ax   = robotarium_instance._axes_handle
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    arena_w = robotarium_instance.BOUNDARIES[1] - robotarium_instance.BOUNDARIES[0]
    return (marker_size_meters / arena_w) * bbox.width

def determine_robot_marker_size(robotarium_instance):
    """Marker size in points matching the physical robot diameter."""
    return determine_marker_size(
        robotarium_instance,
        robotarium_instance.ROBOT_DIAMETER + 0.03)

def determine_font_size(robotarium_instance, font_height_meters):
    """Font size in points matching a physical height in metres."""
    fig  = robotarium_instance._fig
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    arena_h  = robotarium_instance.BOUNDARIES[3] - robotarium_instance.BOUNDARIES[2]
    return fig_h_px * (font_height_meters / arena_h)

# =========================================================
# PLOT INITIALIZATION
# =========================================================
ax = r._axes_handle

# Draw GT logo as background image
try:
    gt_img = plt.imread(f'{os.path.dirname(os.path.realpath(__file__))}/GTLogo.png')
    
    # 1. DOWNSAMPLE: Slice the NumPy array to take every 3rd pixel. 
    # If it's still slow, increase this to 4 or 5. If it looks too blurry, drop it to 2.
    gt_img_small = gt_img[::3, ::3] 
    
    # 2. INTERPOLATION: 'nearest' stops Matplotlib from doing expensive 
    # anti-aliasing math on every single frame redraw.
    ax.imshow(gt_img_small, extent=[-1.0, 1.0, -1.0, 1.0], zorder=0, interpolation='nearest')
except FileNotFoundError:
    print("Warning: GTLogo.png not found, you need to execute this script in the directory of the image. Proceeding without background image.")

# Get initial poses before plotting so markers start in the right place
x = r.get_poses()

# One random colour per robot, consistent across all plot elements
rng    = np.random.default_rng(42)
CM     = rng.random((N, 3))

marker_size_goal  = determine_marker_size(r, 0.20)
marker_size_robot = determine_robot_marker_size(r)
font_size         = determine_font_size(r, 0.05)
line_width        = 5

# Per-robot plot handles
d = []             # goal square markers
a = []             # goal orientation arrows
g = []             # robot position circles
goal_labels  = []
robot_labels = []
robot_details_text = []

for i in range(N):
    c = CM[i]

    # Goal square marker (Unfilled)
    dm, = ax.plot(goal_poses[0, i], goal_poses[1, i], 's',
                  markersize=marker_size_goal, 
                  markerfacecolor='none',
                  markeredgecolor=c,
                  markeredgewidth=line_width,
                  zorder=2)
    d.append(dm)

    # Goal orientation arrow
    am = ax.quiver(goal_poses[0, i], goal_poses[1, i],
                   0.12 * np.cos(goal_poses[2, i]),
                   0.12 * np.sin(goal_poses[2, i]),
                   color=c, scale=1, scale_units='xy', angles='xy',
                   width=0.007, linewidth=line_width, zorder=2)
    a.append(am)

    # Goal label
    gl = ax.text(goal_poses[0, i] - 0.05, goal_poses[1, i],
                 f'G{i+1}', fontsize=font_size, fontweight='bold', zorder=2)
    goal_labels.append(gl)

    # Robot circle marker (Unfilled)
    gm, = ax.plot(x[0, i], x[1, i], 'o',
                  markersize=marker_size_robot, 
                  markerfacecolor='none',
                  markeredgecolor=c,
                  markeredgewidth=5,
                  zorder=5)
    g.append(gm)

    # Robot name label (off-screen initially)
    rl = ax.text(500, 500, f'Robot {i+1}',
                 fontsize=font_size, fontweight='bold', zorder=5)
    robot_labels.append(rl)

    # Robot position detail label (off-screen initially)
    rd = ax.text(500, 500, '',
                 fontsize=font_size, fontweight='bold', zorder=5)
    robot_details_text.append(rd)

# Iteration counter and elapsed time
start_time = time.time()
iteration_label = ax.text(
    r.BOUNDARIES[0] + 0.05, r.BOUNDARIES[2] + 0.1,
    'Iteration 0', fontsize=font_size, color='r', fontweight='bold', zorder=6)
time_label = ax.text(
    r.BOUNDARIES[0] + 0.05, r.BOUNDARIES[2] + 0.02,
    'Total Time Elapsed 0.00', fontsize=font_size, color='r',
    fontweight='bold', zorder=6)

# Step once before setting draw order so all objects exist
r.step()

# Draw order: goal elements behind robots, counters on top
for dm in d:
    dm.set_zorder(1)
for am in a:
    am.set_zorder(1)
for gl in goal_labels:
    gl.set_zorder(1)
iteration_label.set_zorder(6)
time_label.set_zorder(6)

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers and text when the figure window is resized."""
    # Recalculate sizes
    new_goal_size  = determine_marker_size(r, 0.20)
    new_robot_size = determine_robot_marker_size(r)
    font_size      = determine_font_size(r, 0.05)
    
    # Apply new sizes to all plot handles
    for j in range(N):
        d[j].set_markersize(new_goal_size)
        g[j].set_markersize(new_robot_size)
        robot_labels[j].set_fontsize(font_size)
        goal_labels[j].set_fontsize(font_size)
        robot_details_text[j].set_fontsize(font_size)
        
    iteration_label.set_fontsize(font_size)
    time_label.set_fontsize(font_size)
    
    # Request a fast redraw so elements visually scale while dragging
    ax.figure.canvas.draw_idle()

# Connect the callback function to the figure canvas
ax.figure.canvas.mpl_connect('resize_event', on_resize)

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for i in range(iterations):

    x = r.get_poses()

    if not init_checker(x, goal_poses)[0]:
        # Robots still moving: update robot marker positions and labels
        for q in range(N):
            g[q].set_xdata([x[0, q]])
            g[q].set_ydata([x[1, q]])
            robot_labels[q].set_position((x[0, q] - 0.15, x[1, q] + 0.15))
            robot_details_text[q].set_text(f'X: {x[0,q]:.2f}\nY: {x[1,q]:.2f}')
            robot_details_text[q].set_position((x[0, q] - 0.20, x[1, q] - 0.25))
    else:
        # All robots reached goals: generate new goal poses
        goal_poses = generate_random_poses(
            N, width=arena_width, height=arena_height, spacing=0.5)

        for j in range(N):
            d[j].set_xdata([goal_poses[0, j]])
            d[j].set_ydata([goal_poses[1, j]])
            a[j].set_offsets([goal_poses[0, j], goal_poses[1, j]])
            a[j].set_UVC(0.12 * np.cos(goal_poses[2, j]),
                         0.12 * np.sin(goal_poses[2, j]))
            goal_labels[j].set_position(
                (goal_poses[0, j] - 0.05, goal_poses[1, j]))

    # Update iteration counter and elapsed time
    iteration_label.set_text(f'Iteration {i}')
    time_label.set_text(f'Total Time Elapsed {time.time() - start_time:.2f}')

    # Compute and apply control
    dxu = controller(x, goal_poses)
    dxu = unicycle_barrier_certificate(dxu, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()