# BARRIERS_WITH_PLOTTING
# Demonstrates the unicycle barrier certificate with plotting.
# Ten robots swap positions on opposite sides of an ellipse three times
# while avoiding collisions. Colored circle markers track each robot.
#
# Sean Wilson / Python port
# 3/2026

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
from rps.utilities.controllers import create_si_position_controller

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N                   = 10
safety_radius       = 0.15
projection_distance = 0.03

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=True)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
xybound = np.array([-1.0, 1.0, -0.8, 0.8])
p_theta = 2 * np.pi * (np.arange(1, 2*N, 2) / (2*N))
p_circ  = np.vstack([
    np.hstack([xybound[1] * np.cos(p_theta), xybound[1] * np.cos(p_theta + np.pi)]),
    np.hstack([xybound[3] * np.sin(p_theta), xybound[3] * np.sin(p_theta + np.pi)]),
])

x_goal = p_circ[:, :N]
flag   = 0   # 0 = first half of ellipse, 1 = second half
count  = 0   # number of completed swaps

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
position_int = create_si_position_controller(
    x_velocity_gain=2.0, y_velocity_gain=2.0, velocity_magnitude_limit=0.15)

si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
    projection_distance=projection_distance)

uni_barrier_certificate = create_uni_barrier_certificate(
    barrier_gain=500, safety_radius=safety_radius,
    projection_distance=projection_distance)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def determine_marker_size(robotarium_instance, marker_size_meters):
    """Marker size in points matching a physical width in metres."""
    fig   = robotarium_instance._fig
    ax    = robotarium_instance._axes_handle
    fig.canvas.draw()
    bbox  = ax.get_window_extent()
    arena_w = robotarium_instance.BOUNDARIES[1] - robotarium_instance.BOUNDARIES[0]
    return (marker_size_meters / arena_w) * bbox.width

def determine_font_size(robotarium_instance, font_height_meters):
    """Font size in points matching a physical height in metres."""
    fig   = robotarium_instance._fig
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    arena_h  = robotarium_instance.BOUNDARIES[3] - robotarium_instance.BOUNDARIES[2]
    return fig_h_px * (font_height_meters / arena_h)

# =========================================================
# PLOT INITIALIZATION
# =========================================================
ax = r._axes_handle

# Get initial poses before plotting so markers start in the right place
x = r.get_poses()

# One colour per robot — using Matplotlib's distinct 'tab10' qualitative colormap
cmap = plt.get_cmap('tab10')
colors = [cmap(i % 10) for i in range(N)]

marker_size_robot = determine_marker_size(r, 1.5 * safety_radius)

# One circle marker per robot
g = []
for i in range(N):
    dot, = ax.plot(x[0, i], x[1, i], 'o',
                   markersize=marker_size_robot, 
                   markerfacecolor='none',        # Makes the inside transparent
                   markeredgecolor=colors[i],     # Applies the robot's color to the outline
                   markeredgewidth=4,             # Adjusts the thickness of the ring
                   zorder=1)
    g.append(dot)

# Step once to render robots before the main loop
r.step()

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers when the figure window is resized."""
    new_size = determine_marker_size(r, 1.5 * safety_radius)
    for dot in g:
        dot.set_markersize(new_size)
    
    ax.figure.canvas.draw_idle()

ax.figure.canvas.mpl_connect('resize_event', on_resize)

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
while True:

    x  = r.get_poses()
    xi = uni_to_si_states(x)

    # Termination: exit after three completed swaps
    if count == 3:
        break

    # Goal selection: alternate between the two ellipse halves
    if flag == 0:
        x_goal = p_circ[:, :N]
    else:
        x_goal = p_circ[:, N:]

    # SI position controller
    dxi = position_int(xi, x_goal)

    # Update robot marker positions
    for q in range(N):
        g[q].set_xdata([x[0, q]])
        g[q].set_ydata([x[1, q]])

    # Convert to unicycle and stop robots already at goal
    dxu = si_to_uni_dyn(dxi, x)
    for j in range(N):
        if np.linalg.norm(x_goal[:, j] - xi[:, j], 1) < 0.05:
            dxu[:, j] = 0.0

    dxu = uni_barrier_certificate(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Swap detection: if all robots close enough, flip flag
    if np.linalg.norm(x_goal - xi, 1) < 0.07:
        flag  = 1 - flag
        count += 1

# =========================================================
# COMPLETION CAPTION
# =========================================================
font_size = determine_font_size(r, 0.07)
ax.text(0, 0, 'All robots safely reached \n their destination!',
        fontsize=font_size, color='k', fontweight='bold',
        ha='center', va='center', zorder=10)
r._fig.canvas.draw_idle()

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()
