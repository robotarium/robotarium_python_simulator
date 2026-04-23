# LEADER_FOLLOWER_WITH_PLOTTING
# One leader robot travels between four waypoints while N-1 follower robots
# maintain a formation relative to it using a graph Laplacian-based law.
#
# Visualization includes:
#   - Graph edges (blue for follower-follower, red for leader-follower)
#   - Robot identification labels
#   - Leader waypoint goal markers
#
# Sean Wilson / Python port
# 07/2019

import numpy as np

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.graph import completeGL, topological_neighbors
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_poses

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N          = 4       # 1 leader + N-1 followers
iterations = 5000

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_positions = generate_random_poses(N, width=1.6, height=1.0, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                           initial_conditions=initial_positions)

# =========================================================
# GRAPH LAPLACIAN AND FORMATION SETUP
# =========================================================
# Followers form a complete graph among themselves; robot 2 (index 1) is
# additionally connected to the leader (robot 1, index 0)
followers = -completeGL(N - 1)
L = np.zeros((N, N))
L[1:N, 1:N] = followers
L[1, 1] += 1
L[1, 0]  = -1

formation_control_gain = 5      # matches MATLAB
desired_distance        = 0.3   # matches MATLAB

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
si_to_uni_dyn    = create_si_to_uni_dynamics(linear_velocity_gain=0.8)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()
leader_controller = create_si_position_controller(
    x_velocity_gain=0.8, y_velocity_gain=0.8, velocity_magnitude_limit=0.08)

# =========================================================
# WAYPOINT INITIALIZATION
# =========================================================
waypoints   = np.array([[-1, -1, 1, 1],
                         [0.8, -0.8, -0.8, 0.8]])   # (2, 4)
close_enough = 0.03
state        = 0
dxi          = np.zeros((2, N))

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

# One colour per robot
CM = ['k', 'b', 'r', 'g']

marker_size_goal = determine_marker_size(r, 0.20)
font_size        = determine_font_size(r, 0.05)
line_width       = 5

# Waypoint goal markers and labels
g            = []
goal_labels  = []
for i in range(waypoints.shape[1]):
    gm, = ax.plot(waypoints[0, i], waypoints[1, i], 's',
                  markersize=marker_size_goal, 
                  markerfacecolor='none',
                  markeredgecolor=CM[i],
                  markeredgewidth=line_width,
                  zorder=2)
    g.append(gm)
    gl = ax.text(waypoints[0, i] - 0.05, waypoints[1, i],
                 f'G{i+1}', fontsize=font_size, fontweight='bold',
                 color='k', zorder=2)
    goal_labels.append(gl)

# Get initial poses before drawing edges
x = r.get_poses()

# Follower-follower edges (blue): L == 1 gives the directed edge source
rows, cols = np.where(L == 1)
# Each undirected edge appears once as a directed entry with value 1
lf = []
for k in range(len(rows)):
    line, = ax.plot([x[0, rows[k]], x[0, cols[k]]],
                    [x[1, rows[k]], x[1, cols[k]]],
                    color='b', linewidth=line_width, zorder=1)
    lf.append(line)

# Leader-follower edge (red): always between robot 0 and robot 1
ll, = ax.plot([x[0, 0], x[0, 1]], [x[1, 0], x[1, 1]],
              color='r', linewidth=line_width, zorder=1)

# Follower labels (off-screen initially)
follower_labels = []
for j in range(N - 1):
    fl = ax.text(500, 500, f'Follower Robot {j+1}',
                 fontsize=font_size, fontweight='bold', color='k', zorder=5)
    follower_labels.append(fl)

# Leader label (off-screen initially)
leader_label = ax.text(500, 500, 'Leader Robot',
                       fontsize=font_size, fontweight='bold', color='r', zorder=5)

r.step()

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers and text when the figure window is resized."""
    new_goal_size = determine_marker_size(r, 0.20)
    font_size     = determine_font_size(r, 0.05)
    
    # Scale goal markers and their labels
    for gm in g:
        gm.set_markersize(new_goal_size)
    for gl in goal_labels:
        gl.set_fontsize(font_size)
        
    # Scale robot labels
    leader_label.set_fontsize(font_size)
    for fl in follower_labels:
        fl.set_fontsize(font_size)
        
    ax.figure.canvas.draw_idle()

ax.figure.canvas.mpl_connect('resize_event', on_resize)

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for t in range(iterations):

    x = r.get_poses()

    # ---------------------------------------------------------
    # Formation control for followers
    # ---------------------------------------------------------
    for i in range(1, N):
        dxi[:, i] = 0.0
        for j in topological_neighbors(L, i):
            dxi[:, i] += (
                formation_control_gain
                * (np.linalg.norm(x[:2, j] - x[:2, i])**2 - desired_distance**2)
                * (x[:2, j] - x[:2, i])
            )

    # ---------------------------------------------------------
    # Leader waypoint tracking
    # ---------------------------------------------------------
    waypoint = waypoints[:, state].reshape(2, 1)
    dxi[:, 0] = leader_controller(x[:2, 0:1], waypoint).flatten()
    if np.linalg.norm(x[:2, 0] - waypoints[:, state]) < close_enough:
        state = (state + 1) % waypoints.shape[1]

    # ---------------------------------------------------------
    # Velocity thresholding for followers (not leader)
    # ---------------------------------------------------------
    threshold = 0.75 * r.MAX_LINEAR_VELOCITY
    for k in range(1, N):
        spd = np.linalg.norm(dxi[:, k])
        if spd > threshold:
            dxi[:, k] *= threshold / spd

    # ---------------------------------------------------------
    # Convert to unicycle and apply barrier certificate
    # ---------------------------------------------------------
    dxu = si_to_uni_dyn(dxi, x)
    dxu = uni_barrier_cert(dxu, x)
    r.set_velocities(np.arange(N), dxu)

    # ---------------------------------------------------------
    # Update plot handles
    # ---------------------------------------------------------
    # Follower labels track their robots
    for q in range(N - 1):
        follower_labels[q].set_position((x[0, q + 1] - 0.15, x[1, q + 1] + 0.15))

    # Follower-follower graph edges
    for m in range(len(rows)):
        lf[m].set_xdata([x[0, rows[m]], x[0, cols[m]]])
        lf[m].set_ydata([x[1, rows[m]], x[1, cols[m]]])

    # Leader label and leader-follower edge
    leader_label.set_position((x[0, 0] - 0.15, x[1, 0] + 0.15))
    ll.set_xdata([x[0, 0], x[0, 1]])
    ll.set_ydata([x[1, 0], x[1, 1]])

    r.step()

# =========================================================
# DEBUG REPORT
# =========================================================
r.debug()