import numpy as np
import matplotlib.pyplot as plt


def generate_initial_conditions(N, spacing=0.3, width=3, height=1.8):
    """
    Generate N random 2D poses (x, y, theta) within a rectangular area,
    with a minimum spacing between any two points.

    Parameters:
        N (int): Number of poses to generate.
        spacing (float): Minimum distance between poses (default: 0.3).
        width (float): Width of the rectangle (default: 3.0).
        height (float): Height of the rectangle (default: 1.8).

    Returns:
        poses (np.ndarray): 3 x N array of poses [x; y; theta].
    """

    #Check user input types
    assert isinstance(N, int), "In the function generate_initial_conditions, the number of robots (N) to generate intial conditions for must be an integer. Recieved type %r." % type(N).__name__
    assert isinstance(spacing, (float,int)), "In the function generate_initial_conditions, the minimum spacing between robots (spacing) must be an integer or float. Recieved type %r." % type(spacing).__name__
    assert isinstance(width, (float,int)), "In the function generate_initial_conditions, the width of the area to place robots randomly (width) must be an integer or float. Recieved type %r." % type(width).__name__
    assert isinstance(height, (float,int)), "In the function generate_initial_conditions, the height of the area to place robots randomly (width) must be an integer or float. Recieved type %r." % type(height).__name__

    #Check user input ranges/sizes
    assert N > 0, "In the function generate_initial_conditions, the number of robots to generate initial conditions for (N) must be positive. Recieved %r." % N
    assert spacing > 0, "In the function generate_initial_conditions, the spacing between robots (spacing) must be positive. Recieved %r." % spacing
    assert width > 0, "In the function generate_initial_conditions, the width of the area to initialize robots randomly (width) must be positive. Recieved %r." % width
    assert height >0, "In the function generate_initial_conditions, the height of the area to initialize robots randomly (height) must be positive. Recieved %r." % height

    # Feasibility check (conservative)
    approx_max_points = int((width * height) / (spacing ** 2))
    if N > approx_max_points:
        raise ValueError(f"In the function generate_initial_conditions, "
                         f"Cannot fit {N} points with spacing {spacing} "
                         f"in a {width}m x {height}m area. "
                         f"Max possible (approx): {approx_max_points}.")

    points = []
    max_attempts = 10000
    attempts = 0

    while len(points) < N and attempts < max_attempts:
        candidate = [(np.random.rand() - 0.5) * width,
                     (np.random.rand() - 0.5) * height]

        if not points:
            points.append(candidate)
        else:
            dists = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.all(dists >= spacing):
                points.append(candidate)

        attempts += 1

    if len(points) < N:
        raise RuntimeError("Could not generate enough points with the given spacing. "
                           "Try reducing N or spacing.")

    poses = np.zeros((3, N))
    poses[0:2, :] = np.array(points).T
    poses[2, :] = np.random.uniform(-np.pi, np.pi, N)

    return poses

def at_pose(states, poses, position_error=0.05, rotation_error=0.2):
    """Checks whether robots are "close enough" to poses

    states: 3xN numpy array (of unicycle states)
    poses: 3xN numpy array (of desired states)

    -> 1xN numpy index array (of agents that are close enough)
    """
    #Check user input types
    assert isinstance(states, np.ndarray), "In the at_pose function, the robot current state argument (states) must be a numpy ndarray. Recieved type %r." % type(states).__name__
    assert isinstance(poses, np.ndarray), "In the at_pose function, the checked pose argument (poses) must be a numpy ndarray. Recieved type %r." % type(poses).__name__
    assert isinstance(position_error, (float,int)), "In the at_pose function, the allowable position error argument (position_error) must be an integer or float. Recieved type %r." % type(position_error).__name__
    assert isinstance(rotation_error, (float,int)), "In the at_pose function, the allowable angular error argument (rotation_error) must be an integer or float. Recieved type %r." % type(rotation_error).__name__

    #Check user input ranges/sizes
    assert states.shape[0] == 3, "In the at_pose function, the dimension of the state of each robot must be 3 ([x;y;theta]). Recieved %r." % states.shape[0]
    assert poses.shape[0] == 3, "In the at_pose function, the dimension of the checked pose of each robot must be 3 ([x;y;theta]). Recieved %r." % poses.shape[0]
    assert states.shape == poses.shape, "In the at_pose function, the robot current state and checked pose inputs must be the same size (3xN, where N is the number of robots being checked). Recieved a state array of size %r x %r and checked pose array of size %r x %r." % (states.shape[0], states.shape[1], poses.shape[0], poses.shape[1])

    # Calculate rotation errors with angle wrapping
    res = states[2, :] - poses[2, :]
    res = np.abs(np.arctan2(np.sin(res), np.cos(res)))

    # Calculate position errors
    pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)

    # Determine which agents are done
    done = np.nonzero((res <= rotation_error) & (pes <= position_error))

    return done

def at_position(states, points, position_error=0.02):
    """Checks whether robots are "close enough" to desired position

    states: 3xN numpy array (of unicycle states)
    points: 2xN numpy array (of desired points)

    -> 1xN numpy index array (of agents that are close enough)
    """

    #Check user input types
    assert isinstance(states, np.ndarray), "In the at_position function, the robot current state argument (states) must be a numpy ndarray. Recieved type %r." % type(states).__name__
    assert isinstance(points, np.ndarray), "In the at_position function, the desired pose argument (poses) must be a numpy ndarray. Recieved type %r." % type(points).__name__
    assert isinstance(position_error, (float,int)), "In the at_position function, the allowable position error argument (position_error) must be an integer or float. Recieved type %r." % type(position_error).__name__
    
    #Check user input ranges/sizes
    assert states.shape[0] == 3, "In the at_position function, the dimension of the state of each robot (states) must be 3. Recieved %r." % states.shape[0]
    assert points.shape[0] == 2, "In the at_position function, the dimension of the checked position for each robot (points) must be 2. Recieved %r." % points.shape[0]
    assert states.shape[1] == poses.shape[1], "In the at_position function, the number of checked points (points) must match the number of robot states provided (states). Recieved a state array of size %r x %r and desired pose array of size %r x %r." % (states.shape[0], states.shape[1], points.shape[0], points.shape[1])

    # Calculate position errors
    pes = np.linalg.norm(states[:2, :] - points, 2, 0)

    # Determine which agents are done
    done = np.nonzero((pes <= position_error))

    return done

def determine_marker_size(robotarium_instance, marker_size_meters):

	# Get the x and y dimension of the robotarium figure window in pixels
	fig_dim_pixels = robotarium_instance.axes.transData.transform(np.array([[robotarium_instance.boundaries[2]],[robotarium_instance.boundaries[3]]]))

	# Determine the ratio of the robot size to the x-axis (the axis are
	# normalized so you could do this with y and figure height as well).
	marker_ratio = (marker_size_meters)/(robotarium_instance.boundaries[2])

	# Determine the marker size in points so it fits the window. Note: This is squared
	# as marker sizes are areas.
	return (fig_dim_pixels[0,0] * marker_ratio)**2.


def determine_font_size(robotarium_instance, font_height_meters):

	# Get the x and y dimension of the robotarium figure window in pixels
	y1, y2 = robotarium_instance.axes.get_window_extent().get_points()[:,1]

	# Determine the ratio of the robot size to the y-axis.
	font_ratio = (y2-y1)/(robotarium_instance.boundaries[2])

	# Determine the font size in points so it fits the window.
	return (font_ratio*font_height_meters)
