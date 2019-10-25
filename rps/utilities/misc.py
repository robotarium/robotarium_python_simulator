import numpy as np
import matplotlib.pyplot as plt


def generate_initial_conditions(N, spacing=0.3, width=3, height=1.8):
    """Generates random initial conditions in an area of the specified
    width and height at the required spacing.

    N: int (number of agents)
    spacing: double (how far apart positions can be)
    width: double (width of area)
    height: double (height of area)

    -> 3xN numpy array (of poses)
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

    x_range = int(np.floor(width/spacing))
    y_range = int(np.floor(height/spacing))

    assert x_range != 0, "In the function generate_initial_conditions, the space between robots (space) is too large compared to the width of the area robots are randomly initialized in (width)."
    assert y_range != 0, "In the function generate_initial_conditions, the space between robots (space) is too large compared to the height of the area robots are randomly initialized in (height)."
    assert x_range*y_range > N, "In the function generate_initial_conditions, it is impossible to place %r robots within a %r x %r meter area with a spacing of %r meters." % (N, width, height, spacing)

    choices = (np.random.choice(x_range*y_range, N, replace=False)+1)

    poses = np.zeros((3, N))

    for i, c in enumerate(choices):
        x,y = divmod(c, y_range)
        poses[0, i] = x*spacing - width/2
        poses[1, i] = y*spacing - height/2
        poses[2, i] = np.random.rand()*2*np.pi - np.pi

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
