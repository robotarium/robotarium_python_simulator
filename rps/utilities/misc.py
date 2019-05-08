import numpy as np


def generate_initial_conditions(N, spacing=0.1, width=3, height=3):
    """Generates random initial conditions in an area of the specified
    width and height at the required spacing.

    N: int (number of agents)
    spacing: double (how far apart positions can be)
    width: double (width of area)
    height: double (height of area)

    -> 3xN numpy array (of poses)
    """

    x_range = int(np.floor(width/spacing))
    y_range = int(np.floor(height/spacing))

    choices = (np.random.choice(x_range*y_range, N, replace=False)+1)

    poses = np.zeros((3, N))

    for i, c in enumerate(choices):
        x,y = divmod(c, y_range)
        poses[0, i] = x*spacing - width/2
        poses[1, i] = y*spacing - height/2
        poses[2, i] = np.random.rand()*2*np.pi - np.pi

    return poses

def at_pose(states, poses, position_error=0.02, rotation_error=0.5):
    """Checks whether robots are "close enough" to poses

    states: 3xN numpy array (of unicycle states)
    poses: 3xN numpy array (of desired states)

    -> 1xN numpy index array (of agents that are close enough)
    """

    # Calculate rotation errors with angle wrapping
    res = states[2, :] - poses[2, :]
    res = np.abs(np.arctan2(np.sin(res), np.cos(res)))

    # Calculate position errors
    pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)

    # Determine which agents are done
    done = np.nonzero((res <= rotation_error) & (pes <= position_error))

    return done

def at_position(states, poses, position_error=0.02):
    """Checks whether robots are "close enough" to desired position

    states: 3xN numpy array (of unicycle states)
    poses: 3xN numpy array (of desired states)

    -> 1xN numpy index array (of agents that are close enough)
    """
    # Calculate position errors
    pes = np.linalg.norm(states[:2, :] - poses[:2, :], 2, 0)

    # Determine which agents are done
    done = np.nonzero((pes <= position_error))

    return done
