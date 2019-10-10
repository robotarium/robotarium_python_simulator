import numpy as np
from rps.utilities.transformations import *

def create_si_position_controller(x_velocity_gain=1, y_velocity_gain=1, velocity_magnitude_limit=0.15):
    """Creates a position controller for single integrators.  Drives a single integrator to a point
    using a propoertional controller.

    x_velocity_gain - the gain impacting the x (horizontal) velocity of the single integrator
    y_velocity_gain - the gain impacting the y (vertical) velocity of the single integrator
    velocity_magnitude_limit - the maximum magnitude of the produce velocity vector (should be less than the max linear speed of the platform)

    -> function
    """

    #Check user input types
    assert isinstance(x_velocity_gain, (int, float)), "In the function create_si_position_controller, the x linear velocity gain (x_velocity_gain) must be an integer or float. Recieved type %r." % type(x_velocity_gain).__name__
    assert isinstance(y_velocity_gain, (int, float)), "In the function create_si_position_controller, the y linear velocity gain (y_velocity_gain) must be an integer or float. Recieved type %r." % type(y_velocity_gain).__name__
    assert isinstance(velocity_magnitude_limit, (int, float)), "In the function create_si_position_controller, the velocity magnitude limit (y_velocity_gain) must be an integer or float. Recieved type %r." % type(y_velocity_gain).__name__
    
    #Check user input ranges/sizes
    assert x_velocity_gain > 0, "In the function create_si_position_controller, the x linear velocity gain (x_velocity_gain) must be positive. Recieved %r." % x_velocity_gain
    assert y_velocity_gain > 0, "In the function create_si_position_controller, the y linear velocity gain (y_velocity_gain) must be positive. Recieved %r." % y_velocity_gain
    assert velocity_magnitude_limit >= 0, "In the function create_si_position_controller, the velocity magnitude limit (velocity_magnitude_limit) must not be negative. Recieved %r." % velocity_magnitude_limit
    
    gain = np.diag([x_velocity_gain, y_velocity_gain])

    def si_position_controller(xi, positions):

        """
        x: 2xN numpy array (of single-integrator states of the robots)
        points: 2xN numpy array (of desired points each robot should achieve)

        -> 2xN numpy array (of single-integrator control inputs)

        """

        #Check user input types
        assert isinstance(xi, np.ndarray), "In the si_position_controller function created by the create_si_position_controller function, the single-integrator robot states (xi) must be a numpy array. Recieved type %r." % type(xi).__name__
        assert isinstance(positions, np.ndarray), "In the si_position_controller function created by the create_si_position_controller function, the robot goal points (positions) must be a numpy array. Recieved type %r." % type(positions).__name__

        #Check user input ranges/sizes
        assert xi.shape[0] == 2, "In the si_position_controller function created by the create_si_position_controller function, the dimension of the single-integrator robot states (xi) must be 2 ([x;y]). Recieved dimension %r." % xi.shape[0]
        assert positions.shape[0] == 2, "In the si_position_controller function created by the create_si_position_controller function, the dimension of the robot goal points (positions) must be 2 ([x_goal;y_goal]). Recieved dimension %r." % positions.shape[0]
        assert xi.shape[1] == positions.shape[1], "In the si_position_controller function created by the create_si_position_controller function, the number of single-integrator robot states (xi) must be equal to the number of robot goal points (positions). Recieved a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (xi.shape[0], xi.shape[1], positions.shape[0], positions.shape[1])


        # Calculate control input
        dxi = gain*(positions-xi)

        # Threshold magnitude
        norms = np.linalg.norm(dxi, axis=0)
        idxs = np.where(norms > magnitude_limit)
        if idxs.size != 0:
            dxi[:, idxs] *= magnitude_limit/norms[idxs]

        return dxi

    return si_position_controller

def unicycle_pose_controller(x, poses, magnitude_limit=0.25, position_error=0.01, rotation_error=0.25):
    """  A pose controller for unicycle models.  This is a hybrid controller that first
    drives the unicycle to a point then turns the unicycle to match the orientation.

    x: 3xN numpy array (of unicycle states)
    poses: 3xN numpy array (of desired poses)
    magnitude_limit: int (norm limit for produced inputs (affects going to point))
    position_error: double (how close the unicycle will come to the point)
    rotation_error: double (how close the unicycle)

    -> 2xN numpy array (of unicycle control inputs)
    """

    _,N = np.shape(x)
    dxu = np.zeros((2, N))

    # Get the norms
    norms = np.linalg.norm(poses[:2, :] - x[:2, :], axis=0)

    # Figure out who's close enough
    not_there = np.where(norms > position_error)[0]
    there = np.where(norms <= position_error)[0]

    # Calculate angle proportional controller
    wrapped_angles = poses[2, there] - x[2, there]
    wrapped_angles = np.arctan2(np.sin(wrapped_angles), np.cos(wrapped_angles))

    # Get a proportional controller for position
    dxi = single_integrator_position_controller(x[:2, :], poses[:2, :], magnitude_limit=magnitude_limit)

    # Decide what to do based on how close we are
    dxu[:, not_there] = single_integrator_to_unicycle2(dxi[:, not_there], x[:, not_there])
    dxu[:, there] = np.vstack([np.zeros(np.size(there)), wrapped_angles])

    return dxu
