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
        assert xi.shape[1] == positions.shape[1], "In the si_position_controller function created by the create_si_position_controller function, the number of single-integrator robot states (xi) must be equal to the number of robot goal points (positions). Recieved a single integrator current position input array of size %r x %r and desired position array of size %r x %r." % (xi.shape[0], xi.shape[1], positions.shape[0], positions.shape[1])

        _,N = np.shape(xi)
        dxi = np.zeros((2, N))

        # Calculate control input
        dxi[0][:] = x_velocity_gain*(positions[0][:]-xi[0][:])
        dxi[1][:] = y_velocity_gain*(positions[1][:]-xi[1][:])

        # Threshold magnitude
        norms = np.linalg.norm(dxi, axis=0)
        idxs = np.where(norms > velocity_magnitude_limit)
        if norms[idxs].size != 0:
            dxi[:, idxs] *= velocity_magnitude_limit/norms[idxs]

        return dxi

    return si_position_controller

def create_clf_unicycle_position_controller(linear_velocity_gain=0.8, angular_velocity_gain=3):
    """Creates a unicycle model pose controller.  Drives the unicycle model to a given position
    and orientation. (($u: \mathbf{R}^{3 \times N} \times \mathbf{R}^{2 \times N} \to \mathbf{R}^{2 \times N}$)

    linear_velocity_gain - the gain impacting the produced unicycle linear velocity
    angular_velocity_gain - the gain impacting the produced unicycle angular velocity
    
    -> function
    """

    #Check user input types
    assert isinstance(linear_velocity_gain, (int, float)), "In the function create_clf_unicycle_position_controller, the linear velocity gain (linear_velocity_gain) must be an integer or float. Recieved type %r." % type(linear_velocity_gain).__name__
    assert isinstance(angular_velocity_gain, (int, float)), "In the function create_clf_unicycle_position_controller, the angular velocity gain (angular_velocity_gain) must be an integer or float. Recieved type %r." % type(angular_velocity_gain).__name__
    
    #Check user input ranges/sizes
    assert linear_velocity_gain >= 0, "In the function create_clf_unicycle_position_controller, the linear velocity gain (linear_velocity_gain) must be greater than or equal to zero. Recieved %r." % linear_velocity_gain
    assert angular_velocity_gain >= 0, "In the function create_clf_unicycle_position_controller, the angular velocity gain (angular_velocity_gain) must be greater than or equal to zero. Recieved %r." % angular_velocity_gain
     

    def position_uni_clf_controller(x, positions):

        """  A position controller for unicycle models.  This utilized a control lyapunov function
        (CLF) to drive a unicycle system to a desired position. This function operates on unicycle
        states and desired positions to return a unicycle velocity command vector.

        x: 3xN numpy array (of unicycle states, [x;y;theta])
        poses: 3xN numpy array (of desired positons, [x_goal;y_goal])

        -> 2xN numpy array (of unicycle control inputs)
        """

        #Check user input types
        assert isinstance(x, np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the single-integrator robot states (xi) must be a numpy array. Recieved type %r." % type(x).__name__
        assert isinstance(positions, np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the robot goal points (positions) must be a numpy array. Recieved type %r." % type(positions).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert positions.shape[0] == 2, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the robot goal positions (positions) must be 2 ([x_goal;y_goal]). Recieved dimension %r." % positions.shape[0]
        assert x.shape[1] == positions.shape[1], "In the function created by the create_clf_unicycle_position_controller function, the number of unicycle robot states (x) must be equal to the number of robot goal positions (positions). Recieved a current robot pose input array (x) of size %r x %r and desired position array (positions) of size %r x %r." % (x.shape[0], x.shape[1], positions.shape[0], positions.shape[1])


        _,N = np.shape(x)
        dxu = np.zeros((2, N))

        pos_error = positions - x[:2][:]
        rot_error = np.arctan2(pos_error[1][:],pos_error[0][:])
        dist = np.linalg.norm(pos_error, axis=0)

        dxu[0][:]=linear_velocity_gain*dist*np.cos(rot_error-x[2][:])
        dxu[1][:]=angular_velocity_gain*dist*np.sin(rot_error-x[2][:])

        return dxu

    return position_uni_clf_controller
