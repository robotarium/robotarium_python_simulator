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
        xi: 2xN numpy array (of single-integrator states of the robots)
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
     

    def position_uni_clf_controller(states, positions):

        """  A position controller for unicycle models.  This utilized a control lyapunov function
        (CLF) to drive a unicycle system to a desired position. This function operates on unicycle
        states and desired positions to return a unicycle velocity command vector.

        states: 3xN numpy array (of unicycle states, [x;y;theta])
        poses: 3xN numpy array (of desired positons, [x_goal;y_goal])

        -> 2xN numpy array (of unicycle control inputs)
        """

        #Check user input types
        assert isinstance(states, np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the single-integrator robot states (xi) must be a numpy array. Recieved type %r." % type(states).__name__
        assert isinstance(positions, np.ndarray), "In the function created by the create_clf_unicycle_position_controller function, the robot goal points (positions) must be a numpy array. Recieved type %r." % type(positions).__name__

        #Check user input ranges/sizes
        assert states.shape[0] == 3, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the unicycle robot states (states) must be 3 ([x;y;theta]). Recieved dimension %r." % states.shape[0]
        assert positions.shape[0] == 2, "In the function created by the create_clf_unicycle_position_controller function, the dimension of the robot goal positions (positions) must be 2 ([x_goal;y_goal]). Recieved dimension %r." % positions.shape[0]
        assert states.shape[1] == positions.shape[1], "In the function created by the create_clf_unicycle_position_controller function, the number of unicycle robot states (states) must be equal to the number of robot goal positions (positions). Recieved a current robot pose input array (states) of size %r states %r and desired position array (positions) of size %r states %r." % (states.shape[0], states.shape[1], positions.shape[0], positions.shape[1])


        _,N = np.shape(states)
        dxu = np.zeros((2, N))

        pos_error = positions - states[:2][:]
        rot_error = np.arctan2(pos_error[1][:],pos_error[0][:])
        dist = np.linalg.norm(pos_error, axis=0)

        dxu[0][:]=linear_velocity_gain*dist*np.cos(rot_error-states[2][:])
        dxu[1][:]=angular_velocity_gain*dist*np.sin(rot_error-states[2][:])

        return dxu

    return position_uni_clf_controller

def create_clf_unicycle_pose_controller(approach_angle_gain=1, desired_angle_gain=2.7, rotation_error_gain=1):
    """Returns a controller ($u: \mathbf{R}^{3 \times N} \times \mathbf{R}^{3 \times N} \to \mathbf{R}^{2 \times N}$) 
    that will drive a unicycle-modeled agent to a pose (i.e., position & orientation). This control is based on a control
    Lyapunov function.

    approach_angle_gain - affects how the unicycle approaches the desired position
    desired_angle_gain - affects how the unicycle approaches the desired angle
    rotation_error_gain - affects how quickly the unicycle corrects rotation errors.


    -> function
    """

    gamma = approach_angle_gain
    k = desired_angle_gain
    h = rotation_error_gain

    def R(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    def pose_uni_clf_controller(states, poses):

        N_states = states.shape[1]
        dxu = np.zeros((2,N_states))

        for i in range(N_states):
            translate = R(-poses[2,i]).dot((poses[:2,i]-states[:2,i]))
            e = np.linalg.norm(translate)
            theta = np.arctan2(translate[1],translate[0])
            alpha = theta - (states[2,i]-poses[2,i])
            alpha = np.arctan2(np.sin(alpha),np.cos(alpha))

            ca = np.cos(alpha)
            sa = np.sin(alpha)

            dxu[0,i] = gamma* e* ca
            dxu[1,i] = k*alpha + gamma*((ca*sa)/alpha)*(alpha + h*theta)

        return dxu

    return pose_uni_clf_controller

def create_hybrid_unicycle_pose_controller(linear_velocity_gain=1, angular_velocity_gain=2, velocity_magnitude_limit=0.15, angular_velocity_limit=np.pi, position_error=0.05, position_epsilon=0.03, rotation_error=0.05):
    '''Returns a controller ($u: \mathbf{R}^{3 \times N} \times \mathbf{R}^{3 \times N} \to \mathbf{R}^{2 \times N}$)
    that will drive a unicycle-modeled agent to a pose (i.e., position & orientation). This controller is
    based on a hybrid controller that will drive the robot in a straight line to the desired position then rotate
    to the desired position.
    
    linear_velocity_gain - affects how much the linear velocity is scaled based on the position error
    angular_velocity_gain - affects how much the angular velocity is scaled based on the heading error
    velocity_magnitude_limit - threshold for the max linear velocity that will be achieved by the robot
    angular_velocity_limit - threshold for the max rotational velocity that will be achieved by the robot
    position_error - the error tolerance for the final position of the robot
    position_epsilon - the amount of translational distance that is allowed by the rotation before correcting position again.
    rotation_error - the error tolerance for the final orientation of the robot

    '''

    si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=linear_velocity_gain, angular_velocity_limit=angular_velocity_limit)

    def pose_uni_hybrid_controller(states, poses, approach_state = np.empty([0,0])):
        N=states.shape[1]
        dxu = np.zeros((2,N))

        #This is essentially a hack since default arguments are evaluated only once we will mutate it with each call
        if approach_state.shape[1] != N: 
            approach_state = np.ones((1,N))[0]

        for i in range(N):
            wrapped = poses[2,i] - states[2,i]
            wrapped = np.arctan2(np.sin(wrapped),np.cos(wrapped))

            dxi = poses[:2,[i]] - states[:2,[i]]

            #Normalize Vector
            norm_ = np.linalg.norm(dxi)

            if(norm_ > (position_error - position_epsilon) and approach_state[i]):
                if(norm_ > velocity_magnitude_limit):
                    dxi = velocity_magnitude_limit*dxi/norm_
                dxu[:,[i]] = si_to_uni_dyn(dxi, states[:,[i]])
            elif(np.absolute(wrapped) > rotation_error):
                approach_state[i] = 0
                if(norm_ > position_error):
                    approach_state = 1
                dxu[0,i] = 0
                dxu[1,i] = angular_velocity_gain*wrapped
            else:
                dxu[:,[i]] = np.zeros((2,1))

        return dxu

    return pose_uni_hybrid_controller
