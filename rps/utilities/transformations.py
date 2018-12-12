import numpy as np

def single_integrator_to_unicycle2(dxi, poses, linear_velocity_gain=1, angular_velocity_limit=4*np.pi):
    """A mapping from single-integrator to unicycle dynamics.

    dxi: 2xN numpy array with single-integrator control inputs
    poses: 2xN numpy array with single-integrator poses
    linear_velocity_gain: Gain for unicycle linear velocity
    angular_velocity_limit: Limit for angular velocity (i.e., |w| < angular_velocity_limit)

    -> 2xN numpy array of unicycle control inputs
    """

    M,N = np.shape(dxi)

    a = np.cos(poses[2, :])
    b = np.sin(poses[2, :])

    dxu = np.zeros((2, N))
    dxu[0, :] = linear_velocity_gain*(a*dxi[0, :] + b*dxi[1, :])
    dxu[1, :] = (np.pi/2)*angular_velocity_limit*np.arctan2(-b*dxi[0, :] + a*dxi[1, :], dxu[0, :])

    return dxu

def create_single_integrator_to_unicycle(projection_distance=0.05):
    """Creates a mapping from single integrator to unicycle dynamics by placing
    a virtual "point" in front of the unicycle.

    projection_distance: How far ahead to place the point

    -> (function, function)
    """

    def f1(dxi, poses):
        """Takes single-integrator velocities and transforms them to unicycle
        control inputs.

        dxi: 2xN numpy array of single-integrator control inputs
        poses: 3xN numpy array of unicycle poses

        -> 2xN numpy array of unicycle control inputs
        """

        M,N = np.shape(dxi)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = (cs*dxi[0, :] + ss*dxi[1, :])
        dxu[1, :] = (1/projection_distance)*(-ss*dxi[0, :] + cs*dxi[1, :])

        return dxu

    def f2(poses):
        """Takes unicycle states and returns single-integrator states

        poses: 3xN numpy array of unicycle states

        -> 2xN numpy array of single-integrator states
        """

        _,N = np.shape(poses)

        si_states = np.zeros((2, N))
        si_states[0, :] = poses[0, :] + projection_distance*np.cos(poses[2, :])
        si_states[1, :] = poses[1, :] + projection_distance*np.sin(poses[2, :])

        return si_states

    return f1, f2

def unicycle_to_single_integrator(dxu, poses, projection_distance=0.05):
    """A function for converting from unicycle to single-integrator dynamics.
    Utilizes a virtual point placed in front of the unicycle.

    dxu: 2xN numpy array of unicycle control inputs
    poses: 3xN numpy array of unicycle poses
    projection_distance: How far ahead of the unicycle model to place the point

    -> 2xN numpy array of single-integrator control inputs
    """

    M,N = np.shape(dxu)

    cs = np.cos(poses[2, :])
    ss = np.sin(poses[2, :])

    dxi = np.zeros((2, N))
    dxi[0, :] = (cs*dxu[0, :] - projection_distance*ss*dxu[1, :])
    dxi[1, :] = (ss*dxu[0, :] + projection_distance*cs*dxu[1, :])

    return dxi
