import numpy as np

def create_single_integrator_to_unicycle2(linear_velocity_gain=1, angular_velocity_limit=np.pi):
    """
    A mapping from
    """

    def f(dxi, poses):
        M,N = np.shape(dxi)

        a = np.cos(poses[2, :])
        b = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = linear_velocity_gain*(a*dxi[0, :] + b*dxi[1, :])
        dxu[1, :] = (np.pi/2)*angular_velocity_limit*np.arctan2(-b*dxi[0, :] + a*dxi[1, :], dxu[0, :])

        return dxu

    return f

def create_single_integrator_to_unicycle(projection_distance=0.05):
    """
    A
    """

    def f1(dxi, poses):
        M,N = np.shape(dxi)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = (cs*dxi[0, :] + ss*dxi[1, :])
        dxu[1, :] = (1/projection_distance)*(-ss*dxi[0, :] + cs*dxi[1, :])

        return dxu

    def f2(poses):
        _,N = np.shape(poses)

        si_states = np.zeros((2, N))
        si_states[0, :] = poses[0, :] + projection_distance*np.cos(poses[2, :])
        si_states[1, :] = poses[1, :] + projection_distance*np.sin(poses[2, :])

        return si_states

    return f1, f2
