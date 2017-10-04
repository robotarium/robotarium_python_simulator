import numpy as np

def single_integrator_position_controller(x, poses, gain = 1, magnitude_limit = 0.08):

    # Calculate control input
    dxi = gain*(poses-x)

    # Threshold magnitude
    norms = np.linalg.norm(dxi, axis=0)
    idxs = np.where(norms > magnitude_limit)
    dxi[:, idxs] = dxi[:, idxs] / norms[idxs]

    return dxi

def unicycle_pose_controller(x, poses, magnitude_limit=0.08, position_error=0.01, rotation_error = 0.25):
    """
    A pose controller for unicycle models.
    """

    _,N = np.shape(x)
    dxu = zeros(2, N)

    # Get the norms
    norms = np.linalg.norm(poses[2, :] - states[2, :], axis=0)

    # Figure out who's close enough
    not_there = np.which(norms > position_error)
    there = np.which(norms <= position_error)

    # Calculate angle proportional controller
    wrapped_angles = poses[2, there] - states[2, there]
    wrapped_angles = atan2(np.sin(wrapped_angles), np.cos(wrapped_angles))

    # Get a proportional controller for position
    dxi = single_integrator_position_controller(x[:2, :], poses[:2, :], magnitude_limit=magnitude_limit)

    # Decide what to do based on how close we are
    dxu[:, not_there] = single_integrator_to_unicycle2(dxi, x)
    dxu[:, there] = np.vstack([np.zeros(N), wrapped_angles])

    return dxu
