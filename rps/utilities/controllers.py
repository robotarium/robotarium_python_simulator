import numpy as np
from rps.utilities.transformations import *

def single_integrator_position_controller(x, poses, gain=1, magnitude_limit=0.08):
    """A position controller for single integrators.  Drives a single integrator to a point
    using a propoertional controller.

    x: 2xN numpy array (of single-integrator states)
    poses: 2xN numpy array (of desired poses)
    gain: double (proportional gain for controller)
    magnitude_limit: double (limit of the magnitude of the single-integrator control input
    produced)

    -> 2xN numpy array (of single-integrator control inputs)
    """

    # Calculate control input
    dxi = gain*(poses-x)

    # Threshold magnitude
    norms = np.linalg.norm(dxi, axis=0)
    idxs = np.where(norms > magnitude_limit)
    dxi[:, idxs] *= magnitude_limit/norms[idxs]

    return dxi

def unicycle_pose_controller(x, poses, magnitude_limit=0.08, position_error=0.01, rotation_error=0.25):
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
