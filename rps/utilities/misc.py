import numpy as np
from numpy.typing import NDArray
from datetime import datetime
import random
from rps.robotarium_abc import ARobotarium

def determine_font_size(robotarium_instance: ARobotarium, font_height_meters: float) -> float:
    """Font size in points matching a physical height in metres."""
    fig   = robotarium_instance._fig
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    arena_h  = robotarium_instance.BOUNDARIES[3] - robotarium_instance.BOUNDARIES[2]
    return fig_h_px * (font_height_meters / arena_h)


def generate_initial_poses(
    N: int,
    width: float = 3.0,
    height: float = 1.0
) -> NDArray[np.floating]:
    """Generate N initial poses for the robots starting from where they come off of the chargers"""
    xs = np.linspace(-width/2, width/2, 11, endpoint=True)
    poses = [[x, y, np.pi/2 if y < 0 else -np.pi/2] for x in xs for y in [-height/2, height/2]]
    return np.array(random.sample(poses, N)).T


def generate_random_poses(N, spacing=0.5, width=3.0, height=1.8):
    """
    Generate N random, well-spaced poses inside a rectangle.
    
    Returns a 3xN matrix of poses [x; y; theta] whose (x,y) positions are
    uniformly sampled inside a Width x Height rectangle centred at the origin,
    subject to a minimum separation distance between every pair of points.
    Headings theta are drawn uniformly from [-pi, pi).
    """
    assert isinstance(N, int) and N > 0, "N must be a positive integer."
    assert spacing > 0, "spacing must be positive."
    assert width > 0 and height > 0, "width and height must be positive."

    # Feasibility check
    approx_max = int(np.floor((width * height) / (spacing ** 2)))
    assert N <= approx_max, f"Cannot fit {N} poses with spacing {spacing:.2f} m into a {width:.2f} x {height:.2f} m area. Approximate maximum: {approx_max}."

    accepted = []
    max_attempts = 10000

    # Rejection sampling
    for _ in range(max_attempts):
        if len(accepted) == N:
            break

        candidate = np.array([(np.random.rand() - 0.5) * width,
                              (np.random.rand() - 0.5) * height])

        if not accepted or np.all(np.linalg.norm(np.array(accepted) - candidate, axis=1) >= spacing):
            accepted.append(candidate)

    assert len(accepted) == N, f"Could not place {N} poses with spacing {spacing:.2f} m after {max_attempts} attempts. Try reducing N or Spacing."

    poses = np.zeros((3, N))
    poses[:2, :] = np.array(accepted).T
    poses[2, :] = np.random.rand(N) * 2 * np.pi - np.pi

    return poses

def generate_random_positions(N, spacing=0.5, width=3.0, height=1.8):
    """
    Generate N random, well-spaced 2D positions inside a rectangle.
 
    Returns a 2xN matrix of positions [x; y] uniformly sampled inside a
    Width x Height rectangle centred at the origin, subject to a minimum
    separation distance between every pair of points.  Unlike
    generate_random_poses, no heading column is added.
 
    Parameters
    ----------
    N       : int   -- number of positions
    spacing : float -- minimum centre-to-centre distance (m), default 0.5
    width   : float -- rectangle width  (m), default 3.0
    height  : float -- rectangle height (m), default 1.8
    """
    assert isinstance(N, int) and N > 0, "N must be a positive integer."
    assert spacing > 0, "spacing must be positive."
    assert width > 0 and height > 0, "width and height must be positive."
 
    approx_max = int(np.floor((width * height) / (spacing ** 2)))
    assert N <= approx_max, (
        f"Cannot fit {N} positions with spacing {spacing:.2f} m into a "
        f"{width:.2f} x {height:.2f} m area. Approximate maximum: {approx_max}."
    )
 
    accepted = []
    max_attempts = 10000
 
    for _ in range(max_attempts):
        if len(accepted) == N:
            break
        candidate = np.array([(np.random.rand() - 0.5) * width,
                               (np.random.rand() - 0.5) * height])
        if not accepted or np.all(
            np.linalg.norm(np.array(accepted) - candidate, axis=1) >= spacing
        ):
            accepted.append(candidate)
 
    assert len(accepted) == N, (
        f"Could not place {N} positions with spacing {spacing:.2f} m after "
        f"{max_attempts} attempts. Try reducing N or spacing."
    )
 
    positions = np.zeros((2, N))
    positions[:, :] = np.array(accepted).T
    return positions


def create_at_pose(position_error=0.05, rotation_error=0.2):
    """
    Creates a function to check whether unicycle robots have reached their 
    desired poses (position + orientation).
    """
    def check_at_pose(states, poses):
        N = states.shape[1]
        assert states.shape[0] == 3, f"states must be 3xN. Got {states.shape[0]}x{states.shape[1]}."
        assert poses.shape[0] == 3, f"poses must be 3xN. Got {poses.shape[0]}x{poses.shape[1]}."
        assert poses.shape[1] == N, f"states and poses must have the same number of columns ({N} vs {poses.shape[1]})."

        pos_converged = np.linalg.norm(states[:2, :] - poses[:2, :], axis=0) <= position_error
        
        heading_err = states[2, :] - poses[2, :]
        rot_converged = np.abs(np.arctan2(np.sin(heading_err), np.cos(heading_err))) <= rotation_error
        
        idxs = pos_converged & rot_converged
        done = bool(np.all(idxs))
        
        return done, idxs

    return check_at_pose


def create_at_position(position_error=0.05):
    """
    Creates a function to check whether robots have reached their desired 
    positions (no heading requirement).
    """
    def check_at_position(states, positions):
        N = states.shape[1]
        assert states.shape[0] >= 2, f"states must be at least 2xN. Got {states.shape[0]}x{states.shape[1]}."
        assert positions.shape[0] == 2, f"positions must be 2xN. Got {positions.shape[0]}x{positions.shape[1]}."
        assert positions.shape[1] == N, f"states and positions must have the same number of columns ({N} vs {positions.shape[1]})."

        idxs = np.linalg.norm(states[:2, :] - positions[:2, :], axis=0) <= position_error
        done = bool(np.all(idxs))
        
        return done, idxs

    return check_at_position


def rotation_matrix(theta):
    """
    Returns a 2D rotation matrix for each angle in theta.
    
    Returns a 3x3xN array of rotation matrices to match the MATLAB implementation.
    """
    theta = np.atleast_1d(theta).flatten()
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.zeros((3, 3, len(theta)))
    
    R[0, 0, :] = c
    R[0, 1, :] = -s
    R[1, 0, :] = s
    R[1, 1, :] = c
    R[2, 2, :] = 1.0

    return R


def unique_filename(file_name):
    """
    Append a timestamp to a filename and add a .npy extension.
    """
    t = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return f"{file_name}_{t}.npy"