"""Miscellaneous utility functions"""

import jax
import jax.numpy as jnp
from functools import partial

def generate_initial_conditions(key, N, spacing=0.3, width=3.0, height=1.8) -> jnp.ndarray:
    """
    Generates random initial conditions in an area of the specified
    width and height at the required spacing.

    Args:
        key: PRNGKey
        N: (int) number of agents
        spacing: (float) how far apart positions can be
        width: (float) width of area
        height: (float) height of area

    Returns:
        (jnp.ndarray) 3xN robot poses
    """
    def grid_positions(width, height, spacing):
        x_vals = jnp.arange(-width / 2, width / 2, spacing)
        y_vals = jnp.arange(-height / 2, height / 2, spacing)
        X, Y = jnp.meshgrid(x_vals, y_vals)
        positions = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        return positions
    
    positions = grid_positions(width, height, spacing)
    num_positions = positions.shape[0]
    
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, num_positions, shape=(N,), replace=False)
    sampled_positions = positions[indices]
    
    key, subkey = jax.random.split(key)
    orientations = jax.random.uniform(subkey, shape=(N,), minval=-jnp.pi, maxval=jnp.pi)
    
    return jnp.vstack([sampled_positions.T, orientations])

def at_pose(states: jnp.ndarray, poses: jnp.ndarray, position_error=0.05, rotation_error=0.2) -> jnp.ndarray:
    """
    Checks if robots are close enough to their target poses.
    
    Args:
        states: (jnp.ndarray) 3xN unicycle states
        poses: (jnp.ndarray) 3xN desired states
        position_error: (float) allowable position error
        rotation_error: (float) allowable angular error

    Returns:
        (jnp.ndarray) 1xN array indicating agents that are close enough
    """
    # Check input types
    assert isinstance(states, jnp.ndarray), f"States must be a JAX array. Recieved type {type(states).__name__}."
    assert isinstance(poses, jnp.ndarray), f"Poses must be a JAX array. Recieved type {type(poses).__name__}."
    
    # Check input sizes
    assert poses.shape[0] == 3, "Poses must have 3 rows ([x;y;theta]"
    assert states.shape[0] == 3, "States must have 3 rows ([x;y;theta]"
    assert states.shape == poses.shape, "States and poses must have the same shape."

    # Compute position errors
    position_diffs = states[:2, :] - poses[:2, :]
    position_errors = jnp.linalg.norm(position_diffs, axis=0)

    # Compute rotation errors with angle wrapping
    rotation_diffs = states[2, :] - poses[2, :]
    rotation_errors = jnp.abs(jnp.arctan2(jnp.sin(rotation_diffs), jnp.cos(rotation_diffs)))

    # Find robots that meet both criteria
    return ((position_errors <= position_error) & (rotation_errors <= rotation_error))

def at_position(states: jnp.ndarray, points: jnp.ndarray, position_error: float=0.02):
    """
    Checks if robots are close enough to desired positions.

    Args:
        states: (jnp.ndarray) 3xN unicycle states
        points: (jnp.ndarray) 2xN desired points
        position_error: (float) allowable position error
    
    """
    # Check input types
    assert isinstance(states, jnp.ndarray), f"States must be a JAX array. Recieved type {type(states).__name__}."
    assert isinstance(points, jnp.ndarray), f"Points must be a JAX array. Recieved type {type(points).__name__}."
    assert isinstance(position_error, float), f"Position error must be a float. Recieved type {type(position_error).__name__}."

    # Check input shapes
    assert states.shape[0] == 3, "States must be (3, N)."
    assert points.shape[0] == 2, "Points must be (2, N)."
    assert states.shape[1] == points.shape[1], "Number of robots must match between states and points."

    # Compute position errors
    position_diffs = states[:2, :] - points
    position_errors = jnp.linalg.norm(position_diffs, axis=0)

    # Find robots that meet position criteria
    return position_errors <= position_error