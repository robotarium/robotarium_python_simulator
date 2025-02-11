import jax.numpy as jnp

def create_si_position_controller(x_velocity_gain=1, y_velocity_gain=1, velocity_magnitude_limit=0.15):
    """
    Creates a position controller for single integrators. Drives a single integrator to a point using
    a P controller

    Args:
        x_velocity_gain: (float) gain for x velocity
        y_velocity_gain: (float) gain for y velocity
        velocity_magnitude_limit: (float) limit on the velocity magnitude
    
    Returns:
        (function) si_position_controller
    """

    def si_position_controller(poses, goals):
        """
        Computes control inputs to drive the single integrator to the goal using a P controller.

        Args:
            poses: (jnp.ndarray) 2xN single integrator states (x, y)
            goals: (jnp.ndarray) 2xN desired points (x, y)

        Returns:
            (jnp.ndarray) 2xN single integrator controls (x and y velocities)
        """
        _, N = poses.shape

        # Compute position errors
        position_errors = goals - poses[:2, :]

        # Compute control inputs
        u = jnp.vstack([x_velocity_gain * position_errors[0, :],
                        y_velocity_gain * position_errors[1, :]])

        # Limit control inputs
        u_magnitude = jnp.linalg.norm(u, axis=0)
        u = jnp.where(u_magnitude > velocity_magnitude_limit, u * velocity_magnitude_limit / u_magnitude, u)

        return u
    
    return si_position_controller