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

    def si_position_controller(poses: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
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
        pos_error = goals - poses[:2, :]

        # Compute control inputs
        u = jnp.vstack([x_velocity_gain * pos_error[0, :],
                        y_velocity_gain * pos_error[1, :]])

        # Limit control inputs
        u_magnitude = jnp.linalg.norm(u, axis=0)
        u = jnp.where(u_magnitude > velocity_magnitude_limit, u * velocity_magnitude_limit / u_magnitude, u)

        return u
    
    return si_position_controller

def create_clf_unicycle_position_controller(linear_velocity_gain=0.8, angular_velocity_gain=3):
    """
    Creates a position controller for unicycle moels. Drives a unicycle to a point using a control
    lyaupnov function (clf).

    Args:
        linear_velocity_gain: (float) gain for linear velocity
        angular_velocity_gain: (float) gain for angular velocity
    
    Returns:
        (function) clf_unicycle_position_controller
    """

    def clf_unicycle_position_controller(poses: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        """
        Computes control inputs to drive the unicycle to the goal using a control lyapunov function.

        Args:
            poses: (jnp.ndarray) 3xN unicycle states (x, y, theta)
            goals: (jnp.ndarray) 2xN desired points (x, y)

        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        _, N = poses.shape

        # Compute errors
        pos_error = goals - poses[:2, :]
        rot_error = jnp.arctan2(pos_error[1, :], pos_error[0, :])

        # Compute magnitude of position error
        m = jnp.linalg.norm(pos_error, axis=0)

        # Compute control inputs
        u = jnp.vstack(
            [
                linear_velocity_gain * m * jnp.cos(rot_error - poses[2, :]),
                angular_velocity_gain * m * jnp.sin(rot_error - poses[2, :])
            ]
        )

        return u
    
    return clf_unicycle_position_controller