import jax.numpy as jnp

def create_si_to_uni_dynamics(linear_velocity_gain=1.0, angular_velocity_limit=jnp.pi):
    """
    Creates a function that converts single integrator states to unicycle states.

    Args:
        linear_velocity_gain: (float) linear velocity gain
        angular_velocity_limit: (float) angular velocity limit

    Returns:
        (function) si_to_uni_dynamics
    """
    def si_to_uni_dynamics(u, poses):
        """
        Converts single integrator states to unicycle states.
        
        Args:
            u: (jnp.ndarray) 2xN single integrator controls (x and y velocities)
            poses: (jnp.ndarray) 3xN single integrator states (x, y, theta)
            
        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """
        _, N = poses.shape

        a = jnp.cos(poses[2, :]) 
        b = jnp.sin(poses[2, :])
        
        linear_velocity = linear_velocity_gain * (u[0] * a + u[1] * b)
        angular_velocity = angular_velocity_limit * jnp.arctan2(-b * u[0, :] + a * u[1, :], linear_velocity) / (jnp.pi/2)

        return jnp.vstack([linear_velocity, angular_velocity])
    
    return si_to_uni_dynamics 