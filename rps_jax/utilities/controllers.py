import jax
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
    Creates a position controller for unicycle models. Drives a unicycle to a point using a control
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

def create_clf_unicycle_pose_controller(approach_angle_gain=1, desired_angle_gain=2.7, rotation_error_gain=1):
    """
    Creates a pose controller for unicycle models. Drives a unicycle model to a pose using a control lyapunov
    function.

    Args:
        approach_angle_gain: (float) gain for the approach angle
        desired_angle_gain: (float) gain for the desired angle
        rotation_error_gain: (float) gain for the rotation error
    
    Returns:
        (function) clf_unicycle_pose_controller
    """

    gamma = approach_angle_gain
    k = desired_angle_gain
    h = rotation_error_gain

    def clf_unicycle_pose_controller(poses: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        """
        Computes control inputs to drive the unicycle to the goal pose using a control lyapunov function.

        Args:
            poses: (jnp.ndarray) 3xN unicycle states (x, y, theta)
            goals: (jnp.ndarray) 3xN desired poses (x, y, theta)

        Returns:
            (jnp.ndarray) 2xN unicycle controls (linear velocity, angular velocity)
        """

        def compute_control(pose: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
            """
            Computes control inputs for a single unicycle model.

            Args:
                pose: (jnp.ndarray) 3x1 unicycle state (x, y, theta)
                goal: (jnp.ndarray) 3x1 desired pose (x, y, theta)
            
            Returns:
                (jnp.ndarray) 2x1 unicycle control (linear velocity, angular velocity)
            """

            # compute rotation matrix for the goal orientation
            theta = goal[2]
            c, s = jnp.cos(theta), jnp.sin(theta)
            R = jnp.array([[c, -s], [s, c]])

            # get error in the frame of the goal
            pos_error = R.T @ (goal[:2] - pose[:2])
            e = jnp.linalg.norm(pos_error)
            theta_error = jnp.arctan2(pos_error[1], pos_error[0])

            # compute control inputs
            alpha = theta_error - (pose[2] - goal[2])
            ca, sa = jnp.cos(alpha), jnp.sin(alpha)
            alpha = jnp.arctan2(jnp.sin(alpha), jnp.cos(alpha))
            u = jnp.array([
                    gamma * e * ca,
                    k * alpha + gamma * ((ca * sa) / alpha) * (alpha + h * theta_error) 
                ]
            )

            return u
        
        # compute control inputs for each robot
        u = jax.vmap(compute_control, in_axes=(1,1), out_axes=(1))(poses, goals)

        return u
    
    return clf_unicycle_pose_controller