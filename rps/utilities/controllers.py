import numpy as np
from rps.utilities.transformations import *

def create_si_position_controller(x_velocity_gain=0.8, y_velocity_gain=0.8, velocity_magnitude_limit=0.15):
    """Creates a single-integrator position controller.
    
    Returns a function that drives N single-integrator robots toward desired positions 
    using proportional control, with output velocities clamped to velocity_magnitude_limit.
    """
    assert isinstance(x_velocity_gain, (int, float)) and x_velocity_gain > 0, "x_velocity_gain must be a positive number."
    assert isinstance(y_velocity_gain, (int, float)) and y_velocity_gain > 0, "y_velocity_gain must be a positive number."
    assert isinstance(velocity_magnitude_limit, (int, float)) and velocity_magnitude_limit >= 0, "velocity_magnitude_limit must be non-negative."

    gains = np.diag([x_velocity_gain, y_velocity_gain])

    def position_controller_si(states, poses):
        N = states.shape[1]
        assert states.shape[0] == 2, "states must be 2xN."
        assert poses.shape[0] == 2, "poses must be 2xN."
        assert poses.shape[1] == N, "states and poses must have the same number of columns."

        dx = gains @ (poses - states)

        norms = np.linalg.norm(dx, axis=0)
        to_clamp = norms > velocity_magnitude_limit
        if np.any(to_clamp):
            dx[:, to_clamp] = velocity_magnitude_limit * dx[:, to_clamp] / norms[to_clamp]

        return dx

    return position_controller_si


def create_uni_position_controller(x_velocity_gain=0.8, y_velocity_gain=0.8, velocity_magnitude_limit=0.15, projection_distance=0.03):
    """Creates a unicycle position controller.
    
    Returns a function that drives N unicycle-modeled robots toward desired 2D positions 
    by wrapping a single-integrator controller with a projection mapping.
    """
    si_controller = create_si_position_controller(
        x_velocity_gain=x_velocity_gain, 
        y_velocity_gain=y_velocity_gain, 
        velocity_magnitude_limit=velocity_magnitude_limit
    )

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    def position_controller_uni(states, poses):
        assert states.shape[0] == 3, "states must be 3xN."
        assert poses.shape[0] == 2, "poses must be 2xN."
        assert poses.shape[1] == states.shape[1], "states and poses must have the same number of columns."

        xi = uni_to_si_states(states)
        dxi = si_controller(xi, poses)
        dxu = si_to_uni_dyn(dxi, states)

        return dxu

    return position_controller_uni


def create_pose_controller_clf(approach_angle_gain=1, desired_angle_gain=2.7, rotation_error_gain=1, 
                               wheel_velocity_limit=10.0, wheel_radius=0.016, base_length=0.105):
    """Creates a unicycle pose controller based on a Control Lyapunov Function (CLF).
    
    Returns a function that drives N unicycle-modeled robots to desired poses 
    (position + orientation) simultaneously.
    """
    gamma = approach_angle_gain
    k = desired_angle_gain
    h = rotation_error_gain

    D = np.array([[wheel_radius/2, wheel_radius/2], 
                  [-wheel_radius/base_length, wheel_radius/base_length]])
    Dinv = np.linalg.inv(D)

    def controller_clf(states, poses):
        N = states.shape[1]
        assert states.shape[0] == 3, "states must be 3xN."
        assert poses.shape[0] == 3, "poses must be 3xN."
        assert poses.shape[1] == N, "states and poses must have the same number of columns."

        dxu = np.zeros((2, N))

        for i in range(N):
            theta_goal = poses[2, i]
            R = np.array([[np.cos(-theta_goal), -np.sin(-theta_goal)], 
                          [np.sin(-theta_goal),  np.cos(-theta_goal)]])
            
            pos_error = R @ (poses[:2, i] - states[:2, i])

            e = np.linalg.norm(pos_error)
            theta = np.arctan2(pos_error[1], pos_error[0])
            alpha = theta - (states[2, i] - poses[2, i])
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

            ca = np.cos(alpha)
            sa = np.sin(alpha)

            dxu[0, i] = gamma * e * ca
            
            # Prevent division by zero mathematically
            if np.abs(alpha) < 1e-6:
                sinc_alpha = 1.0
            else:
                sinc_alpha = sa / alpha
                
            dxu[1, i] = k * alpha + gamma * (ca * sinc_alpha) * (alpha + h * theta)

            wheel_vels = Dinv @ dxu[:, i]
            max_wheel = np.max(np.abs(wheel_vels))
            if max_wheel > wheel_velocity_limit:
                dxu[:, i] = dxu[:, i] * (wheel_velocity_limit / max_wheel)

        return dxu

    return controller_clf


def create_pose_controller_hybrid(linear_velocity_gain=0.8, angular_velocity_limit=np.pi, 
                                  velocity_magnitude_limit=0.15, position_error=0.03, 
                                  position_epsilon=0.01, rotation_error=0.05):
    """Creates a unicycle pose controller based on a two-phase hybrid control strategy.
    
    Returns a function that drives N unicycle robots to desired poses using a 
    sequential two-phase approach (APPROACH -> ROTATE).
    """
    si_to_uni = create_si_to_uni_dynamics(linear_velocity_gain=linear_velocity_gain, 
                                          angular_velocity_limit=angular_velocity_limit)

    # Mimics MATLAB's persistent closure state
    internal_state = {'approach_state': np.array([])}

    def controller_hybrid(states, poses, in_approach_state=None):
        N = states.shape[1]

        if in_approach_state is not None:
            approach_state = in_approach_state
        else:
            if internal_state['approach_state'].shape[0] != N:
                internal_state['approach_state'] = np.ones(N)
            approach_state = internal_state['approach_state']

        dxu = np.zeros((2, N))

        for i in range(N):
            pos_vec = poses[:2, i] - states[:2, i]
            pos_dist = np.linalg.norm(pos_vec)
            
            heading_err = poses[2, i] - states[2, i]
            heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))

            if approach_state[i] and pos_dist > position_error - position_epsilon:
                # Phase 1: APPROACH
                if pos_dist > velocity_magnitude_limit:
                    pos_vec = velocity_magnitude_limit * pos_vec / pos_dist
                
                dx_uni = si_to_uni(pos_vec.reshape(2, 1), states[:, i].reshape(3, 1))
                dxu[:, i] = dx_uni.flatten()

            elif np.abs(heading_err) > rotation_error:
                # Phase 2: ROTATE
                approach_state[i] = 1 if pos_dist > position_error else 0
                dxu[0, i] = 0
                dxu[1, i] = 2 * heading_err

            else:
                # CONVERGED
                dxu[:, i] = 0

        if in_approach_state is not None:
            return dxu, approach_state
        return dxu

    return controller_hybrid


def create_pose_parking_controller_clf(approach_angle_gain=1, desired_angle_gain=2.7, rotation_error_gain=1, 
                                       wheel_velocity_limit=10.0, wheel_radius=0.016, base_length=0.105, 
                                       position_error=0.05, rotation_error=0.2):
    """Wraps create_pose_controller_clf to zero out commands for robots that have 
    already reached their goal pose.
    """
    clf_controller = create_pose_controller_clf(
        approach_angle_gain=approach_angle_gain,
        desired_angle_gain=desired_angle_gain,
        rotation_error_gain=rotation_error_gain,
        wheel_velocity_limit=wheel_velocity_limit,
        wheel_radius=wheel_radius,
        base_length=base_length
    )

    def controller_clf_parking(states, poses):
        dxu = clf_controller(states, poses)

        pos_converged = np.linalg.norm(states[:2, :] - poses[:2, :], axis=0) < position_error
        rot_converged = np.abs(np.arctan2(np.sin(states[2, :] - poses[2, :]),
                                          np.cos(states[2, :] - poses[2, :]))) < rotation_error
        converged = pos_converged & rot_converged

        dxu[:, converged] = 0
        return dxu

    return controller_clf_parking


def create_pose_parking_controller_hybrid(linear_velocity_gain=0.8, angular_velocity_limit=np.pi, 
                                          velocity_magnitude_limit=0.15, position_error=0.05, 
                                          position_epsilon=0.01, rotation_error=0.2):
    """Wraps create_pose_controller_hybrid to zero out commands for robots that have 
    already reached their goal pose.
    """
    hybrid_controller = create_pose_controller_hybrid(
        linear_velocity_gain=linear_velocity_gain,
        angular_velocity_limit=angular_velocity_limit,
        velocity_magnitude_limit=velocity_magnitude_limit,
        position_error=position_error,
        position_epsilon=position_epsilon,
        rotation_error=rotation_error
    )

    def controller_hybrid_parking(states, poses):
        dxu = hybrid_controller(states, poses)

        pos_converged = np.linalg.norm(states[:2, :] - poses[:2, :], axis=0) < position_error
        rot_converged = np.abs(np.arctan2(np.sin(states[2, :] - poses[2, :]),
                                          np.cos(states[2, :] - poses[2, :]))) < rotation_error
        converged = pos_converged & rot_converged

        dxu[:, converged] = 0
        return dxu

    return controller_hybrid_parking