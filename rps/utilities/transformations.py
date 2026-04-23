import numpy as np

def create_si_to_uni_dynamics(linear_velocity_gain=1, angular_velocity_limit=np.pi):
    """
    Create a mapping from single-integrator to unicycle dynamics.
    
    Returns a function that converts 2xN single-integrator velocities into 
    2xN unicycle velocities [v; omega] using a projection-based approach.
    """
    assert isinstance(linear_velocity_gain, (int, float)) and linear_velocity_gain > 0, "linear_velocity_gain must be a positive number."
    assert isinstance(angular_velocity_limit, (int, float)) and angular_velocity_limit >= 0, "angular_velocity_limit must be non-negative."

    lvg = linear_velocity_gain
    avl = angular_velocity_limit

    def si_to_uni(dxi, states):
        M, N = dxi.shape
        M_states, N_states = states.shape

        assert M == 2, f"dxi must be 2xN. Got {M}x{N}."
        assert M_states == 3, f"states must be 3xN. Got {M_states}x{N_states}."
        assert N == N_states, f"dxi and states must have the same number of columns ({N} vs {N_states})."

        dxu = np.zeros((2, N))
        for i in range(N):
            h = states[2, i]
            e_fwd  = np.array([np.cos(h), np.sin(h)])   # unit vector along heading
            e_perp = np.array([-np.sin(h), np.cos(h)])  # unit vector perpendicular (left)

            # Linear velocity: projection of dxi onto the heading direction.
            dxu[0, i] = lvg * np.dot(e_fwd, dxi[:, i])

            # Angular velocity: proportional to the angle between dxi and
            # the heading, normalised so that a 90-degree error maps to avl.
            dxu[1, i] = avl * np.arctan2(np.dot(e_perp, dxi[:, i]), np.dot(e_fwd, dxi[:, i])) / (np.pi/2)

        return dxu

    return si_to_uni


def create_si_to_uni_dynamics_with_backwards_motion(linear_velocity_gain=1, angular_velocity_limit=np.pi):
    """
    Create a mapping from single-integrator to unicycle dynamics that permits 
    reverse driving.
    
    Unlike the standard mapping, the robot may drive backwards when doing
    so requires less heading rotation than driving forwards.
    """
    assert isinstance(linear_velocity_gain, (int, float)) and linear_velocity_gain > 0, "linear_velocity_gain must be a positive number."
    assert isinstance(angular_velocity_limit, (int, float)) and angular_velocity_limit >= 0, "angular_velocity_limit must be non-negative."

    lvg = linear_velocity_gain
    avl = angular_velocity_limit

    def wrap(x):
        return np.arctan2(np.sin(x), np.cos(x))

    def si_to_uni(dxi, states):
        M, N = dxi.shape
        M_states, N_states = states.shape

        assert M == 2, f"dxi must be 2xN. Got {M}x{N}."
        assert M_states == 3, f"states must be 3xN. Got {M_states}x{N_states}."
        assert N == N_states, f"dxi and states must have the same number of columns ({N} vs {N_states})."

        dxu = np.zeros((2, N))
        for i in range(N):
            # Angle between the desired velocity direction and the heading.
            angle = wrap(np.arctan2(dxi[1, i], dxi[0, i]) - states[2, i])

            # If the error is within +/-90 deg drive forwards; otherwise reverse.
            # For reverse, flip the effective heading by pi so that the
            # projection and angular-velocity calculations remain consistent.
            if angle > -np.pi/2 and angle < np.pi/2:
                h = states[2, i]
                s = 1
            else:
                h = wrap(states[2, i] + np.pi)
                s = -1

            e_fwd  = np.array([np.cos(h), np.sin(h)])   # unit vector along effective heading
            e_perp = np.array([-np.sin(h), np.cos(h)])  # unit vector perpendicular (left)

            # Linear velocity (sign applied after so projection is correct).
            v = lvg * np.dot(e_fwd, dxi[:, i])

            # Angular velocity: normalised so a 90-degree error maps to avl.
            dxu[1, i] = avl * np.arctan2(np.dot(e_perp, dxi[:, i]), v) / (np.pi/2)
            dxu[0, i] = s * v

        return dxu

    return si_to_uni


def create_si_to_uni_mapping(projection_distance=0.05):
    """
    Create a paired dynamics and state mapping from single-integrator to 
    unicycle systems using a forward projection point.
    
    Returns two function handles:
      si_to_uni_dyn: f(dxi, states) -> dxu
      uni_to_si_states: f(states) -> xi
    """
    assert isinstance(projection_distance, (int, float)) and projection_distance > 0, "projection_distance must be a positive number."

    d = projection_distance

    # Transformation matrix from body-frame SI velocities to [v; omega].
    T = np.array([[1, 0], 
                  [0, 1/d]])

    def si_to_uni(dxi, states):
        M, N = dxi.shape
        M_states, N_states = states.shape

        assert M == 2, f"dxi must be 2xN. Got {M}x{N}."
        assert M_states == 3, f"states must be 3xN. Got {M_states}x{N_states}."
        assert N == N_states, f"dxi and states must have the same number of columns ({N} vs {N_states})."

        dxu = np.zeros((2, N))
        for i in range(N):
            h = states[2, i]
            # Rotation matrix from world frame to body frame.
            R = np.array([[np.cos(h),  np.sin(h)], 
                          [-np.sin(h), np.cos(h)]])
            dxu[:, i] = T @ R @ dxi[:, i]
            
        return dxu

    def uni_to_si(states):
        assert states.shape[0] == 3, f"states must be 3xN. Got {states.shape[0]}x{states.shape[1]}."

        xi = states[0:2, :] + d * np.array([np.cos(states[2, :]), np.sin(states[2, :])])
        return xi

    return si_to_uni, uni_to_si


def create_uni_to_si_mapping(projection_distance=0.05):
    """
    Create a paired dynamics and state mapping from unicycle to 
    single-integrator systems using a forward projection point.
    
    Returns two function handles:
      uni_to_si_dyn: f(dxu, states) -> dxi
      si_to_uni_states: f(uni_states, si_states) -> xi
    """
    assert isinstance(projection_distance, (int, float)) and projection_distance > 0, "projection_distance must be a positive number."

    d = projection_distance

    # Inverse transformation: scales omega back to a body-frame lateral velocity.
    T = np.array([[1, 0], 
                  [0, d]])

    def uni_to_si(dxu, states):
        M, N = dxu.shape
        M_states, N_states = states.shape

        assert M == 2, f"dxu must be 2xN. Got {M}x{N}."
        assert M_states == 3, f"states must be 3xN. Got {M_states}x{N_states}."
        assert N == N_states, f"dxu and states must have the same number of columns ({N} vs {N_states})."

        dxi = np.zeros((2, N))
        for i in range(N):
            h = states[2, i]
            # Rotation matrix from body frame to world frame.
            R = np.array([[np.cos(h), -np.sin(h)], 
                          [np.sin(h),  np.cos(h)]])
            dxi[:, i] = R @ T @ dxu[:, i]
            
        return dxi

    def si_to_uni(uni_states, si_states):
        M, N = uni_states.shape
        M_si, N_si = si_states.shape

        assert M == 3, f"uni_states must be 3xN. Got {M}x{N}."
        assert M_si == 2, f"si_states must be 2xN. Got {M_si}x{N_si}."
        assert N == N_si, f"uni_states and si_states must have the same number of columns ({N} vs {N_si})."

        # Subtract the projection offset to recover the unicycle origin.
        xi = si_states[0:2, :] - d * np.array([np.cos(uni_states[2, :]), np.sin(uni_states[2, :])])
        return xi

    return uni_to_si, si_to_uni