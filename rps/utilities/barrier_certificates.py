import warnings
import math
import numpy as np
from cvxopt import matrix, sparse
from cvxopt.solvers import qp, options

from rps.utilities.transformations import (
    create_si_to_uni_mapping,
    create_uni_to_si_mapping,
)

# Disable solver output and tune for speed
options['show_progress'] = False
options['reltol']   = 1e-2
options['feastol']  = 1e-2
options['maxiters'] = 50


def _solve_qp(N: int, vhat: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Solve  min ||v - vhat||²  s.t.  A v <= b  using CVXOPT.
    """
    H = sparse(matrix(2.0 * np.eye(2 * N)))
    f = matrix(-2.0 * vhat.reshape(-1, order='F'))
    
    try:
        sol = qp(H, f, matrix(A), matrix(b))
        if sol['status'] == 'optimal':
            return np.reshape(sol['x'], (2, N), order='F')
    except Exception:
        pass
    
    return None


def create_si_barrier_certificate(safety_radius=0.15, barrier_gain=100.0, magnitude_limit=0.2):
    """
    Creates a single-integrator barrier certificate to avoid inter-robot collisions.
    """
    def barrier_certificate(dxi, x):
        N = dxi.shape[1]
        
        if N < 2:
            return dxi

        dxi = np.copy(dxi)

        x_pos = x[:2, :]

        # Build QP inequality constraints A*v <= b
        num_pairs = math.comb(N, 2) + 8*N
        A = np.zeros((num_pairs, 2 * N))
        b = np.zeros(num_pairs)

        # Build Robot Constraints
        pair = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                diff = x_pos[:, i] - x_pos[:, j]
                h = np.dot(diff, diff) - safety_radius**2

                A[pair, 2*i:2*i+2] = -2 * diff
                A[pair, 2*j:2*j+2] =  2 * diff
                b[pair] = barrier_gain * (h**3)
                
                pair += 1

        # Build Magnitude Constraints (8-sided approximation of the l2-norm)
        constraint = pair
        for i in range(N):
            # vx <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1, 0]
            b[constraint] = magnitude_limit
            constraint += 1

            # 1/sqrt(2) * (vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1/np.sqrt(2), 1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # vy <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [0, 1]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1/np.sqrt(2), 1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vx <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1, 0]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1/np.sqrt(2), -1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vy <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [0, -1]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1/np.sqrt(2), -1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

        vnew = _solve_qp(N, dxi, A, b)
        
        if vnew is None:
            warnings.warn("create_si_barrier_certificate: QP failed. Commanding zero velocities.")
            return np.zeros((2, N))
            
        return vnew

    return barrier_certificate


def create_si_barrier_certificate_with_boundary(safety_radius=0.15, barrier_gain=100.0, magnitude_limit=0.2, boundary_points=None):
    """
    Creates a single-integrator barrier certificate that avoids inter-robot 
    collisions and keeps all robots inside a rectangular boundary.
    """
    def barrier_certificate(dxi, x):
        N = dxi.shape[1]

        if N < 2:
            return dxi

        bp = boundary_points if boundary_points is not None else np.array([-1.7, 1.7, -1.1, 1.1])
        dxi = np.copy(dxi)

        x_pos = x[:2, :]

        # Build QP inequality constraints A*v <= b
        num_pairs = math.comb(N, 2)
        num_constraints = num_pairs + 4 * N + 8 * N
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)

        # Build Robot Constraints
        pair = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                diff = x_pos[:, i] - x_pos[:, j]
                h = np.dot(diff, diff) - safety_radius**2

                A[pair, 2*i:2*i+2] = -2 * diff
                A[pair, 2*j:2*j+2] =  2 * diff
                b[pair] = barrier_gain * (h**3)

                pair += 1

        # Build Boundary Constraints
        row = num_pairs
        for k in range(N):
            cols = slice(2*k, 2*k+2)

            # +Y wall
            A[row, cols] = [0, 1]
            b[row] = 0.4 * barrier_gain * (bp[3] - safety_radius/2 - x_pos[1, k])**3
            row += 1

            # -Y wall
            A[row, cols] = [0, -1]
            b[row] = 0.4 * barrier_gain * (x_pos[1, k] - bp[2] - safety_radius/2)**3
            row += 1

            # +X wall
            A[row, cols] = [1, 0]
            b[row] = 0.4 * barrier_gain * (bp[1] - safety_radius/2 - x_pos[0, k])**3
            row += 1

            # -X wall
            A[row, cols] = [-1, 0]
            b[row] = 0.4 * barrier_gain * (x_pos[0, k] - bp[0] - safety_radius/2)**3
            row += 1

        # Build Magnitude Constraints (8-sided approximation of the l2-norm)
        constraint = row
        for i in range(N):
            # vx <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1, 0]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1/np.sqrt(2), 1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # vy <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [0, 1]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1/np.sqrt(2), 1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vx <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1, 0]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1/np.sqrt(2), -1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # -vy <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [0, -1]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

            # 1/sqrt(2) * (vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1/np.sqrt(2), -1/np.sqrt(2)]
            b[constraint] = magnitude_limit * np.cos(np.pi / 8)
            constraint += 1

        vnew = _solve_qp(N, dxi, A, b)

        if vnew is None:
            warnings.warn("create_si_barrier_certificate_with_boundary: QP failed. Commanding zero velocities.")
            return np.zeros((2, N))

        return vnew

    return barrier_certificate


def create_uni_barrier_certificate(barrier_gain=100.0, safety_radius=0.15, projection_distance=0.05, velocity_magnitude_limit=0.2):
    """
    Creates a unicycle barrier certificate function to avoid collisions by internally 
    using a single-integrator barrier certificate.
    """
    si_uni_dyn, uni_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)
    uni_si_dyn, _ = create_uni_to_si_mapping(projection_distance=projection_distance)

    si_barrier = create_si_barrier_certificate(
        safety_radius=safety_radius + 2 * projection_distance,
        barrier_gain=barrier_gain,
        magnitude_limit=velocity_magnitude_limit
    )

    def barrier_unicycle(dxu, x):
        N = dxu.shape[1]
        if N < 2:
            return dxu

        # 1. Map unicycle states/inputs -> single-integrator domain
        xi = uni_si_states(x)
        dxi = uni_si_dyn(dxu, x)

        # 2. Apply SI barrier certificate for collision safety
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dxi_safe = si_barrier(dxi, xi)
            qp_failed = any("QP failed" in str(warn.message) for warn in w)

        if qp_failed:
            warnings.warn("create_uni_barrier_certificate: Safety QP failed. Commanding zero velocities.")
            return np.zeros((2, N))

        # 3. Map safe SI velocities back to unicycle inputs
        dxu_safe = si_uni_dyn(dxi_safe, x)
        return dxu_safe

    return barrier_unicycle


def create_uni_barrier_certificate_with_boundary(safety_radius=0.12, barrier_gain=150.0, projection_distance=0.03, velocity_magnitude_limit=0.2, boundary_points=None):
    """
    Creates a unicycle barrier certificate that avoids inter-robot collisions, 
    obstacle collisions, and keeps all robots inside a rectangular boundary.
    """
    si_uni_dyn, uni_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)
    uni_si_dyn, _ = create_uni_to_si_mapping(projection_distance=projection_distance)

    bp = boundary_points if boundary_points is not None else np.array([-1.6, 1.6, -1.0, 1.0])

    si_barrier = create_si_barrier_certificate_with_boundary(
        safety_radius=safety_radius + 2 * projection_distance,
        barrier_gain=barrier_gain,
        magnitude_limit=velocity_magnitude_limit,
        boundary_points=bp
    )

    def barrier_unicycle(dxu, x, obstacles=None):
        N = dxu.shape[1]
        if N < 2:
            return dxu

        # 1. Map unicycle states/inputs -> single-integrator domain
        xi = uni_si_states(x)
        dxi = uni_si_dyn(dxu, x)

        # 2. Augment SI positions with obstacle positions
        if obstacles is not None and obstacles.size > 0:
            num_obs = obstacles.shape[1]
            xi_aug = np.hstack((xi, obstacles))
            dxi_aug = np.hstack((dxi, np.zeros((2, num_obs))))
        else:
            xi_aug = xi
            dxi_aug = dxi

        # 3. Apply SI barrier certificate for collision and boundary safety
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dxi_aug_safe = si_barrier(dxi_aug, xi_aug)
            qp_failed = any("QP failed" in str(warn.message) for warn in w)

        if qp_failed:
            warnings.warn("create_uni_barrier_certificate_with_boundary: Safety QP failed. Commanding zero velocities.")
            return np.zeros((2, N))

        # 5. Strip obstacle columns and map safe SI velocities back to unicycle
        dxi_safe = dxi_aug_safe[:, :N]
        dxu_safe = si_uni_dyn(dxi_safe, x)
        return dxu_safe

    return barrier_unicycle