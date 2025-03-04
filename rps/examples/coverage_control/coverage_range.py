import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
import numpy as np
import pandas as pd

# Number of robots
N = 5

# Define initial conditions
initial_conditions = np.array([
    [1.25, 1.0, 1.0, -1.0, 0.1],  # x positions
    [0.25, 0.5, -0.5, -0.75, 0.2],  # y positions
    [0.0, 0.0, 0.0, 0.0, 0.0]  # orientations
])

# Create Robotarium object
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=False,
    initial_conditions=initial_conditions
)

# Simulation parameters
iterations = 1000
MIN_ITERATIONS = 1000  # Minimum iterations before convergence check
CONVERGENCE_THRESHOLD = 1e-3  # Looser convergence threshold

# Dynamics mapping
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Workspace parameters
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.0, 1.0
res = 0.05

# Generate grid points
X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.vstack([X.ravel(), Y.ravel()])  # (2, M)

# Robot parameters (heterogeneous)
R_sr = np.array([1.0, 1.0, 0.5, 1.0, 1.0])  # Sensing ranges
Vr = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Max velocities

# Density function (constant for this example)
phi_q = np.ones(grid_points.shape[1])  # Uniform density


# Weight function with linear decay
def weight_function(distance_squared, R):
    distance = np.sqrt(distance_squared)
    return np.clip(1 - (distance / R), 0, 1)


# Control gain
k_gain = 1.0

# Cost history and convergence tracking
cost_history = []
prev_smoothed_cost = None

# Main simulation loop
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # (2, N)

    # Compute distances between all robots and grid points
    diff = grid_points[:, :, None] - x_si[:, None, :]  # (2, M, N)
    D_squared = np.sum(diff ** 2, axis=0)  # (M, N)

    # Find which points are within each robot's sensing range
    within_range = D_squared <= (R_sr ** 2)[None, :]  # (M, N)

    # Mask distances outside sensing range
    D_masked = np.where(within_range, D_squared, np.inf)

    # Assign each point to the nearest robot within range
    robot_assignments = np.argmin(D_masked, axis=1)  # (M,)

    # Initialize cost components
    covered_cost = 0.0
    uncovered_penalty = 0.0

    # Calculate covered cost and centroids
    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

    for i in range(N):
        # Points assigned to this robot AND within its range
        valid_points = (robot_assignments == i) & within_range[:, i]

        if np.any(valid_points):
            # Get distances and weights for valid points
            dist_sq = D_squared[valid_points, i]
            weights = weight_function(dist_sq, R_sr[i])

            # Covered cost component
            covered_cost += np.sum(phi_q[valid_points] * weights * dist_sq) * (res ** 2)

            # Centroid calculation
            c_v[i] = np.sum(grid_points[:, valid_points] * phi_q[valid_points] * weights, axis=1)
            w_v[i] = np.sum(phi_q[valid_points] * weights)

            if w_v[i] > 1e-6:
                c_v[i] /= w_v[i]
            else:
                c_v[i] = x_si[:, i]  # Fallback to current position

        # Penalty for points in Voronoi cell but outside sensing range
        voronoi_points = (robot_assignments == i)
        out_of_range = voronoi_points & ~within_range[:, i]

        if np.any(out_of_range):
            penalty = np.sum(phi_q[out_of_range] * (R_sr[i] ** 2)) * (res ** 2)
            uncovered_penalty += penalty

    # Total cost including both components
    total_cost = covered_cost + uncovered_penalty
    cost_history.append(total_cost)

    # Compute control inputs
    si_velocities = np.zeros((2, N))

    for i in range(N):
        if w_v[i] > 1e-6:
            # Calculate desired velocity
            desired_velocity = k_gain * (c_v[i] - x_si[:, i])

            # Improved velocity scaling with safety check
            norm = np.linalg.norm(desired_velocity)
            if norm > 1e-6:  # Avoid division by zero
                scaling_factor = min(Vr[i] / norm, 1.0)
                si_velocities[:, i] = desired_velocity * scaling_factor
        else:
            # No valid points - stop the robot
            si_velocities[:, i] = np.zeros(2)

    # Transform to unicycle velocities
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set velocities and step simulation
    try:
        r.set_velocities(np.arange(N), dxu)
        r.step()

        # Enhanced convergence check
        if k >= MIN_ITERATIONS:
            smoothed_cost = np.mean(cost_history[-10:])
            if prev_smoothed_cost is not None:
                rel_change = abs(smoothed_cost - prev_smoothed_cost) / max(abs(prev_smoothed_cost), 1e-6)

                if rel_change < CONVERGENCE_THRESHOLD:
                    print(f"Converged at iteration {k} with relative change {rel_change:.6f}")
                    break
            prev_smoothed_cost = smoothed_cost

        # Detailed debugging output
        print(f"Iter {k:03d}: Cost = {total_cost:.3f} | "
              f"Δ = {rel_change:.2e}" if k > MIN_ITERATIONS else f"Iter {k:03d}: Cost = {total_cost:.3f}")
        print(f"Robot Positions:\n{np.round(x_si, 3)}")
        print(f"Velocities:\n{np.round(si_velocities, 3)}\n")

    except Exception as e:
        print(f"Error at iter {k}: {str(e)}")
        break

# Save results and cleanup
pd_cost = pd.DataFrame(cost_history)
pd_cost.to_csv("cost_case3.csv", index=False)
r.call_at_scripts_end()