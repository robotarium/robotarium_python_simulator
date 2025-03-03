import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
import numpy as np
import pandas as pd

# Number of robots
N = 5

# Define initial conditions: row 1 for x, row 2 for y, row 3 for orientation
initial_conditions = np.array([
    [1.25, 1.0, 1.0, -1.0, 0.1],  # x positions for 5 robots
    [0.25, 0.5, -0.5, -0.75, 0.2],  # y positions for 5 robots
    [0.0, 0.0, 0.0, 0.0, 0.0]  # orientations for 5 robots
])

# Create Robotarium object
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=True,
    initial_conditions=initial_conditions
)

# Simulation parameters
iterations = 1000

# Disable collision avoidance (as per experiment settings)
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Workspace bounds and grid resolution
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.0, 1.0
res = 0.05

# Generate grid points for workspace discretization
X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.vstack([X.ravel(), Y.ravel()])  # Shape: (2, M)

# Robot parameters (heterogeneous sensing ranges and velocities)
R_sr = np.array([1.0, 1.0, 0.5, 1.0, 1.0])  # Sensing ranges for each robot (normalized)
Vr = np.array([1.0] * N)  # Velocity limits for each robot (m/s)

# Density function φ(q), assumed constant (uniform distribution)
phi_q = np.random.uniform(0.5, 1.5, grid_points.shape[1])


# Weight function definition based on distance and sensing range
def weight_function(distance_squared, R):
    distance = np.sqrt(distance_squared)
    return np.maximum(1 - (distance / R), 0)


# Control gain (k=1 as per Baseline)
k_gain = 1

# Variables to store results and track convergence
cost_list = []  # Stores cost history
prev_cost = None  # Cost from previous iteration

# Main simulation loop with debugging and fixes
for k in range(iterations):
    # Get current robot poses and convert to single-integrator states (positions only)
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # Shape: (2, N)

    # Compute distances from all grid points to all robots (vectorized)
    diff = grid_points[:, None] - x_si[:, :, None]  # Shape: (2, N, M)
    D_squared = np.sum(diff ** 2, axis=0)  # Squared distances (N x M)

    # Determine grid cells within sensing range of each robot
    within_range = D_squared <= R_sr[:, None] ** 2

    # Mask out-of-range distances as infinity for assignment purposes
    D_masked = np.where(within_range, D_squared, np.inf)

    # Assign each grid point to the closest robot within its sensing range
    robot_assignments = np.argmin(D_masked, axis=0)  # Robot responsible for each cell (M,)

    # Compute range-limited cost and centroids of Voronoi regions
    range_limited_cost = 0
    c_v = np.zeros((N, 2))  # Centroid vector for each robot
    w_v = np.zeros(N)  # Weight vector for each robot

    for i in range(N):
        mask_i = (robot_assignments == i) & within_range[i]
        if np.any(mask_i):  # If any grid points are assigned to this robot
            distances_squared_i = D_squared[i][mask_i]
            weights_i = weight_function(distances_squared_i, R_sr[i])

            # Compute covered region cost (integral over Voronoi ∩ B(p_i,R))
            covered_cost = np.sum(phi_q[mask_i] * weights_i * distances_squared_i * res ** 2)
            range_limited_cost += covered_cost

            # Compute centroid of the range-limited Voronoi region
            c_v[i] += np.sum(grid_points[:, mask_i] * phi_q[mask_i] * weights_i[None], axis=1)
            w_v[i] += np.sum(phi_q[mask_i] * weights_i)

        if w_v[i] > 0:
            c_v[i] /= w_v[i]

    cost_list.append(range_limited_cost)

    # Compute control inputs (velocities) based on centroids
    si_velocities = np.zeros((2, N))

    for i in range(N):
        if w_v[i] > 0:
            centroid = c_v[i]
            velocity = k_gain * (centroid - x_si[:, i])

            # Scale velocity based on max speed of the robot
            si_velocities[:, i] = velocity / max(np.linalg.norm(velocity), Vr[i])

    dxu = si_to_uni_dyn(si_velocities, x)  # Transform SI to unicycle dynamics

    try:
        r.set_velocities(np.arange(N), dxu)  # Set velocities of agents
        r.step()  # Step simulation forward

        print(f"Iteration {k}: Cost={range_limited_cost:.6f}, Centroids={c_v}, Weights={w_v}")

        smoothed_cost = (
            np.mean(cost_list[-10:]) if len(cost_list) > 10 else range_limited_cost
        )

        if k > 10 and prev_cost is not None:
            relative_change = abs(smoothed_cost - prev_cost) / max(prev_cost, 1e-10)

            if relative_change < 1e-3:  # Convergence condition based on relative change in cost
                print(f"Converged at iteration {k} with relative change {relative_change:.6f}")
                break

        prev_cost = smoothed_cost

    except Exception as e:
        print(f"Simulation terminated early at iteration {k} due to: {e}")
        break

# Save cost data for analysis
df_costs = pd.DataFrame(cost_list)
df_costs.to_csv('range_only_coverage_debugged.csv', index=False)

r.call_at_scripts_end()