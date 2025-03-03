import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_single_integrator_barrier_certificate_with_boundary
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

# Create the Robotarium object using the initial conditions
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=True,
    initial_conditions=initial_conditions
)

# Simulation parameters
iterations = 1000
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Workspace bounds and grid resolution
x_min, x_max = -1.5, 1.5
y_min, y_max = -1, 1
res = 0.05

# Generate grid points for workspace discretization
X, Y = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
grid_points = np.vstack([X.ravel(), Y.ravel()])  # Shape: (2, M)

# Robot parameters (heterogeneous sensing ranges and velocities)
R_sr = np.array([1,1,0.5,1,1])  # Sensing ranges for each robot
Vr = np.array([1,1,1,1,1])  # Velocity limits for each robot

# Control gain
k_gain = 1

# Variables to store results and track convergence
cost_list = []  # Stores cost history
prev_cost = None  # Cost from previous iteration
convergence_iter = None  # Iteration when convergence is declared

# Main simulation loop
for k in range(iterations):
    # Get current robot poses and convert to single-integrator states (positions only)
    x = r.get_poses()
    x_si = uni_to_si_states(x)[:2, :]  # Shape: (2, N)

    # Compute distances from all grid points to all robots (vectorized)
    diff = grid_points[:, None] - x_si[:, :, None]  # Shape: (2, N, M)
    D = np.linalg.norm(diff, axis=0)  # Shape: (N, M)

    # Determine grid cells within sensing range of each robot
    within_range = D <= R_sr[:, None]

    # Identify valid grid cells (covered by at least one robot)
    valid_mask = np.any(within_range, axis=0)  # Shape: (M,)

    # Mask out-of-range distances as infinity for assignment purposes
    D_masked = np.where(within_range, D, np.inf)

    # Assign each grid point to the closest robot within its sensing range
    robot_assignments = np.argmin(D_masked, axis=0)  # Robot responsible for each cell (M,)

    # Compute range-limited cost
    range_limited_cost = 0

    for i in range(N):
        # Mask for grid points assigned to robot i
        mask_i = (robot_assignments == i)

        # Covered cost (first term)
        if np.any(mask_i):
            distances_squared = np.sum((grid_points[:, mask_i] - x_si[:, i:i + 1]) ** 2, axis=0)
            covered_cost = np.sum(distances_squared * res ** 2)  # Multiply by grid resolution squared
        else:
            covered_cost = 0

        # Uncovered cost (second term)
        uncovered_mask = ~valid_mask
        if np.any(uncovered_mask):
            uncovered_cost = R_sr[i] ** 2 * np.sum(uncovered_mask * res ** 2)
        else:
            uncovered_cost = 0

        # Add to total cost
        range_limited_cost += covered_cost + uncovered_cost

    # Append cost to history for debugging or analysis
    cost_list.append(range_limited_cost)

    # Compute centroids of range-limited Voronoi regions for each robot
    c_v = np.zeros((N, 2))  # Accumulator for grid cell coordinates (x,y)
    w_v = np.zeros(N)  # Weight of grid cells assigned to each robot

    for i in range(N):
        mask_i = (robot_assignments == i) & valid_mask  # Grid cells assigned to robot i and covered
        if np.sum(mask_i) > 0:
            c_v[i] += np.sum(grid_points[:, mask_i], axis=1)
            w_v[i] += np.sum(mask_i)

    # Initialize single-integrator control inputs based on centroids
    si_velocities = np.zeros((2, N))

    for i in range(N):
        if w_v[i] > 0:
            centroid = c_v[i] / w_v[i]
            si_velocities[:, i] = k_gain * (centroid - x_si[:, i])

    # Enforce collision and boundary safety constraints using barrier certificates
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Scale velocities by maximum speed limits of each robot
    for i in range(N):
        norm_v = np.linalg.norm(si_velocities[:, i])
        if norm_v > Vr[i]:
            si_velocities[:, i] *= Vr[i] / norm_v

    # Convert single-integrator velocities to unicycle dynamics and set velocities
    dxu = si_to_uni_dyn(si_velocities, x)

    try:
        r.set_velocities(np.arange(N), dxu)
        r.step()  # Iterate the simulation

        # Debugging information
        print(f"Iteration {k}: Cost={range_limited_cost:.6f}, Centroids={c_v}, Weights={w_v}")

        # Check convergence based on relative change in cost function value
        if prev_cost is not None:
            relative_change = abs(range_limited_cost - prev_cost) / (prev_cost)
            if relative_change < 1e-3:  # Relaxed threshold for convergence
                print(f"Converged at iteration {k} with relative change {relative_change:.6f}")
                convergence_iter = k
                break

        prev_cost = range_limited_cost

    except Exception as e:
        print(f"Simulation terminated early at iteration {k} due to: {e}")
        break

# Save results to CSV file after simulation ends
df_results = pd.DataFrame({
    'Iteration': np.arange(len(cost_list)),
    'Range_Limited_Cost': cost_list,
})
df_results.to_csv('range_only_coverage_results.csv', index=False)

r.call_at_scripts_end()  # Properly close the Robotarium instance
