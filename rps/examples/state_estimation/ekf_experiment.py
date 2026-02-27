import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import argparse
import numpy as np
import time
from rps.utilities.uni_ekf import UnicycleEKF
import matplotlib.pyplot as plt

"""
Run Unicycle EKF experiment with the passed parameters.  All parameters have defaults if you don't want to mess with them.

Parameters:
    total_waypoints: int = 20,
    encoder_noise: list[float] = [0.01, 0.01], # right and left encoder noise variances (not modeling at tick level), will be formed into matrix in code
    process_noise: list[float] = [1.0, 1.0, 1.0], # x, y, and theta process noise variances, will be formed into matrix in code
    spoof_gps_measurements: bool = False, # whether to spoof GPS measurements
    gps_measurement_interval_distribution: float = (1.0, 1.0), # distribution of GPS measurement intervals (mean, std)
    gps_measurement_noise: list[float] = [0.1, 0.1], # x and y GPS measurement noise variances, will be formed into matrix in code
    use_imu_measurements: bool = False, # will arrive on every tick if true, otherwise not used
    imu_measurement_noise: list[float] = [0.01, 0.01, 0.01]): # acc_x, acc_y, and yaw rate IMU measurement noise variances, will be formed into matrix in code

Returns:
    None

Raises:
    ValueError: If the inputs are not valid
"""
def run_ekf_experiment(total_waypoints: int = 20,
                       encoder_noise: list[float] = [0.01, 0.01], # right and left encoder noise variances (not modeling at tick level), will be formed into matrix in code
                       process_noise: list[float] = [1.0, 1.0, 1.0], # x, y, and theta process noise variances, will be formed into matrix in code
                       spoof_gps_measurements: bool = False, # whether to spoof GPS measurements
                       gps_measurement_interval_distribution: float = (1.0, 1.0), # distribution of GPS measurement intervals (mean, std)
                       gps_measurement_noise: list[float] = [0.1, 0.1], # x and y GPS measurement noise variances, will be formed into matrix in code
                       use_imu_measurements: bool = False, # will arrive on every tick if true, otherwise not used
                       imu_measurement_noise: list[float] = [0.01, 0.01, 0.01]): # acc_x, acc_y, and yaw rate IMU measurement noise variances, will be formed into matrix in code

    N = 1 # using only one robot for now
    initial_conditions = np.array([[1.0], [0.0], [0.0]], dtype=float)
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions,sim_in_real_time=True)

    goal_points = generate_initial_conditions(total_waypoints) # get random goal points

    # Create unicycle pose controller
    unicycle_pose_controller = create_hybrid_unicycle_pose_controller()

    # Create barrier certificates to avoid collision
    uni_barrier_cert = create_unicycle_barrier_certificate()

    # define x initially
    x = r.get_poses()
    r.step()

    wheel_radius = r.wheel_radius
    base_length = r.base_length
    dt = r.time_step

    # form covariance matrices from input tuples
    encoder_noise_matrix = np.eye(2) * np.array(encoder_noise)
    process_noise_matrix = np.eye(3) * np.array(process_noise)

    if spoof_gps_measurements:
        gps_measurement_noise_matrix = np.eye(2) * np.array(gps_measurement_noise)
    else:
        gps_measurement_noise_matrix = None

    if use_imu_measurements:
        imu_measurement_noise_matrix = np.eye(3) * np.array(imu_measurement_noise)
    else:
        imu_measurement_noise_matrix = None

    imu_measurement_noise_matrix = np.eye(3) * np.array(imu_measurement_noise)

    # Initialize EKF
    ekf = UnicycleEKF(initial_state=initial_conditions.flatten(), 
                initial_covariance=np.eye(3), 
                b=base_length, 
                r=wheel_radius, 
                M=encoder_noise_matrix, 
                Q=process_noise_matrix, 
                R_gps=gps_measurement_noise_matrix, 
                R_imu=imu_measurement_noise_matrix)

    # Plotting setup
    gt_trail, = r.axes.plot([], [], 'b-', linewidth=1.5, label='Ground Truth')
    ekf_trail, = r.axes.plot([], [], 'r--', linewidth=1.5, label='EKF Estimate')
    r.axes.legend(loc='upper left', fontsize=determine_font_size(r, 0.05))

    gt_history = []
    ekf_history = []
    Pdiag_history = []

    # Live variance figure (separate from Robotarium axes)
    var_fig, var_axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    var_lines = [
        var_axes[0].plot([], [], label="Var(x)")[0],
        var_axes[1].plot([], [], label="Var(y)")[0],
        var_axes[2].plot([], [], label="Var(theta)")[0],
    ]
    var_axes[0].set_ylabel("Var(x)")
    var_axes[1].set_ylabel("Var(y)")
    var_axes[2].set_ylabel("Var(theta)")
    var_axes[2].set_xlabel("time (s)")
    var_fig.suptitle("EKF state variances (diag(P)) - live")
    var_fig.tight_layout()

    for waypoint in goal_points.T: # while not all waypoints have been reached
        print(f"Next waypoint: {waypoint.reshape(3, 1)}")
        while at_pose(x, waypoint.reshape(3, 1))[0].size != N: # while not at the waypoint
            # Get poses of agents
            x = r.get_poses()
            r.get_encoders()

            # Create unicycle control inputs
            dxu = unicycle_pose_controller(x, waypoint.reshape(3, 1))

            # Create safe control inputs (i.e., no collisions)
            dxu = uni_barrier_cert(dxu, x)

            # simulate encoder noise and map to velocity and angular velocity noise
            encoder_right_std = np.sqrt(encoder_noise[0])
            encoder_left_std = np.sqrt(encoder_noise[1])
            v_noise = (wheel_radius / 2) * (np.random.randn()*encoder_right_std + np.random.randn()*encoder_left_std)
            w_noise = (wheel_radius / base_length) * (np.random.randn()*encoder_right_std - np.random.randn()*encoder_left_std)

            # Update EKF
            ekf.predict(dxu[0, 0] + v_noise, dxu[1, 0] + w_noise, dt)  # 30 Hz update rate

            # Store trajectories
            gt_history.append(x[:2, 0].copy())
            ekf_history.append(ekf.state[:2].copy())
            Pdiag_history.append(np.diag(ekf.P).copy())   # [var_x, var_y, var_theta]

            # Update Robotarium trajectory plot
            gt_arr = np.array(gt_history)
            ekf_arr = np.array(ekf_history)
            gt_trail.set_data(gt_arr[:, 0], gt_arr[:, 1])
            ekf_trail.set_data(ekf_arr[:, 0], ekf_arr[:, 1])

            # Update live variance plot
            Pdiag_arr = np.asarray(Pdiag_history)
            t_arr = np.arange(Pdiag_arr.shape[0]) * dt
            for i in range(3):
                var_lines[i].set_data(t_arr, Pdiag_arr[:, i])
                var_axes[i].relim()
                var_axes[i].autoscale_view()
            var_fig.canvas.draw_idle()
            var_fig.canvas.flush_events()

            # Set the velocities
            r.set_velocities(np.arange(N), dxu)
            # Iterate the simulation
            r.step()

    #Call at end of script to print debug information and for your script to run on the Robotarium server properly
    r.call_at_scripts_end()

def validate_inputs(total_waypoints: int, encoder_noise: np.ndarray, process_noise: np.ndarray, spoof_gps_measurements: bool, gps_measurement_interval_distribution: float, gps_measurement_noise: np.ndarray, use_imu_measurements: bool, imu_measurement_noise: np.ndarray):
    """ Validate the inputs to the run_ekf_experiment function """
    if total_waypoints < 1:
        raise ValueError("Total waypoints must be at least 1")
    if len(encoder_noise) != 2:
        raise ValueError("Encoder noise must be a list of length 2 with first being right wheel noise variance and second being left wheel noise variance")
    if len(process_noise) != 3:
        raise ValueError("Process noise must be a list of length 3 with first being x process noise variance, second being y process noise variance, and third being theta process noise variance")
    if spoof_gps_measurements and (len(gps_measurement_noise) != 2):
        raise ValueError("GPS measurement noise must be a list of length 2 with first being x GPS measurement noise variance and second being y GPS measurement noise variance if GPS measurements are spoofed")
    if use_imu_measurements and (len(imu_measurement_noise) != 3):
        raise ValueError("IMU measurement noise must be a list of length 3 with first being acc_x measurement noise variance, second being acc_y measurement noise variance, and third being yaw rate measurement noise variance if IMU measurements are used")

def parse_args():
    """ Parse the command line arguments """
    parser = argparse.ArgumentParser(description="Run EKF experimentation")
    parser.add_argument("--total_waypoints", type=int, default=10, help="Total waypoints")
    parser.add_argument("--encoder_noise", type=float, nargs=2, default=[0.01, 0.01], help="Encoder noise (right, left) as a list of two floats representing the right and left wheel noise variances")
    parser.add_argument("--process_noise", type=float, nargs=3, default=[0.01, 0.01, 0.01], help="Process noise (x, y, theta) as a list of three floats representing the x, y, and theta process noise variances")
    parser.add_argument("--spoof_gps_measurements", type=bool, default=False, help="Whether to spoof GPS measurements")
    parser.add_argument("--gps_measurement_interval_distribution", type=float, nargs=2, default=[1.0, 1.0], help="GPS measurement interval distribution (mean, std) as a list")
    parser.add_argument("--gps_measurement_noise", type=float, nargs=2, default=[0.01, 0.01], help="GPS measurement noise (x, y) as a list")
    parser.add_argument("--use_imu_measurements", type=bool, default=False, help="Whether to use IMU measurements")
    parser.add_argument("--imu_measurement_noise", type=float, nargs=3, default=[0.01, 0.01, 0.01], help="IMU measurement noise (acc_x, acc_y, yaw rate) as a list")
    return parser.parse_args()

def main():
    args = parse_args()
    validate_inputs(args.total_waypoints, np.array(args.encoder_noise), np.array(args.process_noise), args.spoof_gps_measurements, args.gps_measurement_interval_distribution, np.array(args.gps_measurement_noise), args.use_imu_measurements, np.array(args.imu_measurement_noise))
    print(f"Inputs validated, running EKF experiment with the following parameters:")
    print(f"Total waypoints: {args.total_waypoints}")
    print(f"Encoder noise: {args.encoder_noise}")
    print(f"Process noise: {args.process_noise}")
    print(f"Spoof GPS measurements: {args.spoof_gps_measurements}")
    print(f"GPS measurement interval distribution: {args.gps_measurement_interval_distribution}")
    print(f"GPS measurement noise: {args.gps_measurement_noise}")
    print(f"Use IMU measurements: {args.use_imu_measurements}")
    print(f"IMU measurement noise: {args.imu_measurement_noise}")
    print(f"Running EKF experiment...")
    run_ekf_experiment(args.total_waypoints,
                       args.encoder_noise,
                       args.process_noise,
                       args.spoof_gps_measurements,
                       args.gps_measurement_interval_distribution,
                       args.gps_measurement_noise,
                       args.use_imu_measurements,
                       args.imu_measurement_noise)


if __name__ == "__main__":
    main()