import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import argparse
import numpy as np
import time
from rps.examples.state_estimation.extended_kalman_filter.uni_ekf import UnicycleEKF
import matplotlib.pyplot as plt

"""
Run Unicycle EKF experiment with the passed parameters.  All parameters have defaults if you don't want to mess with them.

Parameters:
    cycles: int = 4,
    process_noise: list[float] = [1.0, 1.0, 1.0], 
    spoof_gps_measurements: bool = False, 
    gps_measurement_interval_distribution: float = (1.0, 1.0), 
    gps_measurement_noise: list[float] = [0.1, 0.1], 
    use_gyro_measurements: bool = False, 
    gyro_measurement_noise: float = 0.01): 
    use_orientation_measurements: bool = False, 
    orientation_measurement_noise: float = 0.01): 
Returns:
    None
"""

RECTANGLE_WAYPOINTS = [
    [-1.25, -0.75, 0.0], # 0 is just for the angle, we don't use it
    [1.25, -0.75, 0.0],
    [1.25, 0.75, 0.0],
    [-1.25, 0.75, 0.0],
]

def run_ekf_experiment(cycles: int = 4, # will drive the robot around the rectangle 3 times
                       process_noise: list[float] = [0.005, 0.005, 0.005], 
                       spoof_gps_measurements: bool = True, 
                       gps_measurement_interval_distribution: tuple = (3.0, 1.0), 
                       gps_measurement_noise: list[float] = [0.05, 0.05], 
                       use_gyro_measurements: bool = False, 
                       gyro_measurement_noise: float = 0.1, 
                       use_orientation_measurements: bool = False, 
                       orientation_measurement_noise: float = 0.1): 

    goal_points = np.array(RECTANGLE_WAYPOINTS * cycles).reshape(-1, 3)
    print(f"Goal points: {goal_points}")

    N = 1 
    initial_conditions = goal_points[0].reshape(3, 1)
    print(f"Initial conditions: {initial_conditions}")
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

    arena_width = r.BOUNDARIES[1] - r.BOUNDARIES[0] - 4*r.ROBOT_DIAMETER
    arena_height = r.BOUNDARIES[3] - r.BOUNDARIES[2] - 4*r.ROBOT_DIAMETER
    


    # Create unicycle pose controller
    unicycle_pose_controller = create_pose_controller_hybrid()

    # Create barrier certificates to avoid collision
    uni_barrier_cert = create_uni_barrier_certificate()
    
    # Create the at_pose function
    at_pose = create_at_pose()

    # define x initially
    x = r.get_poses()
    r.step()

    wheel_radius = r.WHEEL_RADIUS
    base_length = r.BASE_LENGTH
    dt = r.TIME_STEP

    # Map encoder count noise -> wheel angular velocity noise (rad/s).
    # ENCODER_NOISE_STD = 0.25 counts matches the simulator's simulate_encoder_readings().
    # sigma_ang_vel = sigma_counts * counts_to_rad / dt, so var = (sigma_counts * counts_to_rad / dt)^2.
    counts_to_rad = 2 * np.pi / (r.ENCODER_COUNTS_PER_REVOLUTION * r.MOTOR_GEAR_RATIO)
    encoder_noise_std = r.ENCODER_NOISE_STD  # 0.25 counts — matches simulate_encoder_readings()
    encoder_ang_vel_var = (encoder_noise_std * counts_to_rad / dt) ** 2
    encoder_noise_matrix = np.eye(2) * encoder_ang_vel_var
    process_noise_matrix = np.eye(3) * np.array(process_noise)

    if spoof_gps_measurements:
        gps_measurement_noise_matrix = np.eye(2) * np.array(gps_measurement_noise)
    else:
        gps_measurement_noise_matrix = None

    if use_gyro_measurements:
        gyro_measurement_noise_matrix = gyro_measurement_noise
    else:
        gyro_measurement_noise_matrix = None

    if use_orientation_measurements:
        orientation_measurement_noise_matrix = orientation_measurement_noise
    else:
        orientation_measurement_noise_matrix = None

    # Initialize EKF with updates 
    ekf = UnicycleEKF(initial_state=initial_conditions.flatten(),
                      initial_covariance=np.zeros((3, 3)),
                      b=base_length,
                      r=wheel_radius,
                      M=encoder_noise_matrix,
                      Q=process_noise_matrix,
                      )
                      
    # Second EKF: pure predict only
    ekf_pure_predict = UnicycleEKF(initial_state=initial_conditions.flatten(),
                                  initial_covariance=np.zeros((3, 3)),
                                  b=base_length,
                                  r=wheel_radius,
                                  M=encoder_noise_matrix,
                                  Q=process_noise_matrix,
                                  )
                                  
    # Plotting setup 
    gt_trail, = r._axes_handle.plot([], [], 'b-', linewidth=1.5, label='Ground Truth')
    ekf_trail, = r._axes_handle.plot([], [], 'r--', linewidth=1.5, label='EKF (with updates)')
    gps_scatter = r._axes_handle.scatter([], [], marker='x', s=80, color='green', linewidths=2, label='Spoofed GPS', zorder=5)
    r._axes_handle.legend(loc='upper left', fontsize=12)

    # =========================================================
    # FIGURE RESIZE EVENT LISTENER
    # =========================================================
    def on_resize(event):
        """Scale GPS marker size and trail linewidths when the window is resized."""
        fig  = r._fig
        ax   = r._axes_handle
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        arena_w = r.BOUNDARIES[1] - r.BOUNDARIES[0]
        # GPS cross marker: scale to ~5% of arena width
        gps_size = ((0.05 / arena_w) * bbox.width) ** 2
        gps_scatter.set_sizes([gps_size])
        ax.figure.canvas.draw_idle()

    r._fig.canvas.mpl_connect('resize_event', on_resize)

    gt_history = []
    ekf_history = []
    ekf_pure_predict_history = []
    ekf_theta_history = []
    ekf_pure_predict_theta_history = []
    Pdiag_history = []
    gps_measurement_history = []
    gps_measurement_step_indices = []  
    gyro_measurement_history = []  
    encoder_history = []  
    ground_truth_poses = []  
    orientation_measurement_history = [] 

    # One trajectory figure
    margin = 0.1
    xlim = (r.BOUNDARIES[0] - margin, r.BOUNDARIES[1] + margin)
    ylim = (r.BOUNDARIES[2] - margin, r.BOUNDARIES[3] + margin)

    traj_fig1, ax_traj1 = plt.subplots(1, 1, figsize=(6, 6))
    ax_traj1.set_aspect('equal')
    ax_traj1.set_xlim(xlim)
    ax_traj1.set_ylim(ylim)
    ax_traj1.grid(True, alpha=0.3)
    gt_line1, = ax_traj1.plot([], [], 'b-', linewidth=1.5, label='Ground Truth')
    pure_line1, = ax_traj1.plot([], [], 'm--', linewidth=1.5, label='Pure Predict')
    ax_traj1.legend(fontsize=7)
    ax_traj1.set_xlabel('x (m)', fontsize=8)
    ax_traj1.set_ylabel('y (m)', fontsize=8)
    traj_fig1.suptitle('Trajectory: Ground Truth + Pure Predict', fontsize=9)
    traj_fig1.tight_layout()

    # Error figure
    err_fig, err_axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    err_axes[0].set_ylabel("Position error (m)", fontsize=8)
    err_axes[1].set_ylabel("Angle error (rad)", fontsize=8)
    err_axes[-1].set_xlabel("time (s)", fontsize=8)
    err_fig.suptitle("Error from ground truth", fontsize=9)
    err_pos_pure, = err_axes[0].plot([], [], 'm-', linewidth=1, label='Pure predict')
    err_pos_ekf, = err_axes[0].plot([], [], 'r-', linewidth=1, label='EKF with updates')
    err_ang_pure, = err_axes[1].plot([], [], 'm-', linewidth=1, label='Pure predict')
    err_ang_ekf, = err_axes[1].plot([], [], 'r-', linewidth=1, label='EKF with updates')
    err_axes[0].legend(fontsize=7)
    err_axes[1].legend(fontsize=7)
    for ax in err_axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)
    err_fig.tight_layout()

    encoders_prev = r.get_encoders()
    sim_time = 0.0
    next_gps_measurement_time = sim_time + max(0.1, np.random.normal(gps_measurement_interval_distribution[0], gps_measurement_interval_distribution[1]))

    for waypoint in goal_points: 
        while not at_pose(x, waypoint.reshape(3, 1))[0]: 
            # Get poses of agents
            x = r.get_poses()
            encoders_curr = r.get_encoders()
            encoder_history.append(encoders_curr[:, 0].copy())
            ground_truth_poses.append(x[:, 0].copy())

            # Create unicycle control inputs
            dxu = unicycle_pose_controller(x, waypoint.reshape(3, 1))
            dxu = uni_barrier_cert(dxu, x)

            current_time = time.time()

            sim_time += dt

            if use_gyro_measurements:
                yaw_rate = r.get_gyros()[2, 0]  
                gyro_measurement_history.append(float(yaw_rate))
                ekf.update_gyro(yaw_rate, dt, gyro_measurement_noise_matrix)

            if use_orientation_measurements:
                orientation = r.get_orientations()[0]
                print(f"Orientation reading: {orientation} degrees")
                orientation_measurement_history.append(float(orientation))
                ekf.update_orientation(np.deg2rad(orientation), orientation_measurement_noise_matrix)

            if spoof_gps_measurements and sim_time >= next_gps_measurement_time:
                gps_measurement = x[:2, 0] + np.random.normal(0, np.sqrt(np.array(gps_measurement_noise)), 2)
                ekf.update_gps(gps_measurement, gps_measurement_noise_matrix)
                gps_measurement_step_indices.append(len(gt_history))
                gps_measurement_history.append(gps_measurement.copy())
                next_gps_measurement_time = sim_time + max(0.1, np.random.normal(gps_measurement_interval_distribution[0], gps_measurement_interval_distribution[1]))
                
            delta_counts_left = encoders_curr[0, 0] - encoders_prev[0, 0]
            delta_counts_right = encoders_curr[1, 0] - encoders_prev[1, 0]
            delta_phi_L = delta_counts_left * counts_to_rad
            delta_phi_R = delta_counts_right * counts_to_rad
            v_enc = (wheel_radius / 2) * (delta_phi_R + delta_phi_L) / dt
            w_enc = (wheel_radius / base_length) * (delta_phi_R - delta_phi_L) / dt

            ekf.predict(v_enc, w_enc, dt)
            ekf_pure_predict.predict(v_enc, w_enc, dt)
            encoders_prev = encoders_curr.copy()

            gt_history.append(x[:2, 0].copy())
            ekf_history.append(ekf.state[:2].copy())
            ekf_pure_predict_history.append(ekf_pure_predict.state[:2].copy())
            ekf_theta_history.append(float(np.asarray(ekf.state).flat[2]))
            ekf_pure_predict_theta_history.append(float(np.asarray(ekf_pure_predict.state).flat[2]))
            Pdiag_history.append(np.diag(ekf.P).copy())   

            gt_arr = np.array(gt_history)
            ekf_arr = np.array(ekf_history)
            gt_trail.set_data(gt_arr[:, 0], gt_arr[:, 1])
            ekf_trail.set_data(ekf_arr[:, 0], ekf_arr[:, 1])

            pure_arr = np.array(ekf_pure_predict_history)
            gt_line1.set_data(gt_arr[:, 0], gt_arr[:, 1])
            pure_line1.set_data(pure_arr[:, 0], pure_arr[:, 1])
            traj_fig1.canvas.draw_idle()
            traj_fig1.canvas.flush_events()
            if len(gps_measurement_history) > 0:
                gps_scatter.set_offsets(np.array(gps_measurement_history))
                r._fig.canvas.draw_idle()

            n = len(gt_history)
            t_arr = np.arange(n) * dt
            pos_err_pure = np.linalg.norm(gt_arr - np.array(ekf_pure_predict_history), axis=1)
            pos_err_ekf = np.linalg.norm(gt_arr - np.array(ekf_history), axis=1)
            gt_theta = np.array(ground_truth_poses)[:, 2]
            ang_diff = lambda a, b: (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
            ang_err_pure = np.abs(ang_diff(gt_theta, ekf_pure_predict_theta_history))
            ang_err_ekf = np.abs(ang_diff(gt_theta, ekf_theta_history))
            
            err_pos_pure.set_data(t_arr, pos_err_pure)
            err_pos_ekf.set_data(t_arr, pos_err_ekf)
            err_ang_pure.set_data(t_arr, ang_err_pure)
            err_ang_ekf.set_data(t_arr, ang_err_ekf)
            
            err_axes[0].relim()
            err_axes[0].autoscale_view()
            err_axes[1].relim()
            err_axes[1].autoscale_view()
            err_fig.canvas.draw_idle()
            err_fig.canvas.flush_events()

            r.set_velocities(np.arange(N), dxu)
            r.step()

    np.savez(
        'ekf_experiment_results.npz',
        gt_history=np.array(gt_history),
        ekf_history=np.array(ekf_history),
        ekf_pure_predict_history=np.array(ekf_pure_predict_history),
        Pdiag_history=np.array(Pdiag_history),
        ground_truth_poses=np.array(ground_truth_poses),
        orientation_measurement_history=np.array(orientation_measurement_history) if use_orientation_measurements else np.array([]),
        encoder_history=np.array(encoder_history),
        gyro_measurement_history=np.array(gyro_measurement_history) if use_gyro_measurements else np.array([]),
        gps_measurement_history=np.array(gps_measurement_history) if gps_measurement_history else np.array([]).reshape(0, 2),
        gps_measurement_step_indices=np.array(gps_measurement_step_indices),
        dt=dt,
        initial_conditions=initial_conditions,
        wheel_radius=wheel_radius,
        base_length=base_length,
        process_noise=np.array(process_noise),
        gps_measurement_noise=np.array(gps_measurement_noise),
        gyro_measurement_noise=np.array([gyro_measurement_noise]),
        gps_measurement_interval_distribution=np.array(gps_measurement_interval_distribution),
        spoof_gps_measurements=np.array([spoof_gps_measurements]),
        use_gyro_measurements=np.array([use_gyro_measurements]),
    )
    r.debug()

    if len(gt_history) > 0:
        plt.ioff()
        Pdiag_arr = np.asarray(Pdiag_history)

        r._fig.savefig('ekf_trajectory_with_updates.png', dpi=150)
        pure_arr = np.array(ekf_pure_predict_history)
        traj_fig1.savefig('ekf_trajectory_pure_predict.png', dpi=150)
        err_fig.savefig('ekf_error_vs_ground_truth.png', dpi=150)

        gt_arr = np.array(gt_history)
        ekf_arr = np.array(ekf_history)
        pos_err_pure = np.linalg.norm(gt_arr - pure_arr, axis=1)
        pos_err_ekf = np.linalg.norm(gt_arr - ekf_arr, axis=1)
        gt_theta = np.array(ground_truth_poses)[:, 2]
        ang_diff = lambda a, b: (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
        ang_err_pure = ang_diff(gt_theta, ekf_pure_predict_theta_history)
        ang_err_ekf = ang_diff(gt_theta, ekf_theta_history)
        rms_dist_pure = np.sqrt(np.mean(pos_err_pure**2))
        rms_dist_ekf = np.sqrt(np.mean(pos_err_ekf**2))
        rms_ang_pure = np.sqrt(np.mean(ang_err_pure**2))
        rms_ang_ekf = np.sqrt(np.mean(ang_err_ekf**2))
        print("RMS vs ground truth:")
        print("  Pure predict:  distance (xy) = {:.4f} m,  angle = {:.4f} rad".format(rms_dist_pure, rms_ang_pure))
        print("  EKF (updates): distance (xy) = {:.4f} m,  angle = {:.4f} rad".format(rms_dist_ekf, rms_ang_ekf))

        np.savez(
            'ekf_experiment_results.npz',
            gt_history=gt_arr,
            ekf_history=ekf_arr,
            ekf_pure_predict_history=pure_arr,
            ekf_theta_history=np.array(ekf_theta_history),
            ekf_pure_predict_theta_history=np.array(ekf_pure_predict_theta_history),
            Pdiag_history=Pdiag_arr,
            ground_truth_poses=np.array(ground_truth_poses),
            orientation_measurement_history=np.array(orientation_measurement_history) if use_orientation_measurements else np.array([]),
            encoder_history=np.array(encoder_history),
            gyro_measurement_history=np.array(gyro_measurement_history) if use_gyro_measurements else np.array([]),
            gps_measurement_history=np.array(gps_measurement_history) if gps_measurement_history else np.array([]).reshape(0, 2),
            gps_measurement_step_indices=np.array(gps_measurement_step_indices),
            dt=dt,
            initial_conditions=initial_conditions,
            wheel_radius=wheel_radius,
            base_length=base_length,
            process_noise=np.array(process_noise),
            gps_measurement_noise=np.array(gps_measurement_noise),
            gyro_measurement_noise=np.array([gyro_measurement_noise]),
            gps_measurement_interval_distribution=np.array(gps_measurement_interval_distribution),
            spoof_gps_measurements=np.array([spoof_gps_measurements]),
            use_gyro_measurements=np.array([use_gyro_measurements]),
            rms_dist_pure=np.array([rms_dist_pure]),
            rms_dist_ekf=np.array([rms_dist_ekf]),
            rms_ang_pure=np.array([rms_ang_pure]),
            rms_ang_ekf=np.array([rms_ang_ekf]),
        )

        if use_gyro_measurements and len(gyro_measurement_history) > 0:
            fig_gyro, ax_gyro = plt.subplots(1, 1, figsize=(6, 2.5))
            t_gyro = np.arange(len(gyro_measurement_history)) * dt
            ax_gyro.plot(t_gyro, np.array(gyro_measurement_history), 'g-', linewidth=1, label='Measured yaw rate (gyro)')
            ax_gyro.set_xlabel('time (s)', fontsize=8)
            ax_gyro.set_ylabel('yaw rate (rad/s)', fontsize=8)
            ax_gyro.legend(fontsize=7)
            ax_gyro.grid(True, alpha=0.3)
            ax_gyro.tick_params(axis='both', labelsize=7)
            fig_gyro.suptitle('Simulated gyro: measured yaw rate', fontsize=9)
            fig_gyro.tight_layout()
            fig_gyro.savefig('ekf_gyro_measured_yaw_rate.png', dpi=150)
            
        if use_orientation_measurements and len(orientation_measurement_history) > 0:
            fig_orientation, ax_orientation = plt.subplots(1, 1, figsize=(6, 2.5))
            t_orientation = np.arange(len(orientation_measurement_history)) * dt
            ax_orientation.plot(t_orientation, np.array(orientation_measurement_history), 'r-', linewidth=1, label='Measured orientation (IMU)')
            ax_orientation.set_xlabel('time (s)', fontsize=8)
            ax_orientation.set_ylabel('orientation (rad)', fontsize=8)
            ax_orientation.legend(fontsize=7)
            ax_orientation.grid(True, alpha=0.3)
            ax_orientation.tick_params(axis='both', labelsize=7)
            fig_orientation.suptitle('Simulated IMU: measured orientation', fontsize=9)
            fig_orientation.tight_layout()
            fig_orientation.savefig('ekf_imu_measured_orientation.png', dpi=150)
            
        plt.show(block=True)

def validate_inputs(cycles: int, process_noise: np.ndarray, spoof_gps_measurements: bool, gps_measurement_interval_distribution: float, gps_measurement_noise: np.ndarray, use_gyro_measurements: bool, gyro_measurement_noise: float, use_orientation_measurements: bool, orientation_measurement_noise: float):
    if cycles < 1:
        raise ValueError("Cycles must be at least 1")
    if len(process_noise) != 3:
        raise ValueError("Process noise must be a list of length 3")
    if spoof_gps_measurements and (len(gps_measurement_noise) != 2):
        raise ValueError("GPS measurement noise must be a list of length 2")
    if use_gyro_measurements and (gyro_measurement_noise <= 0):
        raise ValueError("Gyro measurement noise must be a float > 0")
    if use_orientation_measurements and (orientation_measurement_noise <= 0):
        raise ValueError("Orientation measurement noise must be a float > 0")

def parse_args():
    parser = argparse.ArgumentParser(description="Run EKF experimentation")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--process_noise", type=float, nargs=3, default=[0.0001, 0.0001, 0.0001])
    parser.add_argument("--spoof_gps_measurements", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gps_measurement_interval_distribution", type=float, nargs=2, default=[5.0, 2.5])
    parser.add_argument("--gps_measurement_noise", type=float, nargs=2, default=[0.0001, 0.0001])
    parser.add_argument("--use_gyro_measurements", action='store_true', default=False)
    parser.add_argument("--gyro_measurement_noise", type=float, default=0.001)
    parser.add_argument("--use_orientation_measurements", action='store_true', default=False)
    parser.add_argument("--orientation_measurement_noise", type=float, default=0.001)
    return parser.parse_args()

def main():
    args = parse_args()
    validate_inputs(args.cycles, np.array(args.process_noise), args.spoof_gps_measurements, args.gps_measurement_interval_distribution, np.array(args.gps_measurement_noise), args.use_gyro_measurements, args.gyro_measurement_noise, args.use_orientation_measurements, args.orientation_measurement_noise)
    run_ekf_experiment(args.cycles,
                       args.process_noise,
                       args.spoof_gps_measurements,
                       args.gps_measurement_interval_distribution,
                       args.gps_measurement_noise,
                       args.use_gyro_measurements,
                       args.gyro_measurement_noise,
                       args.use_orientation_measurements,
                       args.orientation_measurement_noise)

if __name__ == "__main__":
    main()