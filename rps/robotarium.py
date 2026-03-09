import math
import time
    
import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from rps.robotarium_abc import RobotariumABC
from rps.utilities.misc import rotation_matrix

# Robotarium This object provides routines to interface with the Robotarium.
#
# THIS CLASS SHOULD NEVER BE MODIFIED OR SUBMITTED

class Robotarium(RobotariumABC):
    def __init__(
        self,
        number_of_robots: int = -1,
        show_figure: bool = True,
        sim_in_real_time: bool = True,
        initial_conditions: NDArray[np.floating] = np.array([]),
        use_distance_sensors: bool = False,
        obstacles: NDArray[np.floating] = np.full((1,2,2), np.nan)
    ):
        """
        Instantiate the Robotarium Simulator

        Args:
            number_of_robots (int): The number of robots in the simulation. Must be a positive integer.
            show_figure (bool): Whether to display the simulation figure window. Set to False to disable graphics and speed up the simulation.
            sim_in_real_time (bool): Whether to run the simulation in real time, with each loop taking approximately 0.033 seconds. Set to False to run as fast as possible, which may be useful for debugging but will not reflect real-world timing.
            initial_conditions (np.ndarray): A 3xN numpy array specifying the initial poses of the N robots. Each column should be [x; y; theta] for a robot. If left empty, robots will be initialized at random positions and orientations.
            use_distance_sensors (bool): Whether to simulate distance sensor readings for the robots. Enabling this will add realistic sensor noise and effects, but may slow down the simulation.
            obstacles (np.ndarray): An Mx2x2 numpy array specifying M obstacles in the environment. Each obstacle is defined by two endpoints [[x1, x2], [y1, y2]]. If left empty, no obstacles will be present. 
        """
        super().__init__(number_of_robots, show_figure, sim_in_real_time, initial_conditions, use_distance_sensors, obstacles)

        #Initialize some rendering variables
        self.previous_render_time = time.time()
        self.sim_in_real_time = sim_in_real_time

        #Initialize checks for step and get poses calls
        self._called_step_already = True
        self._checked_poses_already = False

        #Initialization of error collection.
        self._errors = {}

        #Initialize steps
        self._iterations = 0

        # Draw obstacles if any
        if(self.show_figure):
            if self.obstacles is not None:
                num_obstacles = self.obstacles.shape[0]
                for i in range(num_obstacles):
                    obstacle_patch = Line2D([self.obstacles[i,0,0], self.obstacles[i,0,1]], 
                                            [self.obstacles[i,1,0], self.obstacles[i,1,1]], 
                                            linewidth=4, color='0.5')
                    self.axes.add_line(obstacle_patch)


    def get_poses(self) -> NDArray[np.floating]:
        """Returns the states of the agents.

        -> 3xN numpy array (of robot poses)
        """

        assert(not self._checked_poses_already), "Can only call get_poses() once per call of step()."
        # Allow step() to be called again.
        self._called_step_already = False
        self._checked_poses_already = True 

        return self.poses.copy()

    def call_at_scripts_end(self):
        """Call this function at the end of scripts to display potentail errors.  
        Even if you don't want to print the errors, calling this function at the
        end of your script will enable execution on the Robotarium testbed.
        """
        print('##### DEBUG OUTPUT #####')
        print('Your simulation will take approximately {0} real seconds when deployed on the Robotarium. \n'.format(math.ceil(self._iterations*self.time_step)))
        # TODO: check collision string and boundary string
        if bool(self._errors):
            if "boundary" in self._errors:
                boundary_violations = max(self._errors["boundary"].values())
                print('\t Simulation had {0} {1}\n'.format(boundary_violations, self._errors["boundary_string"]))
            if "collision" in self._errors:
                collision_violations = max(self._errors["collision"].values())
                print('\t Simulation had {0} {1}\n'.format(collision_violations, self._errors["collision_string"]))
            if "actuator" in self._errors:
                print('\t Simulation had {0} {1}'.format(self._errors["actuator"], self._errors["actuator_string"]))
        else:
            print('No errors in your simulation! Acceptance of your experiment is likely!')

        return

    def step(self):
        """Increments the simulation by updating the dynamics.
        """
        assert(not self._called_step_already), "Make sure to call get_poses before calling step() again."
        
        # Allow get_poses function to be called again.
        self._called_step_already = True
        self._checked_poses_already = False

        # Validate before thresholding velocities
        self._errors = self._validate()
        self._iterations += 1

        #Perform Thresholding of Motors
        self.velocities = self._threshold(self.velocities)

        # Update dynamics of agents
        self.poses[0, :] = self.poses[0, :] + self.time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
        self.poses[1, :] = self.poses[1, :] + self.time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
        self.poses[2, :] = self.poses[2, :] + self.time_step*self.velocities[1, :]
        # Ensure angles are wrapped
        self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

        # Simulate encoder readings
        self._simulate_encoder_readings()

        # Simulate distance measurements
        if self.distance_sensors_enabled:
            self._simulate_distance_measurements()

        # Simulate IMU measurements
        self._simulate_accelerations()
        self._simulate_gyros()
        self._simulate_magnetometers()
        self._simulate_orientation()

        # Update graphics
        if(self.show_figure):
            for i in range(self.number_of_robots):
                # self.chassis_patches[i].xy = self.poses[:2, i] + self.robot_radius*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                # if i == 0:
                #     print(self.poses[2, i] - math.pi/2)
                #     print('='*50)

                # self.chassis_patches[i].xy = self.poses[:2, i] + np.array(-self.robot_width/2 * np.sin(self.poses[2, i] + math.pi/2), self.robot_length/2 * np.cos(self.poses[2,i] + math.pi/2))
                self.chassis_patches[i].xy = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                        0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                # self.chassis_patches[i].orientation = self.poses[2, i] + math.pi/4
                self.chassis_patches[i].angle = (self.poses[2, i] - math.pi/2) * 180/math.pi

                self.chassis_patches[i].zorder = 2

                self.right_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                        0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                self.right_wheel_patches[i].orientation = self.poses[2, i] + math.pi/4

                self.right_wheel_patches[i].zorder = 2

                self.left_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                        0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                self.left_wheel_patches[i].orientation = self.poses[2,i] + math.pi/4

                self.left_wheel_patches[i].zorder = 2
                
                self.led_patches[i].center = self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                                0.015*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                # self.base_patches[i].center = self.poses[:2, i]

                self.led_patches[i].zorder = 2

                # Update distance sensor rays
                if self.distance_sensors_enabled:
                    self.distance_ray_patch.set_offsets(self.distance_end_points.T)

            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

        if(self.sim_in_real_time):
                t = time.time()
                while(t - self.previous_render_time < self.time_step):
                    t=time.time()
                self.previous_render_time = t

    def _simulate_encoder_readings(self):
        # Simulate encoder readings based on wheel velocities
        left_motor_angular_velocity = self._uni_to_diff(self.velocities)[0, :]
        right_motor_angular_velocity = self._uni_to_diff(self.velocities)[1, :]

        delta_encoder = self.encoder_counts_per_revolution*self.motor_gear_ratio/(2*np.pi)*np.vstack((left_motor_angular_velocity, right_motor_angular_velocity))*self.time_step
        
        # Per-count noise: error std scales with sqrt(number of counts) in this step
        n_counts = np.maximum(np.abs(delta_encoder), 1.0)
        step_noise_std = self.encoder_noise_std * np.sqrt(n_counts)
        encoder_noise = np.random.normal(0, step_noise_std, delta_encoder.shape)
        delta_encoder += encoder_noise
        self.encoders += np.round(delta_encoder) # this is another source of noise with an std of 0.5

    def _simulate_distance_measurements(self):
        """ Simulates the distance sensor readings for all robots in the Robotarium.

        For each robot and each of its sensors, this function:
            1. Computes the global position and orientation of the sensor.
            2. Casts a ray in the sensor direction and finds intersections with
                obstacles and other robots.
            3. Adds realistic sensor effects:
                - Gaussian noise around the true distance.
                - Random “outlier” measurements (random readings anywhere in the distance sensor range).
                - Random dropouts (sensor returns no measurement).
            4. Clamps all readings to the valid sensor range and converts NaNs to -1
                to match the Robotarium API.
        
        This allows testing algorithms with realistic, noisy distance data
        similar to what physical robots would produce. """
        
        N_sensors = self.distance_sensors_orientation.shape[1]
        N_obstacles = self.obstacles.shape[0]
        self.distances = -1*np.ones((N_sensors, self.number_of_robots))  # Reset distances

        # Find global positions and orientations of distance sensors
        R = rotation_matrix(self.poses[2, :]) # N x 3 x 3
        poses = self.poses.T[:, :, None] # N x 3 x 1 for batch matrix multiplication
        global_sensors = poses + np.matmul(R, self.distance_sensors_orientation)  # N x 3 x 7

        # Calculate the endpoints of each sensor ray at max range
        R_sensor = rotation_matrix(global_sensors[:, 2, :]) # N x 3 x 3
        max_distances = np.stack([self.distance_sensor_range*np.ones((self.number_of_robots*N_sensors, 1)),
                                    np.zeros((self.number_of_robots*N_sensors, 1)),
                                    np.zeros((self.number_of_robots*N_sensors, 1))], axis = 1) # 7*N x 3 x 1
        sensor_endpoints_local = np.matmul(R_sensor, max_distances).squeeze(-1).reshape(self.number_of_robots, N_sensors, 3).transpose(0, 2, 1)  # N x 3 x 7
        sensor_endpoints = global_sensors[:, 0:2, :] + sensor_endpoints_local[:, 0:2, :]  # N x 2 x 7
        self.distance_end_points = sensor_endpoints.transpose(1, 0, 2).reshape(2, self.number_of_robots*N_sensors) # 2 x N*7

        # Compute intersections of each sensor ray with each obstacle
        r_all = sensor_endpoints - global_sensors[:, 0:2, :]  # N x 2 x 7. Vectors from sensor origin to max range endpoint
        s_all = self.obstacles[:, :, 1] - self.obstacles[:, :, 0]  # N x 2 x 2. Vectors from start to end of each obstacle edge
        s_all = s_all.reshape(N_obstacles, 2, 1) # num_obstacle x 2 x 1 for batch matrix multiplication

        for i in range(self.number_of_robots):
            rxs = r_all[i, 0, :].reshape(1, 1, N_sensors)*s_all[:, 1, :].reshape(N_obstacles, 1, 1) - \
                    r_all[i, 1, :].reshape(1, 1, N_sensors)*s_all[:, 0, :].reshape(N_obstacles, 1, 1)  # num_obstacles x 1 x N_sensors
            q = self.obstacles[:, :, 0].reshape(N_obstacles, 2, 1) - global_sensors[i, 0:2, :].reshape(1, 2, 7)  # num_obstacles x 2 x N_sensors
            qxs = q[:, 0, :].reshape(N_obstacles, 1, N_sensors)*s_all[:, 1, :].reshape(N_obstacles, 1, 1) - \
                    q[:, 1, :].reshape(N_obstacles, 1, N_sensors)*s_all[:, 0, :].reshape(N_obstacles, 1, 1)  # num_obstacles x 1 x N_sensors
            qxr = q[:, 0, :].reshape(N_obstacles, 1, N_sensors)*r_all[i, 1, :].reshape(1, 1, N_sensors) - \
                    q[:, 1, :].reshape(N_obstacles, 1, N_sensors)*r_all[i, 0, :].reshape(1, 1, N_sensors)  # num_obstacles x 1 x N_sensors

            # Avoid divide-by-zero when the ray and obstacle segment are parallel (rxs == 0)
            t = np.full_like(qxs, np.nan, dtype=float)  # num_obstacles x 1 x N_sensors
            u = np.full_like(qxr, np.nan, dtype=float)  # num_obstacles x 1 x N_sensors
            rxs_nonzero = rxs != 0
            np.divide(qxs, rxs, out=t, where=rxs_nonzero)  # Parameter for the intersection on the sensor lines
            np.divide(qxr, rxs, out=u, where=rxs_nonzero)  # Parameter for the intersection on the obstacle lines

            parameter_on_line = np.logical_and(np.logical_and(t >= 0, t <= 1), np.logical_and(u >= 0, u <= 1)) # num_obstacles x 1 x N_sensors
            valid_parameter = t*parameter_on_line # num_obstacles x 1 x N_sensors
            # valid_parameter[~parameter_on_line] = self.distance_sensor_range # num_obstacles x 1 x N_sensors. Set invalid intersections to NaN
            valid_parameter[~parameter_on_line] = np.nan # num_obstacles x 1 x N_sensors. Set invalid intersections to NaN

            # Check if any rays intersect other robots
            f = global_sensors[i, 0:2, :].reshape(1, 2, N_sensors) - self.poses[:2, :].T.reshape(self.number_of_robots, 2, 1)  # N x 2 x N_sensors. Vectors from other robots to sensor origin
            a = r_all[i, 0, :].reshape(1, 1, N_sensors)**2 + r_all[i, 1, :].reshape(1, 1, N_sensors)**2  # 1 x 1 x N_sensors. Squared magnitude of ray direction vectors
            b = 2*(f[:, 0, :].reshape(self.number_of_robots, 1, N_sensors)*r_all[i, 0, :].reshape(1, 1, N_sensors) + \
                    f[:, 1, :].reshape(self.number_of_robots, 1, N_sensors)*r_all[i, 1, :].reshape(1, 1, N_sensors))  # N x 1 x N_sensors. Dot product of 2*f and ray direction vectors
            c = f[:, 0, :].reshape(self.number_of_robots, 1, N_sensors)**2 + f[:, 1, :].reshape(self.number_of_robots, 1, N_sensors)**2 - \
                self.robot_radius**2  # N x 1 x N_sensors. Squared magnitude of f minus robot radius squared
            discriminant = b**2 - 4*a*c  # N x 1 x N_sensors. Discriminant of quadratic formula
            # Only take sqrt where discriminant is non-negative; otherwise there is no real intersection.
            # This avoids RuntimeWarning: invalid value encountered in sqrt.
            t_circle = np.full_like(discriminant, np.nan, dtype=float)  # N x 1 x N_sensors
            sqrt_discriminant = np.full_like(discriminant, np.nan, dtype=float)
            real_intersection = np.logical_and(discriminant >= 0, a > 0)
            np.sqrt(discriminant, out=sqrt_discriminant, where=real_intersection)
            np.divide(-b - sqrt_discriminant, 2*a, out=t_circle, where=real_intersection)

            parameter_on_line_circle = np.logical_and(t_circle >= 0, t_circle <= 1)  # N x 1 x N_sensors
            valid_parameter_circle = t_circle*parameter_on_line_circle  # N x 1 x N_sensors
            # valid_parameter_circle[~parameter_on_line_circle] = self.distance_sensor_range  # N x 1 x N_sensors. Set invalid intersections to max range
            valid_parameter_circle[~parameter_on_line_circle] = np.nan  # N x 1 x N_sensors. Set invalid intersections to NaN
            
            valid_parameter_all = np.vstack((valid_parameter, valid_parameter_circle))  # Combine obstacle and robot intersection parameters
            # Avoid RuntimeWarning: All-NaN slice encountered (no intersections)
            min_parameter = np.min(np.where(np.isnan(valid_parameter_all), np.inf, valid_parameter_all), axis=0).squeeze(0)  # 1 x N_sensors
            min_parameter[np.isinf(min_parameter)] = np.nan
            # self.distances[:, i] = min_parameter + self.distance_sensor_error*(2*np.random.rand(1, N_sensors) - 1)  # Add noise to distance measurements
            
            # =========================================
            # Mixture sensor model
            # =========================================

            p_d = self.distance_sensor_dropout_prob
            p_o = self.distance_sensor_outlier_prob

            noisy_parameter = min_parameter.copy()
            valid = ~np.isnan(min_parameter)

            # Single random draw per measurement
            u = np.random.rand(N_sensors)

            # Mutually exclusive masks
            dropout_mask = (u < p_d)
            outlier_mask = (u >= p_d) & (u < p_d + p_o)
            nominal_mask = (u >= p_d + p_o)

            # --- Gaussian noise ---
            nominal_valid = nominal_mask & valid

            sigma = self.distance_sensor_error * min_parameter[nominal_valid]
            noise = sigma * np.random.randn(sigma.shape[0])
            noise = np.clip(noise, -3*sigma, 3*sigma)       # Clip Gaussian noise to ±3σ

            noisy_parameter[nominal_valid] = (
                min_parameter[nominal_valid] + noise
            )

            # --- Outliers ---
            outlier_valid = outlier_mask & valid
            random_values = self.distance_sensor_range * np.random.rand(N_sensors)      # get a random reading w/ uniform prob in dist sens range
            noisy_parameter[outlier_valid] = random_values[outlier_valid]

            # --- Dropouts ---
            noisy_parameter[dropout_mask] = np.nan

            # --- Clamp valid values ---
            valid_after = ~np.isnan(noisy_parameter)
            noisy_parameter[valid_after] = np.clip(noisy_parameter[valid_after], 0, self.distance_sensor_range)

            self.distances[:, i] = noisy_parameter

        # Find the endpoint of each sensor ray
        distance_end_points = global_sensors[:, 0:2, :] + self.distances.T.reshape(self.number_of_robots, 1, N_sensors)*r_all # N x 2 x 7
        self.distance_end_points = distance_end_points.transpose(1, 0, 2).reshape(2, self.number_of_robots*N_sensors) # 2 x N*7

        # Convert NaN distances to -1 for consistency with real robot API
        self.distances[np.isnan(self.distances)] = -1

    def _simulate_accelerations(self):
        """
        Simulates the accelerometer readings for the robots based on their current velocities and the physics of the system. This includes:
            - Translational acceleration (change in linear velocity)
            - Tangential acceleration (due to angular acceleration)
            - Centripetal acceleration (due to angular velocity)

        Additionally, the IMU is oriented such that the X-axis points forward, the Y-axis points left, and the Z-axis points up.
        """
        linear_accelerations = ((self.velocities[0, :] - self.velocities_old[0, :])/self.time_step).reshape(1, self.number_of_robots)  # 1 x N
        angular_accelerations = ((self.velocities[1, :] - self.velocities_old[1, :])/self.time_step).reshape(1, self.number_of_robots)  # 1 x N
        omega_z = self.velocities[1, :].reshape(1, self.number_of_robots)  # 1 x N

        translational_acceleration = np.vstack((
            linear_accelerations,
            np.zeros((1, self.number_of_robots)),
            np.zeros((1, self.number_of_robots))
        ))

        tangential_acceleration = np.vstack((
            -angular_accelerations * self.imu_orientation[1],
            angular_accelerations * self.imu_orientation[0],
            np.zeros((1, self.number_of_robots))
        ))

        centripetal_acceleration = np.vstack((
            -omega_z**2 * self.imu_orientation[0],
            -omega_z**2 * self.imu_orientation[1],
            np.zeros((1, self.number_of_robots))
        ))

        gravity = np.vstack((
            np.zeros((1, self.number_of_robots)),
            np.zeros((1, self.number_of_robots)),
            -9.81*np.ones((1, self.number_of_robots))
        ))

        imu_accelerations = translational_acceleration + tangential_acceleration + centripetal_acceleration + gravity

        # Apply noise to accelerations
        x_noise = np.random.normal(0, self.accelerometer_noise_stds[0], self.number_of_robots)
        y_noise = np.random.normal(0, self.accelerometer_noise_stds[1], self.number_of_robots)
        z_noise = np.random.normal(0, self.accelerometer_noise_stds[2], self.number_of_robots)

        self.accelerations = np.vstack((
            imu_accelerations[0, :] + x_noise,
            imu_accelerations[1, :] + y_noise,
            imu_accelerations[2, :] + z_noise
        ))

    def _simulate_gyros(self):
        """
        Simulates the gyrometer readings for the robot based on their current angular velocities.
        The gyros measure the angular velocity around the Z-axis (yaw rate) with some added noise.
        """
        self.gyros = np.vstack((
            np.random.normal(0, self.gyro_noise_stds[0], self.number_of_robots),  # X-axis gyro noise
            np.random.normal(0, self.gyro_noise_stds[1], self.number_of_robots),  # Y-axis gyro noise
            self.velocities[1, :] + np.random.normal(0, self.gyro_noise_stds[2], self.number_of_robots)   # Z-axis gyro measurement with noise
        ))

    def _simulate_magnetometers(self):
        """
        Simulates the magnetometer readings for each robot based on their orientation in the testbed.

        Note: in the testbed, magnetic fields are corrected such that magnetic north is the x-axis of the testbed.
        """
        return np.vstack((
            self.magnetometer_xy_avg * np.cos(self.poses[2, :]) + np.random.normal(0, self.magnetometer_noise_stds[0], self.number_of_robots),   # X-axis
            -self.magnetometer_xy_avg * np.sin(self.poses[2, :]) + np.random.normal(0, self.magnetometer_noise_stds[1], self.number_of_robots),   # Y-axis
            self.magnetometer_z_avg  + np.random.normal(0, self.magnetometer_noise_stds[2], self.number_of_robots)                               # Z-axis
        ))

    def _simulate_orientation(self):
        """
        Simulates the orientation readings for each robot based on their current pose and some added noise.

        Note: On the robotarium, a sensor fusion algorithm is run on the IMU data to produce a stable orientation estimate.
        """
        orientation_degrees = np.rad2deg(self.poses[2, :])
        self.orientations = (orientation_degrees + np.random.normal(0, self.orientation_noise_std, self.number_of_robots)) % 360

