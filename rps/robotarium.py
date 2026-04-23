import time
import numpy as np
from numpy.typing import NDArray

from rps.robotarium_abc import ARobotarium
from rps.utilities.misc import generate_random_poses, generate_initial_poses, create_at_pose, determine_font_size
from rps.utilities.barrier_certificates import create_uni_barrier_certificate
from rps.utilities.controllers import create_pose_controller_hybrid
from rps.utilities.sensors import simulate_distance_sensors

class Robotarium(ARobotarium):
    """
    Robotarium simulator.

    Parameters
    ----------
    number_of_robots : int
        Positive integer number of robots (1 - 50).
    show_figure : bool, optional
        Render a graphical visualisation (default ``True``).
    initial_conditions : np.ndarray, optional
        3 x N array of initial ``[x; y; theta]`` poses.  If omitted or
        empty, collision-free initial poses are generated automatically.
    use_distance_sensors : bool, optional
        Enable simulation of 7 range sensors per robot (default ``False``).
    obstacles : np.ndarray or None, optional
        M x 2 x 2 array of line-segment obstacles, each defined by start
        (``[:, 0]``) and end (``[:, 1]``) endpoints in world coordinates.
    sim_in_real_time : bool, optional
        Throttle each step call to approximately TIME_STEP seconds 
        of wall-clock time (default ``True``).
    skip_initialization: bool, optional
        If True, the simulator will skip the initialization phase where
        robots drive onto the testbed and directly set the initial conditions.
        This will be set to False by default and the verifier will always
        use the initialization phase, but setting this to True may be useful
        when debugging one's algorithm (default ``False``)
    show_arena_boundaries: bool, optional
        Whether to render the arena boundaries (default ``True``)
    show_robot_patches: bool, optional
        Whether to render the robot patches (default ``True``)
    show_distance_endpoints: bool, optional
        Whether to render the distance sensor endpoints (default ``True``)
    show_distance_rays: bool, optional
        Whether to render the distance sensor rays (default ``False``)
    show_obstacles: bool, optional
        Whether to render the obstacles (default ``True``)
    """

    def __init__(self, **kwargs):
        
        # 1.**kwargs parsing
        number_of_robots = kwargs.get('number_of_robots', -1)
        show_figure = kwargs.get('show_figure', False)
        initial_conditions = kwargs.get('initial_conditions', np.array([]))
        use_distance_sensors = kwargs.get('use_distance_sensors', False)
        obstacles = kwargs.get('obstacles', None)
        sim_in_real_time = kwargs.get('sim_in_real_time', True)
        skip_initialization = kwargs.get('skip_initialization', False)
        show_arena_boundaries = kwargs.get("show_arena_boundaries", True)
        show_robot_patches = kwargs.get("show_robot_patches", True)
        show_distance_endpoints = kwargs.get("show_distance_endpoints", True)
        show_distance_rays = kwargs.get("show_distance_rays", False)
        show_obstacles = kwargs.get("show_obstacles", True)

        # 2. Strict Type Checking
        assert isinstance(number_of_robots, int), f"NumberOfRobots must be an integer. Received type {type(number_of_robots).__name__}."
        assert isinstance(show_figure, bool), f"ShowFigure must be a boolean. Received type {type(show_figure).__name__}."
        assert isinstance(sim_in_real_time, bool), f"SimInRealTime must be a boolean. Received type {type(sim_in_real_time).__name__}."
        assert isinstance(use_distance_sensors, bool), f"UseDistanceSensors must be a boolean. Received type {type(use_distance_sensors).__name__}."
        assert isinstance(initial_conditions, np.ndarray), f"InitialConditions must be a numpy ndarray. Received type {type(initial_conditions).__name__}."

        initial_poses = generate_initial_poses(
            number_of_robots,
            width=ARobotarium.BOUNDARIES[1] - ARobotarium.BOUNDARIES[0] - ARobotarium.ROBOT_DIAMETER,
            height=ARobotarium.BOUNDARIES[3] - ARobotarium.BOUNDARIES[2] - ARobotarium.ROBOT_DIAMETER,
        )

        if initial_conditions.size == 0:
            initial_conditions = generate_random_poses(
                number_of_robots,
                spacing=0.5,
                width=2.75,
                height=1.75,
            )

        super().__init__(
            number_of_robots=number_of_robots,
            show_figure=show_figure,
            initial_conditions=initial_poses,
            use_distance_sensors=use_distance_sensors,
            obstacles=obstacles,
            show_arena_boundaries=show_arena_boundaries,
            show_robot_patches=show_robot_patches,
            show_distance_endpoints=show_distance_endpoints,
            show_distance_rays=show_distance_rays,
            show_obstacles=show_obstacles
        )

        self._sim_in_real_time = sim_in_real_time
        self._previous_render_time = time.time()
        self._called_step_already   = True
        self._checked_poses_already = False
        self._errors    = {}
        self._iteration = 0

        if skip_initialization:
            self._poses = initial_conditions.copy()
        else:
            self.initialize(initial_conditions)
        self.initializing = False

    def initialize(self, initial_conditions: NDArray[np.floating]) -> None:
        """
        Drive to the initial conditions

        Note: This is the algorithm we use to drive robots onto the testbed so if
        your experiment cannot consistently reach the initial conditions, it will likely
        be rejected during verification
        """
        # Initialize the barrier, controller, and at pose checker
        print(f"Initializing {self.number_of_robots} robots to initial conditions...")
        init_label = None
        if self.show_figure:
            font_size = determine_font_size(self, 0.1)
            init_label = self._axes_handle.text( 
                0,
                -0.5,
                "...INITIALIZING...",
                fontsize=font_size,
                color='b',
                fontweight='bold',
                horizontalalignment='center',
                verticalalignment='center',
                zorder=999,
            )

        barrier = create_uni_barrier_certificate()
        controller = create_pose_controller_hybrid(
            angular_velocity_limit=np.pi / 3.0,
            position_epsilon=0.03,
            position_error=0.05,
            rotation_error=0.1
        )
        at_pose = create_at_pose(position_error=0.05, rotation_error=0.2)

        # Drive to initial conditions
        x = self.get_poses()

        # When robot's deadlock due to the barriers we will give them an intermediate waypoint to drive towards
        # to break the deadlock
        step = 0
        deadlock_steps = 5 * 30
        deadlock_epsilon = 0.1
        timers = np.array([0] * self.number_of_robots)
        last_deadlock_poses = x.copy()
        waypointing = np.array([False] * self.number_of_robots)
        waypoints = generate_random_poses(self.number_of_robots, spacing=0.4, width=2.75, height=1.75)

        self.step()
        while not at_pose(x, initial_conditions)[0]:
            x = self.get_poses()
            dxu = controller(x, initial_conditions)
            
            initialized_ids = at_pose(x, initial_conditions)[1]
            uninit_ids = np.arange(0, self.number_of_robots)[~np.isin(np.arange(0, self.number_of_robots), initialized_ids)]
            for i in uninit_ids:
                if waypointing[i]:
                    dxu[:, i] = controller(x[:, i].reshape(-1, 1), waypoints[:, i].reshape(-1, 1)).squeeze(1)
                    if np.linalg.norm(last_deadlock_poses[:2, i] - x[:2, i]) > 0.1:
                        waypointing[i] = False
                    elif step - timers[i] > deadlock_steps:
                        waypoints[:, i] = np.array([
                            np.random.uniform(
                                low=min(x[0, i] + 0.25, self.BOUNDARIES[1]),
                                high=max(x[0, i] - 0.25, self.BOUNDARIES[0])
                            ),
                            np.random.uniform(
                                low=min(x[1, i] + 0.25, self.BOUNDARIES[3]),
                                high=max(x[1, i] - 0.25, self.BOUNDARIES[2])
                            ),
                            np.random.uniform(low=-np.pi, high=np.pi)
                        ])
                        timers[i] = step
                elif step - timers[i] > deadlock_steps and np.linalg.norm(last_deadlock_poses[:2, i] - x[:2, i]) < deadlock_epsilon:
                    waypointing[i] = True
                    waypoints[:, i] = np.array([
                        np.random.uniform(
                            low=min(x[0, i] + 0.25, self.BOUNDARIES[1]),
                            high=max(x[0, i] - 0.25, self.BOUNDARIES[0])
                        ),
                        np.random.uniform(
                            low=min(x[1, i] + 0.25, self.BOUNDARIES[3]),
                            high=max(x[1, i] - 0.25, self.BOUNDARIES[2])
                        ),
                        np.random.uniform(low=-np.pi, high=np.pi)
                    ])
                    timers[i] = step
                elif np.linalg.norm(last_deadlock_poses[:2, i] - x[:2, i]) > deadlock_epsilon or \
                     np.linalg.norm(controller(x[:, i].reshape(-1, 1), initial_conditions[:, i].reshape(-1, 1)).squeeze(1)) <= 0.1:
                    timers[i] = step
                    last_deadlock_poses[:, i] = x[:, i].copy()

            step += 1
            dxu_safe = barrier(dxu, x)
            self.set_velocities(np.arange(self.number_of_robots), dxu_safe)
            self.step()

        print("At initial conditions... Starting Experiment")
        if init_label is not None:
            init_label.remove()

    def get_poses(self) -> NDArray[np.floating]:
        assert not self._checked_poses_already, "Can only call get_poses() once per call of step()!"
        self._called_step_already  = False
        self._checked_poses_already = True
        return self._poses.copy()

    def step(self) -> None:
        assert not self._called_step_already, "Make sure you call get_poses before calling step()!"
        self._called_step_already   = True
        self._checked_poses_already = False

        self._errors = self._validate(self._errors)
        self._iteration += 1

        self._velocities = self._threshold(self._velocities)
        self._velocities_old = self._velocities.copy()

        # Unicycle dynamics integration
        i = np.arange(self.number_of_robots)
        temp = self.TIME_STEP * self._velocities[0, i]
        self._poses[0, i] += temp * np.cos(self._poses[2, i])
        self._poses[1, i] += temp * np.sin(self._poses[2, i])
        self._poses[2, i] += self.TIME_STEP * self._velocities[1, i]

        self._poses[2, i] = np.arctan2(np.sin(self._poses[2, i]), np.cos(self._poses[2, i]))

        # Distance Sensors
        if self._distance_sensors_enabled:
            self._distances = simulate_distance_sensors(
                poses=self._poses,
                obstacles=self.obstacles,
                distance_sensors_orientation=self.DISTANCE_SENSORS_ORIENTATION,
                robot_diameter=self.ROBOT_DIAMETER,
                robot_center_offset=self.CENTER_OFFSET,
                distance_sensor_range=self.DISTANCE_SENSOR_RANGE,
                distance_sensor_error=self.DISTANCE_SENSOR_ERROR,
                distance_sensor_dropout_prob=self.DISTANCE_SENSOR_DROPOUT_PROB,
                distance_sensor_outlier_prob=self.DISTANCE_SENSOR_OUTLIER_PROB,
            )
            self._distance_endpoints = self._calculate_endpoints(self._distances)

        self.simulate_encoder_readings()
        self.simulate_accelerations()
        self.simulate_gyros()
        self.simulate_magnetometers()
        self.simulate_orientations()

        # Update figure
        if self.show_figure:
            self._draw_robots()
            if not self.initializing and self._distance_sensors_enabled and self.show_distance_endpoints:
                self._draw_distance_endpoints(self._distance_endpoints)
            if not self.initializing and self._distance_sensors_enabled and self.show_distance_rays:
                self._draw_distance_rays(self._distance_endpoints)

        # Throttle the simulation to match real time using a busy-wait timer
        if self._sim_in_real_time:
            while time.time() - self._previous_render_time < self.TIME_STEP:
                pass
            self._previous_render_time = time.time()

    def simulate_encoder_readings(self) -> None:
        dxdd = self._uni_to_diff(self._velocities)
        # dxdd is wheel angular velocity (rad/s) from _uni_to_diff — matches MATLAB's motor_angular_velocity.
        # Formula: (counts/rev) * gear_ratio / (2*pi rad/rev) * (rad/s) * (s) = counts.
        # Do NOT divide by WHEEL_RADIUS — that would double-convert from angular to linear.
        delta = dxdd * self.TIME_STEP * self.ENCODER_COUNTS_PER_REVOLUTION * self.MOTOR_GEAR_RATIO / (2 * np.pi)
        # Per-count noise: std scales with sqrt(number of counts), floored at 1 count — matches MATLAB
        n_counts = np.maximum(np.abs(delta), 1.0)
        step_noise_std = self.ENCODER_NOISE_STD * np.sqrt(n_counts)
        delta += step_noise_std * np.random.randn(*delta.shape)
        self._encoders += np.round(delta)

    def simulate_accelerations(self) -> None:
        N = self.number_of_robots
        linear_acc = (self._velocities[0, :] - self._velocities_old[0, :]) / self.TIME_STEP
        angular_acc = (self._velocities[1, :] - self._velocities_old[1, :]) / self.TIME_STEP
        omega_z = self._velocities[1, :]

        translational = np.vstack([linear_acc, np.zeros(N), np.zeros(N)])
        tangential = np.vstack([
            -angular_acc * self.IMU_ORIENTATION[1],
             angular_acc * self.IMU_ORIENTATION[0],
             np.zeros(N)
        ])
        centripetal = np.vstack([
            -np.square(omega_z) * self.IMU_ORIENTATION[0],
            -np.square(omega_z) * self.IMU_ORIENTATION[1],
             np.zeros(N)
        ])
        gravity = np.vstack([np.zeros(N), np.zeros(N), -9.81 * np.ones(N)])

        imu_acc = translational + tangential + centripetal + gravity
        noise = self.ACCELEROMETER_NOISE_STDS.reshape(3, 1) * np.random.randn(3, N)
        self._accelerations = imu_acc + noise

    def simulate_gyros(self) -> None:
        N = self.number_of_robots
        omega_z = self._velocities[1, :]
        noise = self.GYRO_NOISE_STDS.reshape(3, 1) * np.random.randn(3, N)
        self._gyros = np.vstack([np.zeros(N), np.zeros(N), omega_z]) + noise

    def simulate_magnetometers(self) -> None:
        N = self.number_of_robots
        theta = self._poses[2, :]
        noise = self.MAGNETOMETER_NOISE_STDS.reshape(3, 1) * np.random.randn(3, N)
        self._magnetic_fields = np.vstack([
             self.MAGNETOMETER_XY_AVG * np.cos(theta),
            -self.MAGNETOMETER_XY_AVG * np.sin(theta),
             self.MAGNETOMETER_Z_AVG  * np.ones(N),
        ]) + noise

    def simulate_orientations(self) -> None:
        N = self.number_of_robots
        orientation_degrees = self._poses[2, :] * (180.0 / np.pi)
        noise = self.ORIENTATION_NOISE_STD * np.random.randn(N)
        self._orientations = np.mod(orientation_degrees + noise, 360.0)

    def debug(self) -> None:
        elapsed = self._iteration * self.TIME_STEP
        print(f"Your simulation took approximately {elapsed:.2f} real seconds to execute.\n")

        too_close    = self._errors.get('robots_too_close', 0)
        out_of_bounds = self._errors.get('robots_outside_boundaries', 0)
        actuator     = self._errors.get('exceeded_actuator_limits', 0)
        hard_errors  = too_close + out_of_bounds

        if hard_errors == 0 and actuator == 0:
            print("No errors or warnings in your simulation!  Your script will run on the Robotarium!\n")
        elif hard_errors == 0:
            print("No hard errors detected, your script will run on the Robotarium!\n")
            print(
                f"WARNING: {actuator} out of {self._iteration} "
                f"(~{100 * actuator / max(self._iteration, 1):.0f}%) timesteps had velocity "
                f"command(s) exceed actuator limits.\n"
                f"         A large percentage of exceeded actuator commands will cause noticeable\n"
                f"         differences between this simulation and the physical Robotarium."
            )
        else:
            print("Errors detected, your script will NOT run on the Robotarium!\n")
            print("Errors:")
            if too_close:
                print(f"\t Simulation had {too_close} time steps where robots were too close (potential collision).")
            if out_of_bounds:
                print(f"\t Simulation had {out_of_bounds} time steps where robots were outside boundaries.")
            print("\nPlease fix the noted errors in your simulation; otherwise, your experiment will be rejected.")
