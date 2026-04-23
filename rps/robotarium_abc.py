import math
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Import your custom GTERNAL patch drawing classes from patch_creation
from rps.patch_creation.gternal_patch import gternal_patch, GTERNALRobotPatch

class ARobotarium(ABC):
    """
    Abstract base class for the Robotarium simulator.
    """

    # ------------------------------------------------------------------
    # Physical constants (Matching ARobotarium.m)
    # ------------------------------------------------------------------
    TIME_STEP                   = 0.033          # seconds
    MAX_LINEAR_VELOCITY         = 0.2            # m/s
    ROBOT_DIAMETER              = 0.11           # m
    WHEEL_RADIUS                = 0.016          # m
    BASE_LENGTH                 = 0.11           # m
    COLLISION_DIAMETER          = 0.135          # m
    COLLISION_OFFSET            = 0.025          # m
    BOUNDARIES                  = np.array([-1.6, 1.6, -1.0, 1.0])  # m [x_min, x_max, y_min, y_max]
    CENTER_OFFSET                = 0.01           # m (Offset forwards where a robot's collider is placed)
    
    DISTANCE_SENSOR_ERROR        = 0.03          # fractional (3% error)
    DISTANCE_SENSOR_DROPOUT_PROB = 0.035         # probability [0, 1]
    DISTANCE_SENSOR_OUTLIER_PROB = 0.014         # probability [0, 1]
    DISTANCE_SENSOR_RANGE        = 1.2           # m
    
    DISTANCE_SENSORS_ORIENTATION = np.array([
        [-0.04,  0.0,   0.04,  0.05,  0.04,   0.0,  -0.04],
        [ 0.04,  0.06,  0.05,  0.0,  -0.05, -0.06, -0.04],
        [ math.pi,  math.pi / 2,  math.pi / 4,  0.0, -math.pi / 4, -math.pi / 2, -math.pi],
    ])

    ENCODER_COUNTS_PER_REVOLUTION = 28           # counts/revolution
    MOTOR_GEAR_RATIO              = 100.37
    IMU_ORIENTATION               = np.array([0.0594 - 0.00319, 0.0344628 - 0.0475, 0.0])
    # [x, y offset from axle center] and rad (heading of IMU in robot frame)

    ENCODER_NOISE_STD         = 0.25                                     # counts
    ACCELEROMETER_NOISE_STDS  = np.array([0.012929, 0.012127, 0.052979]) # m/s² [x; y; z]
    GYRO_NOISE_STDS           = np.array([0.001663, 0.001216, 0.002372]) # rad/s [x; y; z]
    MAGNETOMETER_NOISE_STDS   = np.array([3.275843, 2.365798, 5.232685]) # µT [x; y; z]
    MAGNETOMETER_Z_AVG        = -39.8594                                 # µT
    MAGNETOMETER_XY_AVG       = 6.0712                                   # µT
    ORIENTATION_NOISE_STD     = 0.310830                                 # degrees

    MAX_WHEEL_VELOCITY   = MAX_LINEAR_VELOCITY / WHEEL_RADIUS
    MAX_ANGULAR_VELOCITY = (
        2.0 * (WHEEL_RADIUS / ROBOT_DIAMETER)
        * (MAX_LINEAR_VELOCITY / WHEEL_RADIUS)
    )

    def __init__(self, **kwargs):
        number_of_robots = kwargs.get("number_of_robots", 0)
        show_figure = kwargs.get("show_figure", False)
        initial_conditions = kwargs.get("initial_conditions", np.array([]))
        use_distance_sensors = kwargs.get("use_distance_sensors", False)
        obstacles = kwargs.get("obstacles", None)
        self.show_obstacles = kwargs.get("show_obstacles", True)
        self.show_arena_boundaries = kwargs.get("show_arena_boundaries", True)
        self.show_robot_patches = kwargs.get("show_robot_patches", True)
        self.show_distance_endpoints = kwargs.get("show_distance_endpoints", True)
        self.show_distance_rays = kwargs.get("show_distance_rays", False)

        assert 0 <= number_of_robots <= 50, "Number of robots must be >= 0 and <= 50"

        N = number_of_robots
        self.number_of_robots = N
        self.show_figure = show_figure

        # State Variables
        self._velocities      = np.zeros((2, N))
        self._velocities_old  = np.zeros((2, N))
        
        # Apply initial conditions immediately so the first draw isn't stacked at (0,0)
        if initial_conditions.size > 0:
            self._poses = initial_conditions.copy()
        else:
            self._poses = np.zeros((3, N))
            
        self._distances       = np.full((7, N), np.nan)   # NaN until first sensor sim, like MATLAB
        self._accelerations   = np.zeros((3, N))
        self._orientations    = np.zeros(N)
        self._magnetic_fields = np.zeros((3, N))
        self._gyros           = np.zeros((3, N))
        self._initial_encoders = np.zeros((2, N))
        self._encoders        = np.zeros((2, N))
        self._leds            = np.zeros((3, N))

        self._distance_sensors_enabled = use_distance_sensors
        if self._distance_sensors_enabled:
            self._distance_endpoints = np.full((2, 7, N), np.nan)
        self._cm = plt.get_cmap("tab20")

        self.obstacles = obstacles
        
        if self.show_figure:
            self._initialize_visualization()
        self.initializing = True

    @abstractmethod
    def get_poses(self) -> NDArray[np.floating]:
        pass

    @abstractmethod
    def initialize(self, initial_conditions: NDArray[np.floating]):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def debug(self):
        pass

    def set_velocities(self, ids: NDArray[np.integer], velocities: NDArray[np.floating]):
        assert velocities.shape[1] <= self.number_of_robots, (
            f"Column count of velocities ({velocities.shape[1]}) must be <= number_of_robots ({self.number_of_robots})"
        )
        self._velocities[:, ids] = velocities

    def robot_colormap(self) -> plt.cm.ScalarMappable:
        return self._cm

    def get_distances(self) -> NDArray[np.floating]:
        return self._distances.copy()

    def get_distance_endpoints(self) -> NDArray[np.floating]:
        return self._distance_endpoints.copy()

    def get_accelerations(self) -> NDArray[np.floating]:
        return self._accelerations.copy()

    def get_orientations(self) -> NDArray[np.floating]:
        return self._orientations.copy()

    def get_magnetic_fields(self) -> NDArray[np.floating]:
        return self._magnetic_fields.copy()

    def get_gyros(self) -> NDArray[np.floating]:
        return self._gyros.copy()

    def get_encoders(self) -> NDArray[np.integer]:
        return np.int32(self._encoders) - np.int32(self._initial_encoders)

    def _threshold(self, dxu: NDArray[np.floating]) -> NDArray[np.floating]:
        dxdd = self._uni_to_diff(dxu)
        to_thresh = np.abs(dxdd) > self.MAX_WHEEL_VELOCITY
        dxdd[to_thresh] = self.MAX_WHEEL_VELOCITY * np.sign(dxdd[to_thresh])
        return self._diff_to_uni(dxdd)

    def _uni_to_diff(self, dxu: NDArray[np.floating]) -> NDArray[np.floating]:
        r = self.WHEEL_RADIUS
        l = self.BASE_LENGTH
        return np.vstack([
            (1 / (2 * r)) * (2 * dxu[0, :] - l * dxu[1, :]),
            (1 / (2 * r)) * (2 * dxu[0, :] + l * dxu[1, :]),
        ])

    def _diff_to_uni(self, dxdd: NDArray[np.floating]) -> NDArray[np.floating]:
        r = self.WHEEL_RADIUS
        l = self.BASE_LENGTH
        return np.vstack([
            (r / 2) * (dxdd[0, :] + dxdd[1, :]),
            (r / l) * (dxdd[1, :] - dxdd[0, :]),
        ])

    def _validate(self, errors: dict) -> dict:
        p = self._poses
        b = self.BOUNDARIES
        N = self.number_of_robots

        if not self.initializing:
            # 1. Boundary check
            x, y = p[0, :], p[1, :]
            if np.any((x < b[0]) | (x > b[1]) | (y < b[2]) | (y > b[3])):
                errors['robots_outside_boundaries'] = errors.get('robots_outside_boundaries', 0) + 1

            # 2. Collision check — offset each robot's centre forward along its heading
            offsets = self.COLLISION_OFFSET * np.vstack([np.cos(p[2, :]), np.sin(p[2, :])])  # 2 x N
            centres = p[:2, :] + offsets  # 2 x N
            for i in range(N - 1):
                for j in range(i + 1, N):
                    if np.linalg.norm(centres[:, i] - centres[:, j]) <= self.COLLISION_DIAMETER:
                        errors['robots_too_close'] = errors.get('robots_too_close', 0) + 1

            # 3. Actuator limit check
            dxdd = self._uni_to_diff(self._velocities)
            if np.any(np.abs(dxdd) > self.MAX_WHEEL_VELOCITY):
                errors['exceeded_actuator_limits'] = errors.get('exceeded_actuator_limits', 0) + 1

        return errors

    # =======================================================================
    # VISUALIZATION METHODS
    # =======================================================================

    def _initialize_visualization(self) -> None:
        view_padding = 0.05
        
        # Calculate the total span including padding
        x_span = (self.BOUNDARIES[1] - self.BOUNDARIES[0]) + 2 * view_padding
        y_span = (self.BOUNDARIES[3] - self.BOUNDARIES[2]) + 2 * view_padding
        
        # Set figsize to match the physical aspect ratio (scaled up for screen visibility)
        scale = 2.5
        self._fig, self._axes_handle = plt.subplots(figsize=(x_span * scale, y_span * scale))
        ax = self._axes_handle
        
        ax.set_aspect('equal')
        ax.set_xlim([self.BOUNDARIES[0] - view_padding, self.BOUNDARIES[1] + view_padding])
        ax.set_ylim([self.BOUNDARIES[2] - view_padding, self.BOUNDARIES[3] + view_padding])
        ax.axis('off')

        # Draw the arena boundaries
        if self.show_arena_boundaries:
            rect = patches.Rectangle(
                (self.BOUNDARIES[0], self.BOUNDARIES[2]),
                self.BOUNDARIES[1] - self.BOUNDARIES[0],
                self.BOUNDARIES[3] - self.BOUNDARIES[2],
                linewidth=2, edgecolor='k', facecolor='none'
            )
            ax.add_patch(rect)

        # Get the standard geometry and color data for a single robot
        shared_patch_data = gternal_patch()

        # Build one GTERNALRobotPatch per robot
        self._robot_handle = []
        if self.show_robot_patches:
            for i in range(self.number_of_robots):
                robot_patch = GTERNALRobotPatch(
                    axes=ax,
                    pose=self._poses[:, i],
                    patch_data=shared_patch_data,
                )
                self._robot_handle.append(robot_patch)

        # Distance sensor scatter plot setup
        if self._distance_sensors_enabled and self.show_distance_endpoints:
            N = self.number_of_robots
            self._distance_endpoint_patches = []
            for i in range(N):
                patch = ax.scatter(
                    np.full(7, np.nan), np.full(7, np.nan),
                    s=50, color=self._cm(i % 20), zorder=3, label=f'Robot {i} Sensors'
                )
                self._distance_endpoint_patches.append(patch)

        # Distance rays
        if self._distance_sensors_enabled and self.show_distance_rays:
            self._distance_ray_lines = []
            for i in range(self.number_of_robots):
                ray_lines = []
                for j in range(7):
                    line = Line2D(
                        [self._poses[0, i], self._poses[0, i]],  # Start and end x (updated in _draw_distance_endpoints)
                        [self._poses[1, i], self._poses[1, i]],  # Start and end y (updated in _draw_distance_endpoints)
                        linewidth=0.5, color=self._cm(i % 20), zorder=2
                    )
                    ax.add_line(line)
                    ray_lines.append(line)
                self._distance_ray_lines.append(ray_lines)

        if self.show_obstacles and self.obstacles is not None:
            for i in range(self.obstacles.shape[0]):
                obs_line = Line2D(
                    [self.obstacles[i, 0, 0], self.obstacles[i, 1, 0]],
                    [self.obstacles[i, 0, 1], self.obstacles[i, 1, 1]],
                    linewidth=4, color="0.5",
                )
                ax.add_line(obs_line)

        plt.ion()
        plt.show()
        
        # Keep margins tight to the new perfectly-proportioned figure window
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    def _draw_robots(self) -> None:
        """Update every robot patch to the current poses and LED colours."""
        if self.show_robot_patches:
            for i in range(self.number_of_robots):
                self._robot_handle[i].set_pose(self._poses[:, i])
                self._robot_handle[i].set_led(self._leds[:, i])
                self._robot_handle[i].set_zorder(2)
                
        # Force Matplotlib to flush GUI events and redraw the canvas
        if self.show_figure:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def _calculate_endpoints(self, distances: NDArray[np.floating]) -> NDArray[np.floating]:
        endpoints = np.full((2, 7, self.number_of_robots), np.nan)
        for i in range(self.number_of_robots):
            R = np.array([[np.cos(self._poses[2, i]), -np.sin(self._poses[2, i])],
                          [np.sin(self._poses[2, i]),  np.cos(self._poses[2, i])]])
            for j in range(7):
                if distances[j, i] >= 0:
                    start_point = R @ self.DISTANCE_SENSORS_ORIENTATION[:2, j] + self._poses[:2, i]
                    end_point = start_point + distances[j, i] * np.array([np.cos(self._poses[2, i] + self.DISTANCE_SENSORS_ORIENTATION[2, j]), np.sin(self._poses[2, i] + self.DISTANCE_SENSORS_ORIENTATION[2, j])])
                    endpoints[:, j, i] = end_point
        return endpoints
    
    def _draw_distance_rays(self, endpoints: NDArray[np.floating]):
        """
        Update the distance sensor ray Line2D objects based on the latest sensor endpoint coordinates

        Inputs:
        - endpoints: (2, 7, N) array of distance sensor endpoint coordinates in metres; NaN where no detection 
        """
        for i in range(self.number_of_robots):
                R = np.array([[np.cos(self._poses[2, i]), -np.sin(self._poses[2, i])],
                            [np.sin(self._poses[2, i]),  np.cos(self._poses[2, i])]])
                for j in range(7):
                    start_point = R @ self.DISTANCE_SENSORS_ORIENTATION[:2, j] + self._poses[:2, i]
                    self._distance_ray_lines[i][j].set_data(
                        [start_point[0], endpoints[0, j, i]],
                        [start_point[1], endpoints[1, j, i]]
                    )

    def _draw_distance_endpoints(self, endpoints: NDArray[np.floating]):
        """
        Update the distance sensor endpoint scatter plots based on the latest sensor readings

        Inputs:
        - endpoints: (2, 7, N) array of distance sensor endpoint coordinates in metres; NaN where no detection 
        """
        for i in range(self.number_of_robots):
            self._distance_endpoint_patches[i].set_offsets(endpoints[:, :, i].T)

