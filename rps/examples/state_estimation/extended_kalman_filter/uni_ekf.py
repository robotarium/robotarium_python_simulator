import numpy as np
import time


def wrap_angle(angle):
    """Constrain angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


class UnicycleEKF:
    """Implemntation of EKF for the robotarium.

    This implementation is made for the robotarium.  It can be used for any unicycle robot, but the base length and wheel radius must be set.

    The class implements a prediction based on wheel encoder measurements.
    For measurements, it exposes methods to update using GPS, direct orientation measurments, and bearing/range measurements to beacons.

    Parameters
    ----------
    initial_state : np.ndarray, shape (3, 1) = initial state of the robot (x, y, theta)
    initial_covariance : np.ndarray, shape (3, 3) = initial covariance of the robot
    b : float = wheel base length (not half, the full length)
    r : float = wheel radius
    M : np.ndarray, shape (2, 2) = measurement noise covariance for for the wheel encoder angular velocities, [[right_noise, 0], [0, left_noise]]
    Q : np.ndarray, shape (3, 3) = process noise covariance for the unicycle model
    Attributes
    ----------
    state : np.ndarray, shape (3, 1)
        The current state estimate.
    covariance : np.ndarray, shape (3, 3)
        The state covariance matrix.
    """
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, b: float, r: float, M: np.ndarray, Q: np.ndarray):
        self.state = initial_state
        self.covariance = initial_covariance
        self.base_length = b
        self.wheel_radius = r
        self.wheel_encoder_noise_covariance_matrix = M
        self.Q = Q # process noise covariance for the unicycle mode
        
        self.P = initial_covariance

        self.last_gyro_update_time = time.time()
        self.previous_theta = self.state[2] # we need this for the IMU update

    def predict(self, v, w, dt):
        """ Predict the state and covariance using the unicycle model using wheel encoder measurements. """
        F = self.form_F(v, dt)
        G = self.form_G(dt) # this is really our nonlinear state transition function f_x(x, u, dt) evaluated at the current state and control input, but it's really only G, but in the case of the unicycle model, it's only G
        W = self.form_wheel_encoder_process_noise_matrix(dt)
        Q = self.form_Q(dt)
        self.P = F @ self.P @ F.T + W + Q # update covariance using the linearized state transition function F, encoder noise matrix W, and process noise covariance Q
        self.state = self.state + G @ np.array([v, w]) # update state using discretized nonlinear state transition function f_x(x, u, dt)
        self.state[2] = wrap_angle(self.state[2])
    
    def update_gps(self, gps_measurement: np.ndarray, R_gps: np.ndarray):
        """ Update the state and covariance using the GPS measurement of the robot's position (x, y).  Assumes GPS is at the robot's base link. """
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = R_gps
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ R @ K.T
        self.state = self.state + K @ (gps_measurement - H @ self.state)
    
    def update_range(self, beacon_position: np.ndarray, range_measurement: float, R_range: float):
        """EKF update for scalar range to a beacon at ``beacon_position`` (bx, by) in metres.

        ``R_range`` is the variance (m^2): a float, or any array-like reduced to one scalar.
        Do not pass ``np.array([[R]])`` into ``np.array([[R]])`` — that nests dimensions.
        """
        bx = float(np.asarray(beacon_position).reshape(-1)[0])
        by = float(np.asarray(beacon_position).reshape(-1)[1])
        x = float(self.state[0])
        y = float(self.state[1])
        dx = bx - x
        dy = by - y
        rho = float(np.hypot(dx, dy))
        rho = max(rho, 1e-6)
        H = np.array([[-dx / rho, -dy / rho, 0.0]], dtype=float)
        h_hat = rho
        R = np.array([[R_range]], dtype=float)
        S = H @ self.P @ H.T + R
        K = (self.P @ H.T @ np.linalg.inv(S)).reshape(3, 1)
        innovation = float(range_measurement) - h_hat
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ R @ K.T
        self.state = np.asarray(self.state, dtype=float).reshape(3) + (K * innovation).ravel()

    def update_brearing(self, beacon_position: np.ndarray, bearing_measurement: float, R_bearing: np.ndarray = None):
        """ Update the state and covariance using the bearing measurement to a beacon. """
        H = np.array([[beacon_position[0], beacon_position[1], 0]])
        R = R_bearing
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ R @ K.T
        innovation = wrap_angle(bearing_measurement - H @ self.state)
        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])

    def update_orientation(self, orientation: float, R_orientation: float = None):
        """ Update the state and covariance using direct measurement of the robot's orientation. """
        H = np.array([0, 0, 1]).reshape(1, 3)
        print(f"H shape: {H.shape}")
        R = R_orientation
        print(f"orientation shape: {np.array([orientation]).shape if not np.isscalar(orientation) else 'scalar'}")
   
        K = self.P @ H.T * (1 / ((H @ self.P @ H.T).item() + R))
        print(f"K shape: {K.shape}")
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K * R * K.T

        # Wrap orientation to [-pi, pi] after update
        orientation = wrap_angle(orientation)
        innovation = wrap_angle(orientation - H @ self.state)
        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])

    def update_gyro(self, yaw_rate: float, dt: float, R_gyro: np.ndarray = None):
        """ Update the state and covariance using the gyro measurement of the robot's yaw rate. """
        H = np.array([[0, 0, 1]])

        # form computed measurement vector (what we expect to see based on our state)
        yaw_diff = wrap_angle(self.state[2] - self.previous_theta)
        current_time = time.time()
        dt = current_time - self.last_gyro_update_time

        augmented_state = np.array([self.state[0], self.state[1], yaw_diff])
        augmented_measurement = wrap_angle(yaw_rate*dt)

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R_gyro)

        innovation = wrap_angle(np.array([augmented_measurement]) - H @ augmented_state)

        # update the state
        self.state = self.state + K @ innovation

        # update the covariance
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ R_gyro @ K.T
        self.previous_theta = self.state[2]
        self.last_gyro_update_time = current_time


    def form_fx(self, v, w, dt):
        """ For nonlinear state transition function f_x """
        return np.array([[dt*v*np.cos(self.state[2])],
                         [dt*v*np.sin(self.state[2])],
                         [dt*w]])
        

    def form_F(self, v, dt):
        """ For discretezed state transition matrix F """
        return np.array([[1, 0, -v*dt*np.sin(self.state[2])],
                         [0, 1, v*dt*np.cos(self.state[2])],
                         [0, 0, 1]])
    
    def form_G(self, dt):
        """ For discretezed process noise matrix G """
        return np.array([[np.cos(self.state[2])*dt, 0],
                         [np.sin(self.state[2])*dt, 0],
                         [0, dt]])
    
    def form_Q(self, dt):
        """ For discretezed process noise covariance Q """
        return self.Q*dt

    def form_wheel_encoder_process_noise_matrix(self, dt):
        """ For discretezed wheel encoder process noise matrix W """
        tmp = (self.wheel_radius/2) * dt
        tmp2 = (self.wheel_radius/self.base_length) * dt
        L = np.array([[tmp*np.cos(self.state[2]), tmp*np.cos(self.state[2])], [tmp*np.sin(self.state[2]), tmp*np.sin(self.state[2])], [tmp2, -tmp2]])
        return L @ self.wheel_encoder_noise_covariance_matrix @ L.T


