import numpy as np

class UnicycleEKF:
    """Implemntation of EKF for a unicycle robot.

    This implementation is made for unicylce robot.  Not robotarium specific, can adjust base length and wheel radius.

    Parameters
    ----------
    initial_state : np.ndarray, shape (3, 1) = initial state of the robot (x, y, theta)
    initial_covariance : np.ndarray, shape (3, 3) = initial covariance of the robot
    b : float = wheel base length (not half, the full length)
    r : float = wheel radius
    M : np.ndarray, shape (2, 2) = measurement noise covariance for for the wheel encoder angular velocities, [[right_noise, 0], [0, left_noise]]
    Q : np.ndarray, shape (3, 3) = process noise covariance for the unicycle model
    R_gps : np.ndarray, shape (2, 2) or None = measurement noise covariance for the GPS (x and y are measured), None if not using GPS measurements
    R_imu : np.ndarray, shape (3, 3) or None = measurement noise covariance for the IMU (acc_x, acc_y, and yaw rate are measured), None if not using IMU measurements

    Attributes
    ----------
    state : np.ndarray, shape (3, 1)
        The current state estimate.
    covariance : np.ndarray, shape (3, 3)
        The state covariance matrix.
    """
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, b: float, r: float, M: np.ndarray, Q: np.ndarray, R_gps: np.ndarray = None, R_imu: np.ndarray = None):
        self.state = initial_state
        self.covariance = initial_covariance
        self.base_length = b
        self.wheel_radius = r
        self.wheel_encoder_noise_covariance_matrix = M
        self.Q = Q # process noise covariance for the unicycle mode
        self.R_gps = R_gps # measurement noise covariance for the GPS (2x2 matrix b/c only x and y are measured)
        self.R_imu = R_imu # measurement noise covariance matrix for the IMU (3x3 matrix b/c acc_x, acc_y, and yaw rate are measured)
        self.P = initial_covariance


    def predict(self, v, w, dt):
        F = self.form_F(v, dt)
        G = self.form_G(dt) # this is really our nonlinear state transition function f_x(x, u, dt) evaluated at the current state and control input, but it's really only G, but in the case of the unicycle model, it's only G
        W = self.form_wheel_encoder_process_noise_matrix(dt)
        Q = self.form_Q(dt)
        self.P = F @ self.P @ F.T + W + Q # update covariance using the linearized state transition function F, encoder noise matrix W, and process noise covariance Q
        self.state = self.state + G @ np.array([v, w]) # update state using discretized nonlinear state transition function f_x(x, u, dt)
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi # wrap theta to [-pi, pi]
    
    def update_gps(self, gps_measurement: np.ndarray):
        """ Update the state and covariance using the GPS measurement of the robot's position (x, y).  Assumes GPS is at the robot's base link. """
        if self.R_gps is None:
            raise ValueError("GPS measurement noise covariance matrix is not set")
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = self.R_gps
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        self.P = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ R @ K.T
        self.state = self.state + K @ (gps_measurement - H @ self.state)

    def update_imu(self, imu_measurement: np.ndarray):
        if self.R_imu is None:
            raise ValueError("IMU measurement noise covariance matrix is not set")
        raise NotImplementedError("IMU update not implemented yet")

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
