import numpy as np
from numpy.typing import NDArray

def simulate_distance_sensors_without_robots(
    poses: NDArray[np.floating],
    obstacles: NDArray[np.floating],
    distance_sensors_orientation: NDArray[np.floating],
    distance_sensor_range: float,
    distance_sensor_error: float,
    distance_sensor_dropout_prob: float,
    distance_sensor_outlier_prob: float,
) -> NDArray[np.floating]:
    """
    Simulate the 7-sensor distance measurements for all robots ignoring the other
    robots as obstacles.

    Parameters
    ----------
    poses : (3, N)
        Current robot poses [x; y; theta].
    obstacles : (M, 2, 2) or None
        Line-segment obstacles.  Each obstacle k is defined by
        ``obstacles[k, 0, :]`` (start) and ``obstacles[k, 1, :]`` (end).
    distance_sensors_orientation : (3, 7)
        Body-frame position [x; y] and heading [theta] of each sensor.
    robot_diameter : float
        Collision diameter used for robot-robot ray intersection.
    robot_center_offset: float
        Forward offset of the robot center from the geometric center (m).
    distance_sensor_range : float
        Maximum sensor range (m).
    distance_sensor_error : float
        Fractional Gaussian noise std (sigma = error * true_distance).
    distance_sensor_dropout_prob : float
        Probability a valid reading is dropped (returns -1).
    distance_sensor_outlier_prob : float
        Probability a valid reading is replaced by a uniform random value.

    Returns
    -------
    distances : (7, N)
        Sensor readings in metres; -1 where no detection.
    """
    N = poses.shape[1]
    distances = np.zeros((distance_sensors_orientation.shape[1], N))
    for i in range(N):
        R = np.array([[np.cos(poses[2, i]), -np.sin(poses[2, i])],
                      [np.sin(poses[2, i]),  np.cos(poses[2, i])]])
        for j in range(distance_sensors_orientation.shape[1]):
            start_point = R @ distance_sensors_orientation[:2, j] + poses[:2, i]
            end_point = start_point + distance_sensor_range * \
                np.array([np.cos(poses[2, i] + distance_sensors_orientation[2, j]), np.sin(poses[2, i] + distance_sensors_orientation[2, j])])
            ray_vector = end_point - start_point

            min_distance = distance_sensor_range

            # Check intersection with obstacles
            if obstacles is not None and obstacles.size > 0:
                for k in range(obstacles.shape[0]):
                    obs_start = obstacles[k, 0, :]
                    obs_end = obstacles[k, 1, :]
                    obs_vector = obs_end - obs_start

                    # Compute intersection using cross product
                    r_cross_s = ray_vector[0] * obs_vector[1] - ray_vector[1] * obs_vector[0]
                    if abs(r_cross_s) < 1e-10:
                        continue  # Parallel lines

                    t = ((obs_start - start_point)[0] * obs_vector[1] - (obs_start - start_point)[1] * obs_vector[0]) / r_cross_s
                    u = ((obs_start - start_point)[0] * ray_vector[1] - (obs_start - start_point)[1] * ray_vector[0]) / r_cross_s

                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection_point = start_point + t * ray_vector
                        distance = np.linalg.norm(intersection_point - start_point)
                        min_distance = min(min_distance, distance)

            # Apply sensor noise and dropout/outlier model
            if min_distance < distance_sensor_range:
                # Mixture model for noise, outliers, and dropouts
                if np.random.rand() < distance_sensor_dropout_prob:
                    distances[j, i] = -1  # Dropout
                elif np.random.rand() < distance_sensor_outlier_prob:
                    distances[j, i] = np.random.uniform(0, distance_sensor_range)  # Outlier
                else:
                    noise = distance_sensor_error * min_distance * np.random.randn()
                    distances[j, i] = np.clip(min_distance + noise, 0, distance_sensor_range)  # Noisy reading
            else:
                distances[j, i] = -1  # No detection
    return distances


def simulate_distance_sensors(
    poses: NDArray[np.floating],
    obstacles: NDArray[np.floating],
    distance_sensors_orientation: NDArray[np.floating],
    robot_diameter: float,
    robot_center_offset: float,
    distance_sensor_range: float,
    distance_sensor_error: float,
    distance_sensor_dropout_prob: float,
    distance_sensor_outlier_prob: float,
) -> NDArray[np.floating]:
    """
    Simulate the 7-sensor distance measurements for all robots.

    Parameters
    ----------
    poses : (3, N)
        Current robot poses [x; y; theta].
    obstacles : (M, 2, 2) or None
        Line-segment obstacles.  Each obstacle k is defined by
        ``obstacles[k, 0, :]`` (start) and ``obstacles[k, 1, :]`` (end).
    distance_sensors_orientation : (3, 7)
        Body-frame position [x; y] and heading [theta] of each sensor.
    robot_diameter : float
        Collision diameter used for robot-robot ray intersection.
    robot_center_offset: float
        Forward offset of the robot center from the geometric center (m).
    distance_sensor_range : float
        Maximum sensor range (m).
    distance_sensor_error : float
        Fractional Gaussian noise std (sigma = error * true_distance).
    distance_sensor_dropout_prob : float
        Probability a valid reading is dropped (returns -1).
    distance_sensor_outlier_prob : float
        Probability a valid reading is replaced by a uniform random value.

    Returns
    -------
    distances : (7, N)
        Sensor readings in metres; -1 where no detection.
    """
    N = poses.shape[1]
    offsets = robot_center_offset * np.array([np.cos(poses[2, :]), np.sin(poses[2, :])])
    poses = poses.copy()
    poses[:2, :] += offsets
    distances = np.zeros((distance_sensors_orientation.shape[1], N))
    for i in range(N):
        R = np.array([[np.cos(poses[2, i]), -np.sin(poses[2, i])],
                      [np.sin(poses[2, i]),  np.cos(poses[2, i])]])
        for j in range(distance_sensors_orientation.shape[1]):
            start_point = R @ distance_sensors_orientation[:2, j] + poses[:2, i]
            end_point = start_point + distance_sensor_range * \
                np.array([np.cos(poses[2, i] + distance_sensors_orientation[2, j]), np.sin(poses[2, i] + distance_sensors_orientation[2, j])])
            ray_vector = end_point - start_point
            ray_len = np.linalg.norm(ray_vector)
            ray_unit = ray_vector / ray_len

            min_distance = distance_sensor_range

            # Check intersection with obstacles
            if obstacles is not None and obstacles.size > 0:
                for k in range(obstacles.shape[0]):
                    obs_start = obstacles[k, 0, :]
                    obs_end = obstacles[k, 1, :]
                    obs_vector = obs_end - obs_start

                    # Compute intersection using cross product
                    r_cross_s = ray_vector[0] * obs_vector[1] - ray_vector[1] * obs_vector[0]
                    if abs(r_cross_s) < 1e-10:
                        continue  # Parallel lines

                    t = ((obs_start - start_point)[0] * obs_vector[1] - (obs_start - start_point)[1] * obs_vector[0]) / r_cross_s
                    u = ((obs_start - start_point)[0] * ray_vector[1] - (obs_start - start_point)[1] * ray_vector[0]) / r_cross_s

                    if 0 <= t <= 1 and 0 <= u <= 1:
                        intersection_point = start_point + t * ray_vector
                        distance = np.linalg.norm(intersection_point - start_point)
                        min_distance = min(min_distance, distance)

            # Check intersection with other robots (treated as circles)
            for m in range(N):
                if m == i:
                    continue

                other_robot_center = poses[:2, m]
                to_other_robot = other_robot_center - start_point

                proj_length = np.dot(to_other_robot, ray_unit)

                if 0 <= proj_length <= ray_len:
                    closest_point = start_point + proj_length * ray_unit
                    perp_dist = np.linalg.norm(closest_point - other_robot_center)
                    
                    if perp_dist <= robot_diameter / 2:
                        half_chord = np.sqrt((robot_diameter / 2) ** 2 - perp_dist ** 2)
                        distance = proj_length - half_chord

                        if distance >= 0:
                            min_distance = min(min_distance, distance)

            # Apply sensor noise and dropout/outlier model
            if min_distance < distance_sensor_range:
                # Mixture model for noise, outliers, and dropouts
                if np.random.rand() < distance_sensor_dropout_prob:
                    distances[j, i] = -1  # Dropout
                elif np.random.rand() < distance_sensor_outlier_prob:
                    distances[j, i] = np.random.uniform(0, distance_sensor_range)  # Outlier
                else:
                    noise = distance_sensor_error * min_distance * np.random.randn()
                    distances[j, i] = np.clip(min_distance + noise, 0, distance_sensor_range)  # Noisy reading
            else:
                distances[j, i] = -1  # No detection
    return distances
