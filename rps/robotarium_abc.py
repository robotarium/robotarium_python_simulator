import time
import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import rps.utilities.misc as misc

# RobotariumABC: This is an interface for the Robotarium class that
# ensures the simulator and the robots match up properly.  

# THIS FILE SHOULD NEVER BE MODIFIED OR SUBMITTED!

class RobotariumABC(ABC):

    def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([]), use_distance_sensors=False, obstacles=None):

        #Check user input types
        assert isinstance(number_of_robots,int), "The number of robots used argument (number_of_robots) provided to create the Robotarium object must be an integer type. Recieved type %r." % type(number_of_robots).__name__
        assert isinstance(initial_conditions,np.ndarray), "The initial conditions array argument (initial_conditions) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(initial_conditions).__name__
        assert isinstance(show_figure,bool), "The display figure window argument (show_figure) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        assert isinstance(sim_in_real_time,bool), "The simulation running at 0.033s per loop (sim_real_time) provided to create the Robotarium object must be boolean type. Recieved type %r." % type(show_figure).__name__
        assert isinstance(use_distance_sensors,bool), "The use_distance_sensors argument provided to create the Robotarium object must be boolean type. Recieved type %r." % type(use_distance_sensors).__name__
        if obstacles is not None:
            assert isinstance(obstacles,np.ndarray), "The obstacles array argument (obstacles) provided to create the Robotarium object must be a numpy ndarray. Recieved type %r." % type(obstacles).__name__
        
        #Check user input ranges/sizes
        assert (number_of_robots >= 0 and number_of_robots <= 50), "Requested %r robots to be used when creating the Robotarium object. The deployed number of robots must be between 0 and 50." % number_of_robots 
        if (initial_conditions.size > 0):
            assert initial_conditions.shape == (3, number_of_robots), "Initial conditions provided when creating the Robotarium object must of size 3xN, where N is the number of robots used. Expected a 3 x %r array but recieved a %r x %r array." % (number_of_robots, initial_conditions.shape[0], initial_conditions.shape[1])


        self.number_of_robots = number_of_robots
        self.show_figure = show_figure
        self.initial_conditions = initial_conditions

        # Boundary stuff -> lower left point / width / height
        self.boundaries = [-1.6, -1, 3.2, 2]

        self.file_path = None
        self.current_file_size = 0

        # Constants
        self.time_step = 0.033
        self.robot_diameter = 0.11
        self.wheel_radius = 0.016
        self.base_length = 0.105
        self.max_linear_velocity = 0.2
        self.max_angular_velocity = 2*(self.wheel_radius/self.robot_diameter)*(self.max_linear_velocity/self.wheel_radius)
        self.max_wheel_velocity = self.max_linear_velocity/self.wheel_radius

        self.robot_radius = self.robot_diameter/2
        self.robot_length = 0.095
        self.robot_width = 0.09

        self.encoder_counts_per_revolution = 28.0
        self.motor_gear_ratio = 100.37

        self.collision_offset = 0.025 # May want to increase this
        self.collision_diameter = 0.135

        self.distance_sensors_enabled = use_distance_sensors
        self.distance_sensor_range = 1.2  # in meters
        self.distance_sensors_orientation = np.array([[-0.04, 0.0,  0.04, 0.05, 0.04,   0.0,   -0.04],
                                                      [ 0.04, 0.06, 0.05, 0.0,  -0.05, -0.06, -0.04],
                                                      [ math.pi, math.pi/2, math.pi/4, 0.0,  -math.pi/4, -math.pi/2, -math.pi]])
        if self.distance_sensors_enabled:
            self.distance_end_points = np.full((2, 7*number_of_robots), np.nan)  # x and y for each of the 7 sensors for each robot
        self.distance_sensor_error = 0.03 # 3% error based on the VL53L4CD datasheet

        self.imu_orientation = [[0.0594 - 0.00319], [0.0344628 - 0.0475], [0.0]] # x, y positions, and heading of IMU in robot frame (meters). Origin is assumed to be the center of the axle.

        self.obstacles = obstacles # M x 2 x 2 array of M obstacles defined by their start (left column) and end points (right column)

        self.velocities = np.zeros((2, number_of_robots))
        self.velocities_old = np.zeros((2, number_of_robots)) # Previous robot velocities for acceleration simulation
        self.poses = self.initial_conditions
        if self.initial_conditions.size == 0:
            self.poses = misc.generate_initial_conditions(self.number_of_robots, spacing=0.2, width=2.5, height=1.5)
        
        # Sensor Measurements
        self.distances = -1*np.ones((7, number_of_robots)) # Real robots return -1 if distance sensors are not enabled
        self.accelerations = np.zeros((3, number_of_robots))
        self.orientations = np.zeros((3, number_of_robots))
        self.magnetic_fields = np.zeros((3, number_of_robots))
        self.gyros = np.zeros((3, number_of_robots))
        self.initial_encoders = np.zeros((2, number_of_robots))
        self.encoders = np.zeros((2, number_of_robots))

        # Peripherals
        self.leds = np.zeros((3, number_of_robots))

        # Visualization
        self.figure = []
        self.axes = []
        self.led_patches = []
        self.chassis_patches = []
        self.right_wheel_patches = []
        self.left_wheel_patches = []
        self.base_patches = []
        self.distance_ray_patch = []
        
        self.figure, self.axes = plt.subplots()
        self.axes.set_axis_off()
        if(self.show_figure):
            for i in range(number_of_robots):
                # p = patches.RegularPolygon((self.poses[:2, i]), 4, math.sqrt(2)*self.robot_radius, self.poses[2,i]+math.pi/4, facecolor='#FFD700', edgecolor = 'k')
                p = patches.Rectangle(
                    xy=self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),
                    width=self.robot_length,
                    height=self.robot_width,
                    angle=(self.poses[2, i] + math.pi/4) * 180/math.pi,
                    facecolor="#FFD700",
                    edgecolor="k"
                )

                lled = patches.Circle(self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                        0.015*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                       self.robot_length/2/5, fill=False)
                rw = patches.Circle(self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                lw = patches.Circle(self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')

                # base = patches.Circle(self.poses[:2, i], self.robot_radius/5, facecolor='r')

                #lw = patches.RegularPolygon(self.poses[:2, i]+self.robot_radius*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                #                                0.035*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                #                                4, math.sqrt(2)*0.02, self.poses[2,i]+math.pi/4, facecolor='k')

                self.chassis_patches.append(p)
                self.led_patches.append(lled)
                self.right_wheel_patches.append(rw)
                self.left_wheel_patches.append(lw)
                # self.base_patches.append(base)

                self.axes.add_patch(rw)
                self.axes.add_patch(lw)
                self.axes.add_patch(p)
                self.axes.add_patch(lled)
                # self.axes.add_patch(base)

            # Draw arena
            self.boundary_patch = self.axes.add_patch(patches.Rectangle(self.boundaries[:2], self.boundaries[2], self.boundaries[3], fill=False))

            self.axes.set_xlim(self.boundaries[0]-0.1, self.boundaries[0]+self.boundaries[2]+0.1)
            self.axes.set_ylim(self.boundaries[1]-0.1, self.boundaries[1]+self.boundaries[3]+0.1)

            # Initialize distance sensor ray endpoints
            if self.distance_sensors_enabled:
                self.distance_ray_patch = plt.scatter(np.zeros((1, 7*self.number_of_robots)), np.zeros((1, 7*self.number_of_robots)), s=120, c='r')

            plt.ion()
            plt.show()

            plt.subplots_adjust(left=-0.03, right=1.03, bottom=-0.03, top=1.03, wspace=0, hspace=0)

    def set_velocities(self, ids, velocities):

        # Threshold linear velocities
        idxs = np.where(np.abs(velocities[0, :]) > self.max_linear_velocity)
        velocities[0, idxs] = self.max_linear_velocity*np.sign(velocities[0, idxs])

        # Threshold angular velocities
        idxs = np.where(np.abs(velocities[1, :]) > self.max_angular_velocity)
        velocities[1, idxs] = self.max_angular_velocity*np.sign(velocities[1, idxs])
        self.velocities = velocities

    def set_leds(self, ids, leds):
        """Set the led value for each robot in ids"""
        return

    def get_distances(self):
        """Get the distance sensor readings for each robot"""
        return self.distances
    
    def get_accelerations(self):
        """Get the accelerometer readings for each robot"""
        return self.accelerations
    
    def get_orientations(self):
        """Get the orientation readings for each robot"""
        return self.orientations
    
    def get_magnetic_fields(self):
        """Get the magnetic field readings for each of the robots"""
        return self.magnetic_fields
    
    def get_gyros(self):
        """Get the gyro readings for each of the robots"""
        return self.gyros
    
    def get_encoders(self):
        """Get the encoder readings for each of the robots"""
        return np.int32(self.encoders) - np.int32(self.initial_encoders)

    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    #Protected Functions
    def _threshold(self, dxu):
        dxdd = self._uni_to_diff(dxu)

        to_thresh = np.absolute(dxdd) > self.max_wheel_velocity
        dxdd[to_thresh] = self.max_wheel_velocity*np.sign(dxdd[to_thresh])

        dxu = self._diff_to_uni(dxdd)

    def _uni_to_diff(self, dxu):
        r = self.wheel_radius
        l = self.base_length
        dxdd = np.vstack((1/(2*r)*(2*dxu[0,:]-l*dxu[1,:]),1/(2*r)*(2*dxu[0,:]+l*dxu[1,:])))

        return dxdd

    def _diff_to_uni(self, dxdd):
        r = self.wheel_radius
        l = self.base_length
        dxu = np.vstack((r/(2)*(dxdd[0,:]+dxdd[1,:]),r/l*(dxdd[1,:]-dxdd[0,:])))

        return dxu

    def _validate(self, errors = {}):
        # This is meant to be called on every iteration of step.
        # Checks to make sure robots are operating within the bounds of reality.

        p = self.poses
        b = self.boundaries
        N = self.number_of_robots

        for i in range(N):
            x = p[0,i]
            y = p[1,i]

            if(x < b[0] or x > (b[0] + b[2]) or y < b[1] or y > (b[1] + b[3])):
                    if "boundary" in errors:
                        errors["boundary"] += 1
                    else:
                        errors["boundary"] = 1
                        errors["boundary_string"] = "iteration(s) robots were outside the boundaries."

        for j in range(N-1):
            for k in range(j+1,N):
                first_position = p[:2, j] + self.collision_offset*np.array([np.cos(p[2,j]), np.sin(p[2, j])])
                second_position = p[:2, k] + self.collision_offset*np.array([np.cos(p[2,k]), np.sin(p[2, k])])
                if(np.linalg.norm(first_position - second_position) <= (self.collision_diameter)):
                # if (np.linalg.norm(p[:2,j]-p[:2,k]) <= self.robot_diameter):
                    if "collision" in errors:
                        errors["collision"] += 1
                    else:
                        errors["collision"] = 1
                        errors["collision_string"] = "iteration(s) where robots collided."

        dxdd = self._uni_to_diff(self.velocities)
        exceeding = np.absolute(dxdd) > self.max_wheel_velocity
        if(np.any(exceeding)):
            if "actuator" in errors:
                errors["actuator"] += 1
            else:
                errors["actuator"] = 1
                errors["actuator_string"] = "iteration(s) where the actuator limits were exceeded."

        return errors

