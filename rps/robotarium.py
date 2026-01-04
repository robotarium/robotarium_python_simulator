import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from rps.robotarium_abc import *
from rps.utilities.misc import rotation_matrix

# Robotarium This object provides routines to interface with the Robotarium.
#
# THIS CLASS SHOULD NEVER BE MODIFIED OR SUBMITTED

class Robotarium(RobotariumABC):

        def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time = True, initial_conditions=np.array([]), use_distance_sensors=False, obstacles=np.full((1,2,2), np.nan)):
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
            if self.obstacles is not None:
                num_obstacles = self.obstacles.shape[0]
                for i in range(num_obstacles):
                    obstacle_patch = Line2D([self.obstacles[i,0,0], self.obstacles[i,0,1]], 
                                            [self.obstacles[i,1,0], self.obstacles[i,1,1]], 
                                            linewidth=4, color='0.5')
                    self.axes.add_line(obstacle_patch)

        def get_poses(self):
            """Returns the states of the agents.

            -> 3xN numpy array (of robot poses)
            """

            assert(not self._checked_poses_already), "Can only call get_poses() once per call of step()."
            # Allow step() to be called again.
            self._called_step_already = False
            self._checked_poses_already = True 

            return self.poses
        
        def simulate_encoder_readings(self):
            # Simulate encoder readings based on wheel velocities
            left_motor_angular_velocity = self._uni_to_diff(self.velocities)[0, :]
            right_motor_angular_velocity = self._uni_to_diff(self.velocities)[1, :]

            delta_encoder = self.encoder_counts_per_revolution*self.motor_gear_ratio/(2*math.pi)*np.vstack((left_motor_angular_velocity, right_motor_angular_velocity))*self.time_step
            self.encoders += np.round(delta_encoder)

        def simulate_distance_measurements(self):
            # Simualte distance measurements based on robot poses
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
                
                t = qxs/rxs  # num_obstacles x 1 x N_sensors. % Parameter for the intersection on the sensor lines
                u = qxr/rxs  # num_obstacles x 1 x N_sensors. % Parameter for the intersection on the obstacle lines

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
                t_circle = (-b - np.sqrt(discriminant))/(2*a)  # N x 1 x N_sensors. Parameter for intersection points on ray lines
                parameter_on_line_circle = np.logical_and(np.logical_and(t_circle >= 0, t_circle <= 1), np.imag(t_circle) == 0)  # N x 1 x N_sensors. Check if intersection points are on the rays
                valid_parameter_circle = t_circle*parameter_on_line_circle  # N x 1 x N_sensors
                # valid_parameter_circle[~parameter_on_line_circle] = self.distance_sensor_range  # N x 1 x N_sensors. Set invalid intersections to max range
                valid_parameter_circle[~parameter_on_line_circle] = np.nan  # N x 1 x N_sensors. Set invalid intersections to NaN
                
                valid_parameter_all = np.vstack((valid_parameter, valid_parameter_circle))  # Combine obstacle and robot intersection parameters
                min_parameter = np.nanmin(valid_parameter_all, axis=0).squeeze(0)  # 1 x N_sensors
                self.distances[:, i] = min_parameter + self.distance_sensor_error*(2*np.random.rand(1, N_sensors) - 1)  # Add noise to distance measurements

            # Find the endpoint of each sensor ray
            distance_end_points = global_sensors[:, 0:2, :] + self.distances.T.reshape(self.number_of_robots, 1, N_sensors)*r_all # N x 2 x 7
            self.distance_end_points = distance_end_points.transpose(1, 0, 2).reshape(2, self.number_of_robots*N_sensors) # 2 x N*7

            # Convert NaN distances to -1 for consistency with real robot API
            self.distances[np.isnan(self.distances)] = -1

        def call_at_scripts_end(self):
            """Call this function at the end of scripts to display potentail errors.  
            Even if you don't want to print the errors, calling this function at the
            end of your script will enable execution on the Robotarium testbed.
            """
            print('##### DEBUG OUTPUT #####')
            print('Your simulation will take approximately {0} real seconds when deployed on the Robotarium. \n'.format(math.ceil(self._iterations*0.033)))

            if bool(self._errors):
                if "boundary" in self._errors:
                    print('\t Simulation had {0} {1}\n'.format(self._errors["boundary"], self._errors["boundary_string"]))
                if "collision" in self._errors:
                    print('\t Simulation had {0} {1}\n'.format(self._errors["collision"], self._errors["collision_string"]))
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


            # Update dynamics of agents
            self.poses[0, :] = self.poses[0, :] + self.time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
            self.poses[1, :] = self.poses[1, :] + self.time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
            self.poses[2, :] = self.poses[2, :] + self.time_step*self.velocities[1, :]
            # Ensure angles are wrapped
            self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

            # Simulate encoder readings
            self.simulate_encoder_readings()

            # Update graphics
            if(self.show_figure):
                if(self.sim_in_real_time):
                    t = time.time()
                    while(t - self.previous_render_time < self.time_step):
                        t=time.time()
                    self.previous_render_time = t

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

                    self.right_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                            0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))  + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                    self.right_wheel_patches[i].orientation = self.poses[2, i] + math.pi/4

                    self.left_wheel_patches[i].center = self.poses[:2, i]+self.robot_length/2*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                            0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                    self.left_wheel_patches[i].orientation = self.poses[2,i] + math.pi/4
                    
                    self.led_patches[i].center = self.poses[:2, i]+0.75*self.robot_length/2*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                                    0.015*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i]))) + self.robot_length/2*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i])))
                    # self.base_patches[i].center = self.poses[:2, i]

                # Update distance sensor rays
                if self.distance_sensors_enabled:
                    self.simulate_distance_measurements()
                    self.distance_ray_patch.set_offsets(self.distance_end_points.T)
                    # print(self.distance_end_points)
                    

                self.figure.canvas.draw_idle()
                self.figure.canvas.flush_events()

