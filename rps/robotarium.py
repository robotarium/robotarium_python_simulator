import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rps.robotarium_abc import *

class Robotarium(RobotariumABC):

        def __init__(self, number_of_agents=10, show_figure=True, save_data=True, update_time=1):
            super().__init__(number_of_agents, show_figure, save_data)

            #Initialize some rendering variables
            self.previous_render_time = time.time()
            self.update_time = update_time

        def call_at_scripts_end(self):
            """Call this function at the end of scripts to write data.  Even if you
            don't write any data, calling this function at the end of your script will
            accelerate execution on the server.
            """
            if(self.save_data):
                try:
                    np.save(self.file_path, self.saved_poses)
                except Exception as e:
                    raise

        def get_poses(self):
            """Returns the states of the agents.

            -> 3xN numpy array (of robot poses)
            """
            return self.poses

        def step(self):
            """Increments the simulation by updating the dynamics.
            """

            # Save data
            self.saved_poses.append(self.poses)
            self.saved_velocities.append(self.velocities)

            # Update dynamics of agents
            self.poses[0, :] = self.poses[0, :] + self.time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
            self.poses[1, :] = self.poses[1, :] + self.time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
            self.poses[2, :] = self.poses[2, :] + self.time_step*self.velocities[1, :]
            # Ensure angles are wrapped
            self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

            # Update graphics
            if(self.show_figure):
                t = time.time()
                if((t - self.previous_render_time) > self.update_time):
                    for i in range(self.number_of_agents):
                        self.chassis_patches[i].center = self.poses[:2, i]
                        self.chassis_patches[i].orientation = self.poses[2, i] + math.pi/4

                        self.right_wheel_patches[i].center = self.poses[:2, i]+self.robot_size*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))
                        self.right_wheel_patches[i].orientation = self.poses[2, i] + math.pi/4

                        self.left_wheel_patches[i].center = self.poses[:2, i]+self.robot_size*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))
                        self.left_wheel_patches[i].orientation = self.poses[2,i] + math.pi/4
                        
                        self.right_led_patches[i].center = self.poses[:2, i]+0.75*self.robot_size*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                                        0.04*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i])))
                        self.left_led_patches[i].center = self.poses[:2, i]+0.75*self.robot_size*np.array((np.cos(self.poses[2,i]), np.sin(self.poses[2,i])))-\
                                        0.015*np.array((-np.sin(self.poses[2, i]), np.cos(self.poses[2, i])))

                    self.figure.canvas.draw_idle()
                    self.figure.canvas.flush_events()
                    self.previous_render_time = t

