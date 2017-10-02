import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robotarium_abc import *

class Robotarium(RobotariumABC):

        def __init__(self, number_of_agents=10, show_figure=True, save_data=True):
            super().__init__(number_of_agents, show_figure, save_data)

        def call_at_scripts_end(self):
            pass

        def get_poses(self):
            return self.poses

        def step(self):
            # Update dynamics of agents
            self.poses[0, :] = self.poses[0, :] + self.time_step*np.cos(self.poses[2,:])*self.velocities[0, :]
            self.poses[1, :] = self.poses[1, :] + self.time_step*np.sin(self.poses[2,:])*self.velocities[0, :]
            self.poses[2, :] = self.poses[2, :] + self.time_step*self.velocities[1, :]
            # Ensure angles are wrapped
            self.poses[2, :] = np.arctan2(np.sin(self.poses[2, :]), np.cos(self.poses[2, :]))

            # Update graphics
            if(self.show_figure):
                for i in range(self.number_of_agents):
                    self.circle_patches[i].center = self.poses[:2, i]

                    self.arrow_patches[i].remove()
                    a = patches.Arrow(*self.poses[:2, i],
                        self.robot_size*np.cos(self.poses[2,i]),
                        self.robot_size*np.sin(self.poses[2,i]),
                        width=0.03)
                    self.arrow_patches[i] = a
                    self.axes.add_patch(a)

                self.figure.canvas.draw()
