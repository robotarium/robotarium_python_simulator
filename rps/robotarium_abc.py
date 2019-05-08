import time
import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import rps.utilities.misc as misc


class RobotariumABC(ABC):

    def __init__(self, number_of_agents=10, show_figure=True, save_data=True):

        self.number_of_agents = number_of_agents
        self.show_figure = show_figure
        self.save_data = save_data

        # Boundary stuff -> lower left point / width / height
        self.boundary = [-1.6, -1, 3.2, 2]

        self.file_path = None
        self.current_file_size = 0

        # Constants
        self.max_linear_velocity = 0.4
        self.max_angular_velocity = 4*np.pi

        self.robot_size = 0.055
        self.time_step = 0.033

        self.velocities = np.zeros((2, number_of_agents))
        self.poses = misc.generate_initial_conditions(self.number_of_agents, spacing=0.2, width=2.5, height=1.5)
        self.saved_poses = []
        self.saved_velocities = []
        self.left_led_commands = []
        self.right_led_commands = []

        # Visualization
        self.figure = []
        self.axes = []
        self.left_led_patches = []
        self.right_led_patches = []
        self.chassis_patches = []
        self.right_wheel_patches = []
        self.left_wheel_patches = []

        if(self.save_data):
            self.file_path = "robotarium_data_" + repr(int(round(time.time())))

        if(self.show_figure):
            self.figure, self.axes = plt.subplots()
            self.axes.set_axis_off()
            for i in range(number_of_agents):
                p = patches.RegularPolygon(self.poses[:2, i], 4, math.sqrt(2)*self.robot_size, self.poses[2,i]+math.pi/4, facecolor='#FFD700', edgecolor = 'k')
                rled = patches.Circle(self.poses[:2, i]+0.75*self.robot_size*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                        0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                       self.robot_size/5, fill=False)
                lled = patches.Circle(self.poses[:2, i]+0.75*self.robot_size*np.array((np.cos(self.poses[2, i]), np.sin(self.poses[2, i]))+\
                                        0.015*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2)))),\
                                       self.robot_size/5, fill=False)
                rw = patches.Circle(self.poses[:2, i]+self.robot_size*np.array((np.cos(self.poses[2, i]+math.pi/2), np.sin(self.poses[2, i]+math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                lw = patches.Circle(self.poses[:2, i]+self.robot_size*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                                                0.04*np.array((-np.sin(self.poses[2, i]+math.pi/2))),\
                                                0.02, facecolor='k')
                #lw = patches.RegularPolygon(self.poses[:2, i]+self.robot_size*np.array((np.cos(self.poses[2, i]-math.pi/2), np.sin(self.poses[2, i]-math.pi/2)))+\
                #                                0.035*np.array((-np.sin(self.poses[2, i]+math.pi/2), np.cos(self.poses[2, i]+math.pi/2))),\
                #                                4, math.sqrt(2)*0.02, self.poses[2,i]+math.pi/4, facecolor='k')

                self.chassis_patches.append(p)
                self.left_led_patches.append(lled)
                self.right_led_patches.append(rled)
                self.right_wheel_patches.append(rw)
                self.left_wheel_patches.append(lw)

                self.axes.add_patch(rw)
                self.axes.add_patch(lw)
                self.axes.add_patch(p)
                self.axes.add_patch(lled)
                self.axes.add_patch(rled)

            # Draw arena
            self.boundary_patch = self.axes.add_patch(patches.Rectangle(self.boundary[:2], self.boundary[2], self.boundary[3], fill=False))

            self.axes.set_xlim(self.boundary[0]-0.1, self.boundary[0]+self.boundary[2]+0.1)
            self.axes.set_ylim(self.boundary[1]-0.1, self.boundary[1]+self.boundary[3]+0.1)

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

    @abstractmethod
    def call_at_scripts_end(self):
        raise NotImplementedError()

    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()
