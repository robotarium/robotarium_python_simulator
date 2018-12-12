import time
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
        self.poses = misc.generate_initial_conditions(self.number_of_agents, width=3, height=3)
        self.saved_poses = []
        self.saved_velocities = []
        self.led_commands = []

        # Visualization
        self.figure = []
        self.axes = []
        self.arrow_patches = []
        self.circle_patches = []

        if(self.save_data):
            self.file_path = "robotarium_data_" + repr(int(round(time.time())))

        self.arrow_patches = []

        if(self.show_figure):
            self.figure, self.axes = plt.subplots()
            self.axes.set_axis_off()
            for i in range(number_of_agents):
                p = patches.Circle(self.poses[:2, i], self.robot_size, fill=False)
                front = patches.Circle(self.poses[:2, i]+0.5*np.array((self.robot_size*np.cos(self.poses[2, i]), self.robot_size*np.sin(self.poses[2, i]))),
                                       self.robot_size/3, fill=False)

                self.circle_patches.append(p)
                self.arrow_patches.append(front)
                self.axes.add_patch(p)
                self.axes.add_patch(front)

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
