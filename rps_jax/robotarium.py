import math
import time
import jax
import jax.numpy as jnp
from rps_jax.robotarium_abc import RobotariumABC

class Robotarium(RobotariumABC):
    def __init__(self, number_of_robots=-1, show_figure=True, sim_in_real_time=True, initial_conditions=jnp.array([])):
        """
        Initialize Robotarium object

        Args:
            number_of_robots (int): Number of robots in the simulation
            show_figure (bool): If True, the simulation will be displayed
            sim_in_real_time (bool): If True, the simulation will run in real time
            initial_conditions (jnp.array): Initial conditions of the robots
        """
        
        super().__init__(number_of_robots, sim_in_real_time, initial_conditions)

        # Initialize checks for step and get poses calls
        self._called_step_already = True
        self._checked_poses_already = False

        # Initialization of error collection.
        self._errors = {}

        # Initialize steps
        self._iterations = 0
    
    def get_poses(self):
        """
        Returns the poses of the agents.

        Returns:
            (jnp.ndarray) 3xN array of robot poses
        """

        assert(not self._checked_poses_already), "Can only call get_poses() once per call of step()."
        # Allow step() to be called again.
        self._called_step_already = False
        self._checked_poses_already = True 

        return self.poses
    
    def set_poses(self, poses):
        """
        Set the poses of the agents.

        Args:
            poses (jnp.ndarray): 3xN array of robot poses
        """
        self.poses = poses
    
    def step(self):
        """Increment the simulation one step forward."""
        assert(self._called_step_already), "Must call get_poses() before calling step()."

        # allow get_poses to be called again
        self._called_step_already = True
        self._checked_poses_already = False

        # validate before thresholding velocities
        self._errors = self._validate()
        self._iterations += 1

        # perform thresholding of motors
        self.velocities = self._threshold(self.velocities)

        # x, y, theta
        x = self.poses[0, :] + self.time_step*jnp.cos(self.poses[2,:])*self.velocities[0, :]
        y = self.poses[1, :] + self.time_step*jnp.sin(self.poses[2,:])*self.velocities[0, :]
        theta = self.poses[2, :] + self.time_step*self.velocities[1, :]

        # ensure angles are wrapped
        theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

        # update poses
        self.poses = jnp.vstack((x, y, theta))
    
    def batch_step(self, poses, velocities):
        """Increment the simulation one step forward."""
        # assert(self._called_step_already), "Must call get_poses() before calling step()."

        # # allow get_poses to be called again
        # self._called_step_already = True
        # self._checked_poses_already = False

        # # validate before thresholding velocities
        # self._errors = self._validate()
        # self._iterations += 1

        # perform thresholding of motors
        velocities = self._threshold(velocities)

        # x, y, theta
        x = poses[0, :] + self.time_step * jnp.cos(poses[2, :]) * velocities[0, :]
        y = poses[1, :] + self.time_step * jnp.sin(poses[2, :]) * velocities[0, :]
        theta = poses[2, :] + self.time_step * velocities[1, :]

        # ensure angles are wrapped
        theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))

        # update poses
        poses = jnp.vstack((x, y, theta))
        return poses
