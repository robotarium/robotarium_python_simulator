import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

import rps_jax.utilities.misc as misc


class RobotariumABC(ABC):
    def __init__(self, number_of_robots=-1, sim_in_real_time=True, initial_conditions=jnp.array([])):
        """
        Initialize Robotarium Abstract Base Class

        Args:
            number_of_robots (int): Number of robots in the simulation
            sim_in_real_time (bool): If True, the simulation will run in real time
            initial_conditions (np.array): Initial conditions of the robots
        """
        self.number_of_robots = number_of_robots
        self.initial_conditions = initial_conditions

        # boundaries -> lower left point / width/ height
        self.boundaries = [-1.6, -1, 3.2, 2]

        # constants
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
        self.collision_offset = 0.025 # May want to increase this
        self.collision_diameter = 0.135
        self.velocities = jnp.zeros((2,self.number_of_robots))
        self.poses = self.initial_conditions

        if self.initial_conditions.size == 0:
            key = jax.random.PRNGKey(0)
            self.poses = misc.generate_initial_conditions(key, self.number_of_robots, spacing=0.2, width=2.5, height=1.5)

    def set_velocities(self, ids, velocities):
        """Set velocities for the robots."""
        self.velocities = velocities

    @abstractmethod
    def get_poses(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()
    
    # protected functions
    def _threshold(self, dxu: jnp.ndarray) -> jnp.ndarray:
        """
        Ensure velocity commands do not exceed max wheel velocity.

        Args:
            dxu (jnp.array): 2xN array of unicycle velocities
        
        Returns:
            jnp.array: 2xN array of thresholded unicycle velocities
        """
        
        dxdd = self._uni_to_diff(dxu)
        dxdd = jnp.clip(dxdd)
        return self._diff_to_uni(dxdd)
    
    def _uni_to_diff(self, dxu: jnp.ndarray) -> jnp.ndarray:
        """
        Convert unicycle model to differential drive model.
        
        Args:
            dxu (jnp.array): 2xN array of unicycle velocities
        
        Returns:
            jnp.array: 2xN array of differential drive velocities
        """
        r = self.wheel_radius
        l = self.base_length
        v = dxu[0, :]
        w = dxu[1, :]
        dxdd = jnp.array(
            [1/(2*r)*(2*v - l*w), # v_l
             1/(2*r)*(2*v + l*w)] # v_r
        )
        return dxdd

    def _diff_to_uni(self, dxdd: jnp.ndarray) -> jnp.ndarray:
        """
        Convert differential drive model to unicycle model.
        
        Args:
            dxdd (jnp.array): 2xN array of differential drive velocities
        
        Returns:
            jnp.array: 2xN array of unicycle velocities
        """
        r = self.wheel_radius
        l = self.base_length
        v_l = dxdd[0, :]
        v_r = dxdd[1, :]
        v = r/2*(v_l + v_r)
        w = r/l*(v_r - v_l)
        return jnp.array([v, w])

    def _validate(self, errors: dict = {}) -> dict:
        """
        Check boundary constraints, collisions, and actuator limits.
        
        Args:
            errors (dict): Dictionary to store error counts
        
        Returns:
            dict: updated Dictionary of error counts
        """

        b = self.boundaries
        p = self.poses
        N = self.number_of_robots

        # Check boundary conditions
        x_out_of_bounds = (p[0, :] < b[0]) | (p[0, :] > (b[0] + b[2]))
        y_out_of_bounds = (p[1, :] < b[1]) | (p[1, :] > (b[1] + b[3]))
        boundary_violations = jnp.where(x_out_of_bounds | y_out_of_bounds, 1, 0)
        boundary_violations = jnp.sum(boundary_violations)

        # Pairwise distance computation for collision checking
        distances = jnp.sqrt(jnp.sum((p[:, :, None] - p[:, None, :])**2, axis=0))
        collision_matrix = distances < self.collision_diameter
        collision_violations = jnp.sum(collision_matrix) - N  # Subtract N to remove self-collisions

        # Actuator limit check
        dxdd = self._uni_to_diff(self.velocities)
        exceeding = jnp.abs(dxdd) > self.max_wheel_velocity
        actuator_violations = jnp.sum(exceeding)

        return {
            "boundary": boundary_violations,
            "collision": collision_violations,
            "actuator": actuator_violations
        }
