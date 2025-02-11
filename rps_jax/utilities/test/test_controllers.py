import unittest
import jax
import jax.numpy as jnp
from rps_jax.utilities.controllers import create_si_position_controller, create_clf_unicycle_position_controller
from rps_jax.utilities.transformations import create_si_to_uni_dynamics
from rps_jax.robotarium import Robotarium


class TestControllers(unittest.TestCase):
    """unit tests for controllers.py"""
    
    def test_create_si_position_controller(self):
        controller = create_si_position_controller()
        poses = jnp.array([[0, 0, 0]])
        goals = jnp.array([[1, 1]])
        u = controller(poses.T, goals.T)
        self.assertEqual(u.shape, (2, 1))

        # test velocity limit
        u_magnitude = jnp.linalg.norm(u, axis=0)
        self.assertTrue(jnp.all(u_magnitude <= 0.5))
    
    def test_drive_si_position_controller(self):
        N = 1  # number of robots
        initial_conditions = jnp.array([[0, 0, 0]]).T  # initial positions
        goals = jnp.array([[1, 1]]).T  # goal positions

        # Create Robotarium instance
        r = Robotarium(
            number_of_robots=N,
            initial_conditions=initial_conditions,
            show_figure=False,
            sim_in_real_time=False
        )

        controller = create_si_position_controller(x_velocity_gain=1, y_velocity_gain=1, velocity_magnitude_limit=1)
        si_to_uni_dynamics = create_si_to_uni_dynamics()

        # Drive to goal
        for _ in range(200):
            poses = r.get_poses()
            u = controller(poses, goals)
            r.set_velocities(range(N), si_to_uni_dynamics(u, poses)) # convert dx dy to v w   
            r.step()

        final_poses = r.get_poses()
        self.assertTrue(jnp.linalg.norm(final_poses[:2, :] - goals) < 0.01)
    
    def test_create_clf_unicycle_position_controller(self):
        controller = create_clf_unicycle_position_controller()
        poses = jnp.array([[0, 0]])
        goals = jnp.array([[1, 1]])
        u = controller(poses.T, goals.T)
        self.assertEqual(u.shape, (2, 1))
    
    def test_drive_clf_unicycle_position_controller(self):
        N = 1  # number of robots
        initial_conditions = jnp.array([[0, 0, 0]]).T  # initial positions
        goals = jnp.array([[1, 1]]).T  # goal positions

        # Create Robotarium instance
        r = Robotarium(
            number_of_robots=N,
            initial_conditions=initial_conditions,
            show_figure=False,
            sim_in_real_time=False
        )

        controller = create_clf_unicycle_position_controller()

        # Drive to goal
        for _ in range(200):
            poses = r.get_poses()
            u = controller(poses, goals)
            r.set_velocities(range(N), u)
            r.step()

        final_poses = r.get_poses()
        self.assertTrue(jnp.linalg.norm(final_poses[:2, :] - goals) < 0.01)