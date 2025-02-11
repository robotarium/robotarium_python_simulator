import unittest
import jax
import jax.numpy as jnp
from rps_jax.utilities.controllers import create_si_position_controller


class TestControllers(unittest.TestCase):
    """unit tests for controllers.py"""
    
    def test_create_si_position_controller(self):
        si_position_controller = create_si_position_controller()
        poses = jnp.array([[0, 0]])
        goals = jnp.array([[1, 1]])
        u = si_position_controller(poses.T, goals.T)
        self.assertEqual(u.shape, (2, 1))

        # test velocity limit
        u_magnitude = jnp.linalg.norm(u, axis=0)
        self.assertTrue(jnp.all(u_magnitude <= 0.5))
    