import unittest
import jax
import jax.numpy as jnp
from rps_jax.utilities.misc import generate_initial_conditions, at_pose, at_position


class TestTransformations(unittest.TestCase):
    """unit tests for transformations.py"""
    
    def test_create_si_to_uni_dynamics(self):
        from rps_jax.utilities.transformations import create_si_to_uni_dynamics
        si_to_uni_dynamics = create_si_to_uni_dynamics()
        u = jnp.array([[1], [1]])
        poses = jnp.array([[0, 0, 0]]).T
        dxu = si_to_uni_dynamics(u, poses)
        self.assertEqual(dxu.shape, (2, 1))
        self.assertTrue(jnp.all(dxu == jnp.array([[1], [jnp.pi/2]])))
