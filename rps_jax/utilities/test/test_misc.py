import unittest
import jax
import jax.numpy as jnp
from rps_jax.utilities.misc import generate_initial_conditions, at_pose, at_position


class TestMisc(unittest.TestCase):
    """unit tests for misc.py"""

    def test_generate_initial_conditions(self):
        key = jax.random.PRNGKey(0)
        N = 5
        spacing = 0.3
        width = 3
        height = 1.8
        poses = generate_initial_conditions(key, N, spacing, width, height)
        self.assertEqual(poses.shape, (3, N))
        self.assertTrue(jnp.all(poses[0] >= -width/2))
        self.assertTrue(jnp.all(poses[0] <= width/2))
        self.assertTrue(jnp.all(poses[1] >= -height/2))
        self.assertTrue(jnp.all(poses[1] <= height/2))
        self.assertTrue(jnp.all(poses[2] >= -jnp.pi))
        self.assertTrue(jnp.all(poses[2] <= jnp.pi))

    def test_at_pose(self):
        # at pose
        states = jnp.array([[0, 1, jnp.pi], [0, 1, 0], [0, 1, -jnp.pi]])
        poses = jnp.array([[0, 1, jnp.pi], [0, 1, 0], [0, 1, -jnp.pi]])
        done = at_pose(states.T, poses.T)
        self.assertEqual(done.shape, (3,))
        self.assertEqual(done.tolist(), [1, 1, 1])

        # not at pose
        states = jnp.array([[0, 1, jnp.pi], [0, 1, 0], [0, 1, -jnp.pi]])
        poses = jnp.array([[0, 1+0.06, jnp.pi], [0, 1, 0.3], [0, 1, -jnp.pi]])
        done = at_pose(states.T, poses.T,  position_error=0.05, rotation_error=0.2)
        self.assertEqual(done.shape, (3,))
        self.assertEqual(done.tolist(), [0, 0, 1])

    def test_at_position(self):
        states = jnp.array([[0, 1, jnp.pi], [0, 1, 0]])
        points = jnp.array([[0, 1], [0, 1]])
        done = at_position(states.T, points.T)
        self.assertEqual(done.shape, (2,))
        self.assertEqual(done.tolist(), [1, 1])

        # not at position
        states = jnp.array([[0, 1, jnp.pi], [0, 1+0.6, 0]])
        points = jnp.array([[0, 1], [0, 1]])
        done = at_position(states.T, points.T, position_error=0.05)
        self.assertEqual(done.shape, (2,))
        self.assertEqual(done.tolist(), [1, 0])

if __name__ == '__main__':
    unittest.main()
