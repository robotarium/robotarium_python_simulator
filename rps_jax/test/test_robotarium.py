import unittest
import jax
import jax.numpy as jnp
from jax import jit, vmap
from rps_jax.robotarium import Robotarium
from rps_jax.utilities.misc import generate_initial_conditions, at_pose, at_position

class TestRobotariumABC(unittest.TestCase):
    """unit tests for robotarium.py"""

    def test_get_pose(self):
        robotarium = Robotarium(number_of_robots=1)
        poses = robotarium.get_poses()
        self.assertEqual(poses.shape, (3, 1))
        self.assertTrue(robotarium._checked_poses_already)
    
    def test_step(self):
        robotarium = Robotarium(number_of_robots=1)
        robotarium.poses = jnp.array([[0, 0, 0]]).T
        robotarium.set_velocities(jnp.array([0]), jnp.array([[1], [0]]))
        robotarium.step()
        
        self.assertTrue(robotarium._called_step_already)
        self.assertFalse(robotarium._checked_poses_already)
        self.assertTrue(robotarium.poses[0, 0] > 0) # check that the robot moved forward

        robotarium.poses = jnp.array([[0, 0, 0]]).T
        robotarium.set_velocities(jnp.array([0]), jnp.array([[0], [1]]))
        robotarium.step()
        self.assertTrue(robotarium.poses[2, 0] > 0) # check that the robot rotated


if __name__ == '__main__':
    unittest.main()
