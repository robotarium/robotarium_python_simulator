import unittest
import jax
import jax.numpy as jnp
from jax import jit, vmap
from rps_jax.robotarium_abc import RobotariumABC
from rps_jax.utilities.misc import generate_initial_conditions, at_pose, at_position

class RobotariumTest(RobotariumABC):
    def get_poses(self):
        return

    def step(self):
        return

class TestRobotariumABC(unittest.TestCase):
    """unit tests for robotarium_abc.py"""

    def test_set_velocities(self):
        robotarium = RobotariumTest(number_of_robots=1)
        ids = jnp.array([0])
        velocities = jnp.array([0.1])
        robotarium.set_velocities(ids, velocities)
        self.assertTrue(jnp.all(robotarium.velocities == velocities))

    def test_threshold(self):
        robotarium = RobotariumTest(number_of_robots=1)

        # unthresholded
        dxu = jnp.array([[10], [10]])
        dxu_thresholded = robotarium._threshold(dxu)
        self.assertEqual(dxu_thresholded.shape, (2, 1))
        self.assertTrue(jnp.all(dxu_thresholded == dxu))

        # thresholded
        dxu = jnp.array([[100], [100]])
        dxu_thresholded = robotarium._threshold(dxu)
        self.assertEqual(dxu_thresholded.shape, (2, 1))
        self.assertFalse(jnp.all(dxu_thresholded == dxu))
    
    def test_uni_to_diff(self):
        robotarium = RobotariumTest(number_of_robots=1)
        dxu = jnp.array([[1], [1]])
        dxdd = robotarium._uni_to_diff(dxu)
        self.assertEqual(dxdd.shape, (2, 1))
    
    def test_diff_to_uni(self):
        robotarium = RobotariumTest(number_of_robots=1)
        dxdd = jnp.array([[1], [1]])
        dxu = robotarium._diff_to_uni(dxdd)
        self.assertEqual(dxu.shape, (2, 1))
    
    def test_validate(self):
        robotarium = RobotariumTest(number_of_robots=2)
        
        # check boundary violation
        robotarium.poses = jnp.array([[1.7, 0, 0], [0, 1.1, 0]]).T
        result = robotarium._validate()
        self.assertEqual(result['boundary'], 2)

        # check collision violation
        robotarium.poses = jnp.array([[1, 0, 0], [1, 0, 0]]).T
        result = robotarium._validate()
        self.assertEqual(result['collision'], 2)

        # check actuator violation
        dxu = jnp.array([[100], [100]])
        robotarium.set_velocities(jnp.array([0, 1]), dxu)
        result = robotarium._validate()
        self.assertEqual(result['actuator'], 2)

        # check no violation
        robotarium.poses = jnp.array([[1, 0, 0], [0, 1, 0]]).T
        dxu = jnp.array([[0], [0]])
        robotarium.set_velocities(jnp.array([0, 1]), dxu)
        result = robotarium._validate()
        self.assertEqual(result, {"boundary": 0, "collision": 0, "actuator": 0})


if __name__ == '__main__':
    unittest.main()
