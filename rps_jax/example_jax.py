from rps_jax.robotarium import Robotarium

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap, lax


class WrappedRobotarium(object):
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
    
    def wrapped_step(self):
        return self.env.step()
    
    def move_circle(self, pose, radius):
        x, y, theta = pose
        v = 1.0  # linear velocity
        omega = v / radius  # angular velocity
        return jnp.array([v, omega])[:, None]

    def batched_step(self, poses, unused):
        actions = vmap(self.move_circle, in_axes=(0, None))(poses, 1.0)
        print(actions.shape)
        new_poses = jax.vmap(self.env.batch_step, in_axes=(0, 0))(poses, actions)
        return new_poses, new_poses

def main():
    env = Robotarium(number_of_robots=1)
    num_envs = 1
    timesteps = 1_000_000
    wrapped_env = WrappedRobotarium(env, num_envs)
    initial_poses = jnp.zeros((num_envs, 3, 1))
    final_poses, batch = jax.lax.scan(wrapped_env.batched_step, initial_poses, None, timesteps)
    return batch

if __name__ == "__main__":
    run = jax.jit(main)
    batch = jax.block_until_ready(run())
    print(batch.shape)

    # Select one environment to plot
    env_index = 0

    # Extract x and y positions over timesteps
    x_positions = batch[:, env_index, 0, 0]
    y_positions = batch[:, env_index, 1, 0]

    # Plot x and y positions
    plt.figure(figsize=(5, 5))
    plt.plot(x_positions, y_positions, label=f'Env {env_index}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory of one environment over time')
    plt.legend()
    plt.grid(True)
    plt.show()