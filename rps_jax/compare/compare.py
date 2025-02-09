import time
import matplotlib.pyplot as plt
from example import drive_in_circle
from example_jax import drive_in_circle_jax
import timeit

def benchmark(func, num_envs, timesteps):
    elapsed_time = timeit.timeit(lambda: func(num_envs, timesteps), number=1)
    return elapsed_time

num_envs = 1
timesteps_list = [1000, 10000, 100000, 1000000, 10000000]
elapsed_times_python = []
elapsed_times_jax = []

print("\nBenchmarking drive_in_circle_jax:")
for timesteps in timesteps_list:
    elapsed_time_jax = benchmark(drive_in_circle_jax, num_envs, timesteps)
    print(elapsed_time_jax)
    elapsed_times_jax.append(elapsed_time_jax)

print("Benchmarking drive_in_circle:")
for timesteps in timesteps_list:
    elapsed_time = benchmark(drive_in_circle, num_envs, timesteps)
    elapsed_times_python.append(elapsed_time)

plt.figure(figsize=(8, 5))
plt.plot(timesteps_list, elapsed_times_python, label='Python')
plt.plot(timesteps_list, elapsed_times_jax, label='JAX')
plt.xlabel('Timesteps')
plt.ylabel('Elapsed Time (seconds)')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.savefig('rps_jax/compare/compare.png')
print(f"{timesteps} timesteps: {elapsed_time:.2f} seconds")