import numpy as np
import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_pose_controller_hybrid
from rps.utilities.misc import generate_random_poses, create_at_pose

N = 6
initial_positions = generate_random_poses(N, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_positions)

unicycle_barrier_certificate = create_uni_barrier_certificate_with_boundary()

position_error = 0.05
rotation_error = 0.2

pose_controller = create_pose_controller_hybrid(
    position_error=position_error, 
    rotation_error=rotation_error
)
at_pose = create_at_pose(position_error=position_error, rotation_error=rotation_error)

arena_width = r.BOUNDARIES[1] - r.BOUNDARIES[0] - 3*r.ROBOT_DIAMETER
arena_height = r.BOUNDARIES[3] - r.BOUNDARIES[2] - 3*r.ROBOT_DIAMETER
goal_poses = generate_random_poses(N, width=arena_width, height=arena_height, spacing=0.5)

x = r.get_poses()
r.step()

already_reported = np.zeros(N, dtype=bool)

while not at_pose(x, goal_poses)[0]:
    x = r.get_poses()

    dxu = pose_controller(x, goal_poses)
    dxu = unicycle_barrier_certificate(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    _, converged = at_pose(x, goal_poses)
    newly_arrived = converged & ~already_reported
    for i in np.where(newly_arrived)[0]:
        print(f"Robot {i+1} has reached its goal pose.")
    already_reported |= converged

r.debug()