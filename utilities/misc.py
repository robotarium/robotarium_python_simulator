import numpy as np


def generate_initial_conditions(N, spacing=0.1, width=1.2, height=0.6):

    x_range = int(np.floor(width/spacing))
    y_range = int(np.floor(height/spacing))

    choices = (np.random.choice(x_range*y_range, N, replace=False)+1)

    poses = np.zeros((3, N))

    for i, c in enumerate(choices):
        x,y = divmod(c, y_range)
        poses[0, i] = x*spacing - width/2
        poses[1, i] = y*spacing - height/2
        poses[2, i] = np.random.rand()*2*np.pi - np.pi

    return poses
