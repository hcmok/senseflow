import numpy as np


def slerp(x, y, t):
    # x and y are unit vectors
    dot = np.clip(np.dot(x, y), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-8:
        z_t = (1 - t) * x + t * y  # linear interpolation
        z_t = z_t / (np.linalg.norm(z_t) + 1e-8)
        return z_t
    return (np.sin((1 - t) * theta) * x + np.sin(t * theta) * y) / np.sin(theta)
