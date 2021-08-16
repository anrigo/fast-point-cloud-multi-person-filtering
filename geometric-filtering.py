import numpy as np


def plane(point, passing_point, normal):
    # plane passing through the point passing_point
    # and perpendicular to the vector normal

    return np.dot(normal, point-passing_point)