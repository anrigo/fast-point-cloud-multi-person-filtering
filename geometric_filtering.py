import numpy as np


def plane(x, x0, normal):
    # plane passing through the point x0
    # and perpendicular to the vector normal

    return np.dot(normal, x-x0)


def cylinder(x, x1, x2, radius):
    # cylinder defined as the set of all points 
    # at distance from the line passing through the 
    # two given points equal to the specified radius

    num = np.linalg.norm(np.cross(x-x1, x-x2))
    den = np.linalg.norm(x2-x1)

    return (num/den) - radius


def is_in_truncated_cylinder(point, x1, x2, r):
    n = x1-x2

    if (plane(point, x1, n) <= 0
        and plane(point, x2, n) >=0
        and cylinder(point, x1, x2, r) <= 0):
        return True
    
    return False


def filter(pcd, skels, edges):
    # TODO
    # for each point in the pointcloud, check if it
    # belongs to the truncated cylinder defined between
    # the two points of each each edge
    
    # lElbow, lWrist
    edge = [4,5]

    x1 = skels[0][edge[0]]
    x2 = skels[0][edge[1]]

    n = x1-x2

    r = 5

    indexes = []

    for i, point in enumerate(pcd.points):
        if is_in_truncated_cylinder(point, x1, x2, r):
            indexes.append(i)

    return pcd.select_by_index(indexes)