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


def sphere(x, c, radius):
    return np.linalg.norm(x-c) - radius


def filter(pcd, skels, edges):

    # 0: Neck
    # 1: Nose
    # 2: BodyCenter (center of hips)
    # 3: lShoulder
    # 4: lElbow
    # 5: lWrist,
    # 6: lHip
    # 7: lKnee
    # 8: lAnkle
    # 9: rShoulder
    # 10: rElbow
    # 11: rWrist
    # 12: rHip
    # 13: rKnee
    # 14: rAnkle
    # 15: lEye
    # 16: lEar
    # 17: rEye
    # 18: rEar

    skel = skels[0]
    indexes = []

    head_center = np.mean(skel[[1,16,18]], axis=0)

    for i, point in enumerate(pcd.points):
        if (
            # ARMS

            # lShoulder, lElbow: 3,4
            is_in_truncated_cylinder(point, skel[3], skel[4], 5)

            # lElbow, lWrist: 4,5
            or is_in_truncated_cylinder(point, skel[4], skel[5], 5)

            # rShoulder, rElbow: 9,10
            or is_in_truncated_cylinder(point, skel[9], skel[10], 10)

            # rElbow, rWrist: 10,11
            or is_in_truncated_cylinder(point, skel[10], skel[11], 5)

            # LEGS

            # lHip, lKnee: 6,7
            or is_in_truncated_cylinder(point, skel[6], skel[7], 10)

            # lKnee, lAnkle: 7,8
            or is_in_truncated_cylinder(point, skel[7], skel[8], 10)

            # rHip, rKnee: 12,13
            or is_in_truncated_cylinder(point, skel[12], skel[13], 10)

            # rKnee, rAnkle: 13,14
            or is_in_truncated_cylinder(point, skel[13], skel[14], 10)

            # neck, bodyCenter: 0,2
            or is_in_truncated_cylinder(point, skel[0], skel[2], 18)

            # HEAD
            or sphere(point, head_center, 18) <= 0

            # SHOULDERS

            # lShoulder, Neck: 3,0
            or is_in_truncated_cylinder(point, skel[3], skel[0], 15)

            # rShoulder, Neck: 9,0
            or is_in_truncated_cylinder(point, skel[9], skel[0], 15)
        ):
            indexes.append(i)

    return pcd.select_by_index(indexes)