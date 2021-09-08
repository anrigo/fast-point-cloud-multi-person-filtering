import numpy as np
from numpy.core.numeric import indices
import open3d as o3d


# EULER-RODRIGUES FORMULA
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_indices_of_points_in_prism_bbox(x1, x2, points):
    
    ref = np.array([0,0,1])

    v = x2-x1
    l = np.linalg.norm(v)
    c = (x1+x2)/2


    axis = np.cross(v, ref)

    if np.linalg.norm(axis) != 0:
        theta = np.arccos(np.dot(v,ref)/l)
        R = rotation_matrix(axis, -theta)
    else:
        R = np.eye(3)


    box = o3d.geometry.OrientedBoundingBox(
            center=c,
            R=R,
            extent=np.array([25,25,l])
            )
    
    idx = box.get_point_indices_within_bounding_box(points)

    return idx


def get_indices_of_points_in_cubic_bbox(x, points):
    box = o3d.geometry.OrientedBoundingBox(
            center=x,
            R=np.eye(3),
            extent=np.array([18,18,18])
            )
    
    idx = box.get_point_indices_within_bounding_box(points)

    return idx


def filter(pcd, skels):

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

    edges = [
        # ARMS
        [3,4], # l-shoulder-elbow
        [4,5], # l-elbow-wrist
        [9,10], # r-shoulder-elbow
        [10,11], # r-elbow-wrist

        # LEGS
        [6,7], # l-hip-knee
        [7,8], # l-knee-ankle
        [12,13], # r-hip-ankle
        [13,14], # r-knee-ankle
        
        # UPPER BODY
        [0,2], # neck-body-center
        [9,0], # r-shoulder-neck
        [3,0], # l-shoulder-neck

        # BACKSIDE
        [6, 12], # l-hip-r-hip
    ]

    skel = skels[0]
    body = skel[0]

    centers = [
        np.mean(body[[1,16,18]], axis=0), # head
        np.mean(skel[1], axis=0), # l-hand
        np.mean(skel[2], axis=0), # r-hand
        ]

    joints = [
        3, # l-shoulder
        9, # r-shoulder
        4, # l-elbow
        10, # r-elbow
        5, # l-wrist
        11, # r-wrist
        7, # l-knee
        13, # r-knee
    ]

    indices = []

    for edge in edges:
        indices.extend(
            get_indices_of_points_in_prism_bbox(body[edge[0]], body[edge[1]], pcd.points)
        )
    
    for center in centers:
        indices.extend(
            get_indices_of_points_in_cubic_bbox(center, pcd.points)
        )
    
    for joint in joints:
        indices.extend(
            get_indices_of_points_in_cubic_bbox(body[joint], pcd.points)
        )

    return pcd.select_by_index(indices)