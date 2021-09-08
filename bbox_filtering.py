import numpy as np
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

    skel = skels[0]
    body = skel[0]
    indexes = []

    head_center = np.mean(body[[1,16,18]], axis=0)
    l_hand_center = np.mean(skel[1], axis=0)
    r_hand_center = np.mean(skel[2], axis=0)

    A, B = body[3], body[4]


    v = B-A
    l = np.linalg.norm(v)
    c = (A+B)/2

    ref = np.array([0,0,1])

    theta = np.arccos(np.dot(v,ref)/l)
    axis = np.cross(v, ref)


    R = rotation_matrix(axis, -theta)


    box = o3d.geometry.OrientedBoundingBox(
        center=c,
        R=R,
        extent=np.array([20,20,l])
        )
    
    idx = box.get_point_indices_within_bounding_box(pcd.points)

    return pcd.select_by_index(idx)