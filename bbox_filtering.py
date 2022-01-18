import numpy as np
from numpy.core.numeric import indices
import open3d as o3d


# EULER-RODRIGUES FORMULA
def rotation_matrix(axis, theta):
    '''
    Function that uses the Euler-Rodrigues formula to compute the rotation matrix corresponding to 
    the axis and angle rotation theta.
    Input:
    - axis of rotation -> type: np.array
    - theta: rotation in radiants -> type: float
    Ouput:
    - Rotation Matrix
    '''
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_indices_of_points_in_prism_bbox(x1, x2, ratio, points):
    '''
    Funtion that extracts the points inside a bounding box
    Input:
    - x1: skeleton joint -> type: np.array
    - x2: skeleton joint -> type: np.array
    - ratio: Ratio of x and y dimensions w.r.t the z dimension -> type: float
    - points: Point cloud Points -> type: Vector3dVector
    Output:
    - idx : Indices of points inside the bounding box -> type: int
    '''

    # z axis
    ref = np.array([0,0,1])

    # joints axis
    v = x2-x1
    # length of axis
    l = np.linalg.norm(v)
    # center of the box: median point 
    c = (x1+x2)/2

    # Compute axis with cross 
    axis = np.cross(v, ref)

    # If the axis norm is not 0, then, there is a rotation
    if np.linalg.norm(axis) != 0:
        # Compute the roation angle with dot product 
        theta = np.arccos(np.dot(v,ref)/l)
        R = rotation_matrix(axis, -theta)
    else:
        # Use the identity matrix if there is no rotation
        R = np.eye(3)

    box = o3d.geometry.OrientedBoundingBox(
            center=c,
            R=R,
            extent=np.array([ratio*l,ratio*l,l])
            )
    
    idx = box.get_point_indices_within_bounding_box(points)

    return idx


def get_indices_of_points_in_cubic_bbox(x, extent, points):
    '''
    Funtion that extracts the points inside a bounding box
    Input:
    - x: center of the bounding box -> type: np.array
    - extent: Edge length of the cubic bounding box-> type: float
    - points: Point cloud Points -> type: Vector3dVector
    Output:
    - idx : Indices of points inside the bounding box -> type: int
    '''
    box = o3d.geometry.OrientedBoundingBox(
            center=x,
            R=np.eye(3),
            extent=np.array([extent, extent, extent])
            )
    
    idx = box.get_point_indices_within_bounding_box(points)

    return idx


def filter(pcd, skels):
    '''
    Function that extracts people from the scene given their skeleton
    Input:
    - pcd: point cloud -> type: PointCloud
    - skels: skeletons -> type: list
    Output:
    - Filtered Points -> type: PointCloud
    '''

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

    ratios = [
        # ARMS
        0.5, # l-shoulder-elbow
        0.4, # l-elbow-wrist
        0.5, # r-shoulder-elbow
        0.4, # r-elbow-wrist

        # LEGS
        0.6, # l-hip-knee
        0.6, # l-knee-ankle
        0.6, # r-hip-ankle
        0.6, # r-knee-ankle
        
        # UPPER BODY
        0.6, # neck-body-center
        0.9, # r-shoulder-neck
        0.9, # l-shoulder-neck

        # BACKSIDE
        0.4, # l-hip-r-hip
    ]
    # Cubic bounding boxes' extents
    sizes = [
        27, # head
        18, # l-hand
        18, # r-hand

        18, # l-shoulder
        18, # r-shoulder
        18, # l-elbow
        18, # r-elbow
        18, # l-wrist
        18, # r-wrist
        18, # l-knee
        18, # r-knee
    ]

    indices = []

    for skel in skels:

        body = skel[0]
        #Compute the mean of head, left and right hand in order to center the sphere. 
        centers = [
            np.mean(body[[1,16,18]], axis=0), # head
            np.mean(skel[1], axis=0), # l-hand
            np.mean(skel[2], axis=0), # r-hand
            ]


        for i, edge in enumerate(edges):
            indices.extend(
                get_indices_of_points_in_prism_bbox(body[edge[0]], body[edge[1]], ratios[i], pcd.points)
            )
        
        for i, center in enumerate(centers):
            indices.extend(
                get_indices_of_points_in_cubic_bbox(center, sizes[i], pcd.points)
            )
        
        for i, joint in enumerate(joints):
            indices.extend(
                get_indices_of_points_in_cubic_bbox(body[joint], sizes[i+3], pcd.points)
            )
            
    return pcd.select_by_index(indices)
    