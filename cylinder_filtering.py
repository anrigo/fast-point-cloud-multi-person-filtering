import numpy as np

def plane(x, x0, normal):
    '''
    Equation of a plane passing through the point x0 and perpendicular
    to the normal vector.
    Input:
    - x: point -> type: np.array
    - x0: point, joint -> type: np.array
    - normal: normal vector, joints axis -> type: np.array
    Ouput:
    - Distance of the Plane
    '''
    return np.dot(normal, x-x0)


def cylinder(x, x1, x2, radius):
    '''
    Equation of a cylinder defined as the set of all points at 
    distance from the line passing through the two given points
    equal to the specified radius.
    Input:
    - x: point -> type: np.array
    - x1: point, joint -> type: np.array
    - x2: point, joint -> type: np.array
    - radius: Radius of the cylinder -> type: float
    Ouput:
    - Distance from the cylinder surface
    '''
    num = np.linalg.norm(np.cross(x-x1, x-x2))
    den = np.linalg.norm(x2-x1)

    return (num/den) - radius


def is_in_truncated_cylinder(point, x1, x2, r):
    '''
    System of inequalities that determines if a point belongs to the truncated cylinder defined 
    by the parameters.
    Input:
    - point: point -> type: np.array
    - x1: point, joint -> type: np.array
    - x2: point, joint -> type: np.array
    - r: radius of the cylinder -> type: float
    Ouput:
    - Wether the point belongs to the cylinder or not -> type: bool
    '''
    n = x1-x2

    return (plane(point, x1, n) <= 0
        and plane(point, x2, n) >=0
        and cylinder(point, x1, x2, r) <= 0) 
    

def sphere(x, c, radius): 
    '''
    Equation of the sphere with center x and radius radius.
    Input:
    - x: point -> type: np.array
    - c: center of the sphere -> type: np.array
    - radius: radius of the sphere -> type: float
    Ouput:
    - Distance from the sphere surface
    '''
    return np.linalg.norm(x-c) - radius


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

    #skel = skels[0]
    indices = []
   
    for skel in skels:
        body = skel[0]
        head_center = np.mean(body[[1,16,18]], axis=0)
        l_hand_center = np.mean(skel[1], axis=0)
        r_hand_center = np.mean(skel[2], axis=0)

        for i, point in enumerate(pcd.points):
            if (
                # ARMS

                # lShoulder, lElbow: 3,4
                is_in_truncated_cylinder(point, body[3], body[4], 5)

                # lElbow, lWrist: 4,5
                or is_in_truncated_cylinder(point, body[4], body[5], 5)

                # rShoulder, rElbow: 9,10
                or is_in_truncated_cylinder(point, body[9], body[10], 10)

                # rElbow, rWrist: 10,11
                or is_in_truncated_cylinder(point, body[10], body[11], 5)

                # LEGS

                # lHip, lKnee: 6,7
                or is_in_truncated_cylinder(point, body[6], body[7], 10)

                # lKnee, lAnkle: 7,8
                or is_in_truncated_cylinder(point, body[7], body[8], 10)

                # rHip, rKnee: 12,13
                or is_in_truncated_cylinder(point, body[12], body[13], 10)

                # rKnee, rAnkle: 13,14
                or is_in_truncated_cylinder(point, body[13], body[14], 10)

                # neck, bodyCenter: 0,2
                or is_in_truncated_cylinder(point, body[0], body[2], 18)

                # HEAD
                or sphere(point, head_center, 18) <= 0

                # SHOULDERS

                # lShoulder, Neck: 3,0
                or is_in_truncated_cylinder(point, body[3], body[0], 15)

                # rShoulder, Neck: 9,0
                or is_in_truncated_cylinder(point, body[9], body[0], 15)

                # HANDS

                # rHand
                or sphere(point, r_hand_center, 15) <= 0

                # lHand
                or sphere(point, l_hand_center, 15) <= 0
            ):
                indices.append(i)

    return pcd.select_by_index(indices)