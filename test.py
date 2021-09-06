import open3d as o3d
import numpy as np
import time as tm


def aaa(pt):
    print("({},{},{})".format(pt[0], pt[1], pt[2]))


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


A = np.array([2.1,4.5,3])
B = np.array([4.5,1.7,3.1])

t0 = tm.time()

v = B-A
l = np.linalg.norm(v)
c = (A+B)/2

ref = np.array([0,0,1])

# zero = 0.00000000000001


theta = np.arccos(np.dot(v,ref)/l)
axis = np.cross(v, ref)


R = rotation_matrix(axis, -theta)
# R = o3d.geometry.get_rotation_matrix_from_axis_angle([zero, zero, zero])


t1 = tm.time()
print(t1-t0)

box = o3d.geometry.OrientedBoundingBox(
    center=c,
    R=R,
    extent=np.array([1,1,l])
    )


v3d = o3d.utility.Vector3dVector([A,B])
pcd = o3d.geometry.PointCloud(v3d)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0,0,0])
mesh_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=c)
mesh_frame2.rotate(R)
box_vis = o3d.geometry.PointCloud(box.get_box_points())

line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)

lines = [[0,1]]
line_set2 = o3d.geometry.LineSet()
line_set2.points = o3d.utility.Vector3dVector([A,B])
line_set2.lines = o3d.utility.Vector2iVector(lines)

o3d.visualization.draw_geometries([pcd, mesh_frame, box_vis, line_set, mesh_frame2, line_set2])