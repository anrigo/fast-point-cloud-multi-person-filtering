import open3d as o3d
import numpy as np


def aaa(pt):
    print("({},{},{})".format(pt[0], pt[1], pt[2]))


# A = np.array([2,2,0])
# l = np.linalg.norm(A)


# # theta_x = np.pi/4
# theta_x = np.arccos(A[0]/l)
# theta_y = np.arccos(A[1]/l)
# theta_z = np.arccos(A[2]/l)
# print(theta_x)

# R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,-theta_x])
# print(R)
# # R_inv = np.linalg.inv(R)

# V = np.dot(R, np.array([2,2,0]))
# aaa(V)

A = np.array([1.3,1.1,1.5])
B = np.array([2.6,2.4,2.9])

ref = np.array([np.deg2rad(90), np.deg2rad(90), 0])

v = B-A
l = np.linalg.norm(v)

c = (A+B)/2

rot = np.arccos(v/l)
print(np.rad2deg(rot))

delta = rot-ref

zero = 0.00000000000001


# R = o3d.geometry.get_rotation_matrix_from_axis_angle([zero, zero, zero])
# R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot)
R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([delta[1],delta[2],0])+np.array([zero, zero, zero]))

# R_inv = np.linalg.inv(R)

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