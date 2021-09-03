import open3d as o3d
import numpy as np


def aaa(pt):
    print("({},{},{})".format(pt[0], pt[1], pt[2]))


A = np.array([2,2,0])
l = np.linalg.norm(A)


# theta_x = np.pi/4
theta_x = np.arccos(A[0]/l)
theta_y = np.arccos(A[1]/l)
theta_z = np.arccos(A[2]/l)
print(theta_x)

R = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,-theta_x])
print(R)
# R_inv = np.linalg.inv(R)

V = np.dot(R, np.array([2,2,0]))
aaa(V)