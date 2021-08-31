import open3d as o3d
import numpy as np


box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=2.0, depth=4.0)
box2 = o3d.geometry.TriangleMesh.create_box(width=1.0, height=2.0, depth=4.0)

R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation=np.array([0.3, 0.3, 0.3]))

box.rotate(R)
box2.rotate(R)

box2.vertices = o3d.utility.Vector3dVector(np.asarray(box2.vertices) * np.array([1., 2., 4.]))
box2.translate(translation=np.array([3,3,3]))

o3d.visualization.draw_geometries([box, box2])