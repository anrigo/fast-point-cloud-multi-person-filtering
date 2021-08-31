import open3d as o3d

cyl = o3d.geometry.create_mesh_cylinder(radius=1.0, height=3.0)
# cyl_scaled = cyl
# cyl_scaled.scale(2)

# o3d.visualization.draw_geometries([cyl])