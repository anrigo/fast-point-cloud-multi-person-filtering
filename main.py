import numpy as np
import open3d as o3d

def show_ptcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    show_ptcloud("kinoptic_ptclouds/171204_pose1/ptcloud_hd00000010.ply")