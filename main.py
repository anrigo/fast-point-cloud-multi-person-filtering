import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt

# Constants
# Setup paths
data_path = 'dataset/'
# Edges between joints in the body skeleton
body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1


def load_skeleton_points_as_nparray(seq_name, hd_idx):
    skel_points = []

    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'

    try:
        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

        # Bodies
        for ids in range(len(bframe['bodies'])):
            body = bframe['bodies'][ids]

            # skeleton format: x,y,z,c where c is the confidence
            # keep 3d coordinates, remove confidence score
            body_points = np.delete(np.array(body['joints19']).reshape((-1,4)), [-1], axis=1)

            
            skel_points.insert(ids, body_points)

    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)
    
    
    return skel_points


def show_ptcloud_from_file(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    show_ptcloud_from_file("kinoptic_ptclouds/171204_pose1/ptcloud_hd00000175.ply")
    skels = load_skeleton_points_as_nparray('171204_pose1', 175)

    # print(skels[0])

    skel_points = o3d.utility.Vector3dVector(skels[0])
    pt_cloud = o3d.geometry.PointCloud(skel_points)
    o3d.visualization.draw_geometries([pt_cloud])