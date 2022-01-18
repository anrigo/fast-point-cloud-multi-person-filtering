from typing import Sequence
import numpy as np
import open3d as o3d
import json
import bbox_filtering as bf
import time as tm
import cylinder_filtering as cf


# Constants
# Setup paths
data_path = 'dataset/'


def load_skeleton_points_as_nparray(seq_name, hd_idx):
    '''
    Function that loads the skeleton joints
    Input:
    - seq_name: It is the name of the sequence -> type: string
    - hd_idx : it is the index of the frame  -> type int 

    Output
    - Skeleton points: the joints of the skeleton and the hands -> type: list
    '''

    skel_points = []
    hands = []
    #Path of the folder that contains the json file with the points of the pose
    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
    #Path of the folder that contains the json file with the points of the hands 
    hd_hand_json_path = data_path+seq_name+'/hdHand3d/'

    
    try:
        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)
        

        # Load hand json
        hand_json_fname = hd_hand_json_path+'handRecon3D_hd{0:08d}.json'.format(hd_idx)
        with open(hand_json_fname) as dfile:
            hframe = json.load(dfile)

        # Cycle Bodies
        for ids in range(len(bframe['bodies'])):
            body = bframe['bodies'][ids]

            # skeleton format: x,y,z,c where c is the confidence
            # keep 3d coordinates, remove confidence score
            body_points = np.delete(np.array(body['joints19']).reshape((-1,4)), [-1], axis=1)

            skel_points.insert(ids, body_points)
        
        # Cycle Hands
        for hand in hframe['people']:
            hand3d_r = np.array(hand['right_hand']['landmarks']).reshape((-1,3))
            hand3d_l = np.array(hand['left_hand']['landmarks']).reshape((-1,3))

            hands.append([hand3d_l, hand3d_r])

    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)
    
    
    skels = [[skel_points[i], hands[i][0], hands[i][1]] for i in range(len(skel_points))]

    return skels


def load_ptcloud(sequence, hd_idx, draw=False):
    '''
    Function that loads a point cloud
    Input:
    - sequence: it is the name of the sequence  -> type: string 
    - hd_idx: it is the index of the frame  -> type int 
    - draw : indicates if you want to display the point cloud or not -> type: bool
    Output:
    - pcd: it returns the point cloud and if the variable draw is true, it also draws the point cloud 
    '''

    path = f"kinoptic_ptclouds/{sequence}" + "/ptcloud_hd{0:08d}.ply".format(hd_idx)

    pcd = o3d.io.read_point_cloud(path)
    #If draw = True, it draw the point cloud with a path like variable path
    if draw: 
        o3d.visualization.draw_geometries([pcd])
    
    return pcd


if __name__ == "__main__":
    #Global variables
    sequence = "170407_haggling_a1"
    pcd_idx = 1700

    #Load pcd
    pcd = load_ptcloud(sequence, pcd_idx, draw=True)

    #Load skeleton
    skels = load_skeleton_points_as_nparray(sequence, pcd_idx)

    t0 = tm.time()
    #Filter using the fast algorithm 
    filtered = bf.filter(pcd, skels)
    #Filter with the slow algorithm
    #filtered = cf.filter(pcd,skels)
    t1 = tm.time()

    print(t1-t0)

    #Visualize the output of the algorithm
    o3d.visualization.draw_geometries([filtered])

   