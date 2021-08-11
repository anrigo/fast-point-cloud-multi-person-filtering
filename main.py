import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt

def test():
    # Setup paths
    data_path = 'dataset/'
    seq_name = '171204_pose1'

    skel_points = None

    hd_skel_json_path = data_path+seq_name+'/hdPose3d_stage1_coco19/'
    hd_face_json_path = data_path+seq_name+'/hdFace3d/'
    hd_hand_json_path = data_path+seq_name+'/hdHand3d/'

    colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = -90, azim=-90)
    #ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    ax.axis('auto')

    # Select HD Image index
    hd_idx = 400

    '''
    ## Visualize 3D Body
    '''
    # Edges between joints in the body skeleton
    body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

    try:
        # Load the json file with this frame's skeletons
        skel_json_fname = hd_skel_json_path+'body3DScene_{0:08d}.json'.format(hd_idx)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

        # Bodies
        for ids in range(len(bframe['bodies'])):
            body = bframe['bodies'][ids]
            skel = np.array(body['joints19']).reshape((-1,4)).transpose()

            for edge in body_edges:
                ax.plot(skel[0,edge], skel[1,edge], skel[2,edge], color=colors[body['id']])

                for j in (0,1):
                    point = np.array([skel[0,edge][j], skel[1,edge][j], skel[2,edge][j]])
                    
                    if skel_points is None:
                        skel_points = point
                    elif point not in skel_points:
                        skel_points = np.vstack((skel_points, point))

                

    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)
    
    plt.show()
    print(skel_points.shape)


def show_ptcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # show_ptcloud("kinoptic_ptclouds/171204_pose1/ptcloud_hd00000010.ply")
    test()