import os
import sys
import numpy as np
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI import GraspGroup, Grasp
sys.path.append(sys.path[0] + '/graspness_implementation')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspness_implementation.models.graspnet import GraspNet, pred_decode
from graspness_implementation.dataset.graspnet_dataset import minkowski_collate_fn
from graspness_implementation.utils.collision_detector import ModelFreeCollisionDetector
from scipy.spatial.transform import Rotation as R

class GSNet():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.net = GraspNet(seed_feat_dim=cfgs["seed_feat_dim"], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(cfgs["checkpoint_path"])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (cfgs["checkpoint_path"], start_epoch))
        self.net.eval()
        
    def inference(self, pcs):

        data_dict = {'point_clouds': pcs.astype(np.float32),
                    'coors': pcs.astype(np.float32) / self.cfgs["voxel_size"],
                    'feats': np.ones_like(pcs).astype(np.float32)}
        batch_data = minkowski_collate_fn([data_dict])
        tic = time.time()
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)
        # collision detection
        if self.cfgs["collision_thresh"] > 0:
            cloud = data_dict['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs["voxel_size_cd"])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs["collision_thresh"])
            gg = gg[~collision_mask]

        toc = time.time()
        print('inference time: %fs' % (toc - tic))
        return gg

def vis_grasp(pcs, gg):
    gg = gg.nms()
    gg = gg.sort_by_score()
    keep = 1
    if gg.__len__() > keep:
        gg = gg[:keep]
    grippers = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
    o3d.visualization.draw_geometries([cloud, *grippers])  
    
def grasp_to_pointcloud(grippers, cloud=None, gripper_points=1000):
    grippers_pcd = o3d.geometry.PointCloud()
    for gripper in grippers:
        g_pcd = gripper.sample_points_uniformly(gripper_points)
        grippers_pcd += g_pcd
    grippers_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (len(grippers_pcd.points), 1)))
    if cloud is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (len(cloud.points), 1)))
        grippers_pcd += cloud
    return grippers_pcd

def vis_save_grasp(points, gg, save_path, grasp_position=None, place_position=None):
    # visualize grasp pos, place pos, grasp poses, and pcd
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if type(gg) == GraspGroup:
        pcd_w_grasp = grasp_to_pointcloud(gg.to_open3d_geometry_list(), cloud)
    else:
        pcd_w_grasp = grasp_to_pointcloud([gg.to_open3d_geometry()], cloud)
    if grasp_position is not None and place_position is not None:
        pick_pcd = o3d.geometry.PointCloud()
        place_pcd = o3d.geometry.PointCloud()
        pick_pcd.points = o3d.utility.Vector3dVector(np.array(grasp_position).reshape(1,3).astype(np.float32))
        place_pcd.points = o3d.utility.Vector3dVector(np.array(place_position).reshape(1,3).astype(np.float32))
        pick_pcd.colors = place_pcd.colors = o3d.utility.Vector3dVector(np.array([[0,0,1]]).astype(np.float32))
        pcd_w_grasp = pcd_w_grasp + pick_pcd + place_pcd
    o3d.io.write_point_cloud(save_path, pcd_w_grasp)
    
    
def get_pose_from_grasp(best_grasp):
    grasp_position = best_grasp.translation
    # convert rotation to isaacgym convention
    delta_m = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    rotation = np.dot(best_grasp.rotation_matrix, delta_m)
    quaternion_grasp = R.from_matrix(rotation).as_quat()
    quaternion_grasp = np.array([quaternion_grasp[3],quaternion_grasp[0],quaternion_grasp[1],quaternion_grasp[2]])
    # rotation_unit_vect = rotation[:,2]
    # grasp_position -= 0.03 * rotation_unit_vect
    return grasp_position, quaternion_grasp, rotation

def get_best_grasp(gg, position, max_dis=0.05, top_down=False):
    '''
    # score-based
    best_grasp = None
    for g in gg:
        if np.linalg.norm(g.translation - position) < max_dis:
            if best_grasp is None:
                best_grasp = g
            else:
                if g.score > best_grasp.score:
                    best_grasp = g
    return best_grasp
    '''
    # get the closest grasp (within the threshold) to the position
    # position-based
    min_dis = 1e6
    best_grasp = None
    for g in gg:
        if top_down:
            _, _, rotation = get_pose_from_grasp(g)
            rotation_unit_vect = rotation[:,2]
            if rotation_unit_vect[2] > 0:
                continue
        dis = np.linalg.norm(g.translation - position)
        if dis < max_dis and dis < min_dis:
            min_dis = dis
            best_grasp = g
    return best_grasp
    # '''
    
def get_closest_grasp(gg, position):
    min_dis = 1e6
    best_grasp = None
    for g in gg:
        dis = np.linalg.norm(g.translation - position)
        if dis < min_dis:
            min_dis = dis
            best_grasp = g
    return best_grasp
    
def get_default_grasp(position):
    finger_front = np.array([0, 0, -1])
    finger_side = np.array([0, 1, 0])
    finger_front_norm = finger_front / np.linalg.norm(finger_front)
    finger_side_norm = finger_side / np.linalg.norm(finger_side)
    finger_face_norm = np.cross(finger_side_norm, finger_front_norm)

    quaternion = R.from_matrix(np.concatenate([finger_face_norm.reshape(-1,1), finger_side_norm.reshape(-1,1), finger_front_norm.reshape(-1,1)], axis = 1)).as_quat()
    gg = GraspGroup()
    g = Grasp()
    g.translation = position
    g.rotation_matrix = R.from_quat(quaternion).as_matrix()
    gg.add(g)
    
    return gg 