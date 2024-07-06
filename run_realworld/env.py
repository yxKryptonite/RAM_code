import imageio
import open3d as o3d
import cv2
import math
import numpy as np
import torch
import random
import time
from run_realworld.utils import get_bounding_box, crop_points, cluster_normals, visualize_point_directions
import os, json
import sys
import cv2
sys.path.append("../")
sys.path.append("vision")
from vision.GroundedSAM.grounded_sam_utils import prepare_GroundedSAM_for_inference, inference_one_image
from graspness_implementation.gsnet import GSNet, grasp_to_pointcloud, vis_save_grasp, get_best_grasp, get_pose_from_grasp, get_closest_grasp, get_default_grasp
import argparse
import traceback
from matplotlib import pyplot as plt

class MiniEnv():
    def __init__(
            self, 
            cfgs,
            grounded_dino_model = None, 
            sam_predictor = None,
        ):
        self.cfgs = cfgs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cam_w = cfgs['cam_w']
        self.cam_h = cfgs['cam_h']
        
        if grounded_dino_model is not None and sam_predictor is not None:
            self.grounded_dino_model = grounded_dino_model
            self.sam_predictor = sam_predictor
            self.box_threshold = cfgs['box_threshold']
            self.text_threshold = cfgs['text_threshold']
        elif self.cfgs["INFERENCE_GSAM"]:
            self.prepare_groundedsam()
        if self.cfgs["USE_GSNET"]:
            self.prepare_gsnet()
            
    def prepare_groundedsam(self):
        self.box_threshold = self.cfgs['box_threshold']
        self.text_threshold = self.cfgs['text_threshold']
        sam_version = "vit_h"
        sam_checkpoint = "assets/ckpts/sam_vit_h_4b8939.pth"
        grounded_checkpoint = "assets/ckpts/groundingdino_swint_ogc.pth"
        config = "vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

        self.grounded_dino_model, self.sam_predictor = prepare_GroundedSAM_for_inference(sam_version=sam_version, sam_checkpoint=sam_checkpoint,
                grounded_checkpoint=grounded_checkpoint, config=config, device=self.device)

    def prepare_gsnet(self):
        self.gsnet = GSNet(self.cfgs["gsnet"])
    
    def inference_gsnet(self, pcs, keep=1e6, nms=True):
        gg = self.gsnet.inference(pcs)
        if nms:
            gg = gg.nms()
        gg = gg.sort_by_score()
        if len(gg) > keep:
            gg = gg[:keep]
        if self.cfgs["gsnet"]["vis"]:
            grippers = gg.to_open3d_geometry_list()
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(pcs.astype(np.float32))
            o3d.visualization.draw_geometries([cloud, *grippers]) 
            
            # pcd_w_grasp = grasp_to_pointcloud(grippers, cloud)
            # o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/pcd_w_grasp.ply", pcd_w_grasp)
        
        return gg
    
    def detect_grasp_gsnet(self, points, colors=None, save_vis=False):
        '''GSNet'''
        # need to preprocess point cloud
        pcs_input = points.copy()
        pcs_input[...,2] = -pcs_input[...,2]
        gg = self.inference_gsnet(pcs_input, nms=False)
        # print(gg[0])
        # adjust grasps
        for g_i in range(len(gg)):
            translation = gg[g_i].translation
            rotation = gg[g_i].rotation_matrix
            translation = np.array([translation[0], translation[1], -translation[2]])
            rotation[2, :] = -rotation[2, :]
            rotation[:, 2] = -rotation[:, 2]
            gg.grasp_group_array[g_i][13:16] = translation
            gg.grasp_group_array[g_i][4:13] = rotation.reshape(-1)
        # print(gg[0])
        print('grasp num:', len(gg))
        if save_vis:
            vis_save_grasp(points, gg, f"{self.cfgs['SAVE_ROOT']}/gsnet.ply")
        return gg
    
    def detect_grasp_anygrasp(self, points, colors, save_vis=False):
        '''
        If you want to use AnyGrasp, check out https://github.com/graspnet/anygrasp_sdk to setup the SDK and put the `checkpoint_detection.tar` checkpoint to `assets/ckpts/`.
        
        And `gsnet.so`, `lib_cxx.so`, and `license/` should be in the project root directory.
        '''
        points = points.astype(np.float32)
        colors = colors.astype(np.float32)
        points_input = points.copy()
        colors_input = colors.copy()
        
        from gsnet import AnyGrasp # gsnet.so
        # get a argument namespace
        cfgs = argparse.Namespace()
        cfgs.checkpoint_path = 'assets/ckpts/checkpoint_detection.tar'
        cfgs.max_gripper_width = 0.1
        cfgs.gripper_height = 0.03
        cfgs.top_down_grasp = False
        cfgs.debug = False
        model = AnyGrasp(cfgs)
        model.load_net()
        
        lims = [-1, 1, -1, 1, -1, 1]
        gg, cloud = model.get_grasp(points_input, colors_input, lims, \
            # apply_object_mask=True, dense_grasp=True, collision_detection=True
                                       )
        print('grasp num:', len(gg))
        if save_vis:
            vis_save_grasp(points, gg, f"{self.cfgs['SAVE_ROOT']}/anygrasp.ply")
        return gg

    ### affordance
    def test_keypoint(self, rgb, pcd, pixel, dir):
        post_contact_dirs_2d, post_contact_dirs_3d = None, None
        partial_points = np.array(pcd.points)
        partial_colors = np.array(pcd.colors)
        position = partial_points[pixel[1]*self.cam_w + pixel[0]]
        
        # visualization
        # ds_points, _, _ = get_downsampled_pc(partial_points, None, 20000)
        ds_points, _, _ = crop_points(position, partial_points, thres=0.5)
        save_pcd = o3d.geometry.PointCloud()
        save_pcd.points = o3d.utility.Vector3dVector(ds_points)
        # red
        save_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((ds_points.shape[0], 3)))
        # add a point
        save_pcd.points.append(position)
        save_pcd.colors.append(np.array([0, 1, 0]))
        # save to ply
        o3d.io.write_point_cloud(f"{self.cfgs['SAVE_ROOT']}/grasp_point.ply", save_pcd)
        
        MAX_ATTEMPTS = 20 # in case there is no good grasp at one time
        max_dis = 0.05
        best_grasp = None
        max_radius, min_radius = 0.2, 0.1
        gg = None
        for num_attempt in range(MAX_ATTEMPTS):
            try:
                crop_radius = max_radius - (max_radius - min_radius) * num_attempt / MAX_ATTEMPTS
                print('=> crop_radius:', crop_radius)
                cropped_points, cropped_colors, cropped_normals = crop_points(
                    position, partial_points, partial_colors, thres=crop_radius, save_root=self.cfgs['SAVE_ROOT']
                )
                try:
                    # gg = self.detect_grasp_anygrasp(cropped_points, cropped_colors, save_vis=True) # use AnyGrasp if properly set up
                    gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, save_vis=True)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    traceback.print_exc()
                if gg is None:
                    continue
                print('=> total grasp:', len(gg))
                if len(gg) == 0:
                    continue
                
                best_grasp = get_best_grasp(gg, position, max_dis=max_dis) # original: 0.03
                if best_grasp is None:
                    print('==>> no best grasp')
                else:
                    break
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
        if best_grasp is None:
            try:
                gg = self.detect_grasp_gsnet(cropped_points, cropped_colors, False)
            except:
                gg = self.detect_grasp_gsnet(partial_points, partial_colors, False)
            best_grasp = get_closest_grasp(gg, position)
            print('==>> use GSNet for closest grasp')
        n_clusters = 5
        vis_save_grasp(cropped_points, best_grasp, f"{self.cfgs['SAVE_ROOT']}/best_grasp.ply")
        clustered_centers = cluster_normals(cropped_normals, n_clusters=n_clusters) # (2*n_clusters, 3)
        visualize_point_directions(cropped_points, position, clustered_centers, self.cfgs['SAVE_ROOT'])
        post_contact_dirs_3d = clustered_centers
        post_contact_dirs_2d = self.project_normals(rgb, pixel, clustered_centers)
        grasp_array = best_grasp.grasp_array.tolist()
        
        # post-grasp
        best_dir_3d, best_score = None, -1
        for i in range(post_contact_dirs_2d.shape[0]):
            score = np.dot(post_contact_dirs_2d[i], dir)
            if score > best_score:
                best_score = score
                best_dir_3d = post_contact_dirs_3d[i]
        visualize_point_directions(ds_points, position, [best_dir_3d], self.cfgs['SAVE_ROOT'], "best_dir_3d")
        post_grasp_dir = best_dir_3d.tolist()
        
        ret_dict = {
            "grasp_array": grasp_array,
            "post_grasp_dir": post_grasp_dir
        }
        return ret_dict

    def project_normals(self, img, pixel, normals):
        '''
        project 3D normals into 2D space
        normals: [n, 3], n is the number of normals
        '''
        normals = normals[:, :2]  # Ignore Z component for direction because in camera frame
        normals_2d_direction = torch.from_numpy(normals).float().cuda()

        # Optionally, normalize the 2D vectors to have unit length, showing direction only
        norms = torch.norm(normals_2d_direction, dim=1, keepdim=True)
        normals_2d_direction_normalized = normals_2d_direction / norms
        
        normals_2d_direction_normalized = normals_2d_direction_normalized.cpu().numpy()

        # visualize 2D normal on the image
        img_ = np.asarray(img)
        plt.imshow(img_)
        for i in range(normals_2d_direction_normalized.shape[0]):
            x, y = pixel
            dx, dy = normals_2d_direction_normalized[i]
            cv2.arrowedLine(img_, (x, y), (x + int(dx * 50), y + int(dy * 50)), (0, 0, 255*i/normals_2d_direction_normalized.shape[0]), 2)
            plt.arrow(x, y, dx*100, dy*100, color=(0, 0, i/normals_2d_direction_normalized.shape[0]), linewidth=2.5, head_width=12)
        # revert to RGB
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.cfgs['SAVE_ROOT']}/normals_projected.png")
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{self.cfgs['SAVE_ROOT']}/normals_projected_cv2.png", img_)
        plt.close()

        return normals_2d_direction_normalized

