import cv2
import numpy as np
import os, glob
import argparse
import imageio
from PIL import Image
import torch
import math
import yaml
import random
import sys
sys.path.append("../")
sys.path.append("vision")

import open3d as o3d
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def get_point_cloud_from_rgbd(depth, rgb, seg, vinv, proj, cam_w, cam_h):
    
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    # depth_buffer[seg_buffer == 0] = -10001
    points = []
    colors = []

    centerU = cam_w/2
    centerV = cam_h/2
    for i in range(cam_w):
        for j in range(cam_h):
            if depth[j, i] < -10000:
                continue
            if seg == None or seg[j, i] > 0:
                u = -(i-centerU)/(cam_w)  # image-space coordinate
                v = (j-centerV)/(cam_h)  # image-space coordinate
                d = depth[j, i]  # depth buffer value
                X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                p2 = X2*vinv  # Inverse camera view to get world coordinates
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                colors.append(rgb[j, i, :3])
    points, colors = np.array(points), np.array(colors)
    # import pdb; pdb.set_trace()
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

    # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3]/255.0)

    # o3d.visualization.draw_geometries([point_cloud])
    return np.array(points), np.array(colors)


def get_downsampled_pc(pcs_all_xyz, pcs_feas = None, sampled_num = 20000, sampling_method = "random_fps"):
    '''
    sampling method can be one of the following: random_fps, random, fps
        random_fps: random sampling to 4*sampled_num, then fps to sampled_num
        random: random sampling to sampled_num
        fps: fps sampling to sampled_num
    '''
    pcs_mask = np.zeros(pcs_all_xyz.shape[0], dtype=np.bool_)
    if pcs_all_xyz.shape[0] < sampled_num:
        pcs_xyz = pcs_all_xyz
        pcs_feas = pcs_feas if pcs_feas is not None else None
        pcs_mask[:] = True
        
    else:
        if sampling_method == "random_fps":
            if pcs_all_xyz.shape[0] > 10*sampled_num:
                ids = np.array(random.sample(range(pcs_all_xyz.shape[0]), int(4*sampled_num)))
                points_tmp_xyz = pcs_all_xyz[ids]
                points_tmp_feas = pcs_feas[ids] if pcs_feas is not None else None
            else:
                ids = np.arange(pcs_all_xyz.shape[0])
                points_tmp_xyz = pcs_all_xyz
                points_tmp_feas = pcs_feas if pcs_feas is not None else None

            sampled_points_ids = FPS(points_tmp_xyz, sampled_num)
            pcs_xyz = points_tmp_xyz[sampled_points_ids]
            pcs_feas = points_tmp_feas[sampled_points_ids] if points_tmp_feas is not None else None
            pcs_mask[ids[sampled_points_ids]] = True
        elif sampling_method == "random":
            ids = np.array(random.sample(range(pcs_all_xyz.shape[0]), int(sampled_num)))
            pcs_xyz = pcs_all_xyz[ids]
            pcs_feas = pcs_feas[ids] if pcs_feas is not None else None
            pcs_mask[ids] = True
        elif sampling_method == "fps":
            sampled_points_ids = FPS(pcs_all_xyz, sampled_num)
            pcs_xyz = pcs_all_xyz[sampled_points_ids]
            pcs_feas = pcs_feas[sampled_points_ids] if pcs_feas is not None else None
            pcs_mask[sampled_points_ids] = True
        else:
            raise NotImplementedError
    return pcs_xyz, pcs_feas, pcs_mask

def FPS(pcs, npoint):
    """
    Input:
        pcs: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_pcs: [npoint, 3]
        fps_idx: sampled pointcloud index, [npoint, ]
    """
    if pcs.shape[0] < npoint:
        print('Error! shape[0] of point cloud is less than npoint!')
        return None

    if pcs.shape[0] == npoint:
        return np.arange(pcs.shape[0])

    pcs_tensor = torch.from_numpy(np.expand_dims(pcs, 0)).float()
    fps_idx_tensor = farthest_point_sample(pcs_tensor, npoint)
    fps_idx = fps_idx_tensor.cpu().numpy()[0]
    return fps_idx


def farthest_point_sample(xyz, npoint, use_cuda = True):
    """
    Copied from CAPTRA

    Input:
        xyz: pointcloud data, [B, N, 3], tensor
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    
    if use_cuda:
        # print('Use pointnet2_cuda!')
        from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cuda
        sampled_points_ids = fps_cuda(torch.tensor(xyz).to("cuda").detach(), npoint)
        return sampled_points_ids
    else:
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid)**2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        sampled_points_ids = centroids
    return sampled_points_ids

def compute_camera_intrinsics_matrix(image_width, image_heigth, horizontal_fov, device):
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov *= np.pi / 180

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = torch.tensor([[f_x, 0.0, image_width / 2.0], [0.0, f_y, image_heigth / 2.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float)

    return K

def rgbd_to_point_cloud(rgb_image_np, depth_image_np, width, height, fx, fy, cx, cy):
    # Load RGB and Depth images
    # For this example, let's assume you have RGB and Depth images as numpy arrays
    # rgb_image_np = <your RGB image as numpy array>
    # depth_image_np = <your Depth image as numpy array>

    # Convert numpy arrays to Open3D images
    rgb_image_o3d = o3d.geometry.Image(rgb_image_np)
    depth_image_o3d = o3d.geometry.Image(depth_image_np)

    # Create an RGBD image from the RGB and Depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d, depth_scale=1, depth_trunc=100)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Convert the RGBD image to a point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Optionally visualize the point cloud
    # o3d.visualization.draw_geometries([point_cloud])

    return np.array(point_cloud.points), np.array(point_cloud.colors)

def get_point_cloud_from_rgbd_GPU(camera_depth_tensor, camera_rgb_tensor, camera_seg_tensor, camera_view_matrix_inv, camera_proj_matrix, width:float, height:float):
    # time1 = time.time()
    # print(u,v,width, height)
    # exit(123)
    device = camera_depth_tensor.device
    depth_buffer = camera_depth_tensor.to(device)
    rgb_buffer = camera_rgb_tensor.to(device)
    if camera_seg_tensor is not None:
        seg_buffer = camera_seg_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = torch.tensor(camera_view_matrix_inv).to(device)

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = torch.tensor(camera_proj_matrix).to(device)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]
    
    camera_u = torch.arange(0, width, device=device)
    camera_v = torch.arange(0, height, device=device)

    v, u = torch.meshgrid(
    camera_v, camera_u)

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv
    # print(rgb_buffer.shape)
    # print(seg_buffer.shape)
    R = rgb_buffer[...,0].view(-1)
    G = rgb_buffer[...,1].view(-1)
    B = rgb_buffer[...,2].view(-1)
    if camera_seg_tensor is not None:
        S = seg_buffer.view(-1)
        
    Z = Z.view(-1)
    X = X.view(-1)
    Y = Y.view(-1)

    if camera_seg_tensor is not None:
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B, S))
    else:
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B))
    position = position.permute(1, 0)
    position[:,0:4] = position[:,0:4]@vinv
    # print(position.shape)
    points = torch.cat((position[:, 0:3], position[:, 4:8]), dim = 1)

    return points

def images_to_video(image_folder, video_path, frame_size=(1920, 1080), fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")])

    if not images:
        print("No images found in the specified directory!")
        return
    
    writer = imageio.get_writer(video_path, fps=fps)
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = imageio.imread(img_path)

        if img.shape[1] > frame_size[0] or img.shape[0] > frame_size[1]:
            # print("Warning: frame size is smaller than the one of the images.")
            # print("Images will be resized to match frame size.")
            img = np.array(Image.fromarray(img).resize(frame_size))
        
        writer.append_data(img)
    
    writer.close()
    print("Video created successfully!")

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a Python dictionary
        config = yaml.safe_load(file)
    return config

def get_bounding_box(mask):
    """
    Calculate the bounding box of a mask.
    
    :param mask: A 2D NumPy array representing the mask.
    :return: A tuple (min_row, min_col, max_row, max_col) representing the bounding box.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return min_row, min_col, max_row, max_col


def get_first_frame(file_path):
    """
    Get the first frame of a image sequence file (.mp4 or .gif). NOTE: untested
    
    :param file_path: The path to the image sequence file.
    :return: A 2D NumPy array representing the first frame.
    """
    if file_path.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif file_path.endswith('.gif'):
        return readGif(file_path, asNumpy=True)[0]
    else:
        raise ValueError('Unsupported file format. Only .mp4 and .gif files are supported.')


def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    
    Read images from an animated GIF file.  Returns a list of numpy 
    arrays, or, if asNumpy is false, a list if PIL images.
    
    """
    
    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")
    
    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")
    
    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))
    
    # Load file using PIL
    pilIm = PIL.Image.open(filename)    
    pilIm.seek(0)
    
    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass
    
    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:            
            images.append( PIL.Image.fromarray(im) )
    
    # Done
    return images

def save_frames_as_gif(frames, path='./', filename=f'gifs/re3_multitask.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)


def crop_points(point, pcd_points, pcd_colors=None, thres=0.2, save_root=None):
    '''crop pcd close to a given point'''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    if pcd_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(point, thres)
    pcd = pcd.select_by_index(idx)
    if save_root:
        o3d.io.write_point_cloud(f"{save_root}/cropped_pcd.ply", pcd)
    ret_points = np.array(pcd.points)
    ret_colors = np.array(pcd.colors) if pcd_colors is not None else None
    ret_normals = np.array(pcd.normals)
    return ret_points, ret_colors, ret_normals


def cluster_normals(normals, n_clusters=5):
    '''
    use kmeans to cluster normals
    '''
    normal_normalized = normalize(normals)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normal_normalized)
    centers = kmeans.cluster_centers_
    clustered_normals = centers
    for i in range(n_clusters):
        # add opposite directions
        clustered_normals = np.vstack((clustered_normals, -centers[i]))
    return clustered_normals


def visualize_point_directions(points, point, clustered_normals, save_root=None, filename=None):
    # points, _, _ = get_downsampled_pc(points, None, 20000)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * np.ones((points.shape[0], 3)))
    pcd.points.append(point)
    pcd.colors.append(np.array([0, 1, 0]))
    for i, normal in enumerate(clustered_normals):
        # draw arrows
        n_points = 100
        line_points = [point + t * 0.1 * normal for t in np.linspace(0.01, 1, n_points)]
        line_colors = [np.array([0, 0, i/len(clustered_normals)]) for _ in range(n_points)]
        line_pcd = o3d.geometry.PointCloud()
        line_pcd.points = o3d.utility.Vector3dVector(np.array(line_points))
        line_pcd.colors = o3d.utility.Vector3dVector(np.array(line_colors))
        pcd += line_pcd
    if save_root:
        if filename:
            o3d.io.write_point_cloud(f"{save_root}/{filename}.ply", pcd)
        else:
            o3d.io.write_point_cloud(f"{save_root}/clustered_normals.ply", pcd)
            