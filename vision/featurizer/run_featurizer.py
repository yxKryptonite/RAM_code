import torch
from vision.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer
from vision.featurizer.utils.visualization import IMG_SIZE, Demo, visualize_max_xy, visualize_max_xy_linear, visualize_max_xy_list
from PIL import Image
from torchvision.transforms import PILToTensor
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

featurizers = {
    'sd': SDFeaturizer,
    'clip': CLIPFeaturizer,
    'dino': DINOFeaturizer,
    'dinov2': DINOv2Featurizer,
    'radio': RADIOFeaturizer,
    'sd_dinov2': SD_DINOv2Featurizer
}

def run_demo(src_path, tgt_path, prompt):
    file_list = [src_path, tgt_path]
    img_list = []
    ft_list = []
    for filename in file_list:
        img = Image.open(filename).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_list.append(img)
        ft = extract_ft(img, prompt)
        ft_list.append(ft)
    
    ft = torch.cat(ft_list, dim=0)
    demo = Demo(img_list, ft, IMG_SIZE)
    demo.plot_img_pairs(fig_size=5)


def extract_ft(img: Image.Image, prompt=None, ftype='sd'):
    '''
    preprocess of img to `img`:
    img = Image.open(filename).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    '''
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2 # C, H, W
    img_tensor = img_tensor.unsqueeze(0).cuda() # 1, C, H, W

    assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2']
    featurizer = featurizers[ftype]()
    
    ft = featurizer.forward(
        img_tensor,
        block_index=1, # only for clip & dino
        prompt=prompt, # only for sd
        ensemble_size=2 # only for sd
    )
    return ft

def match_fts(src_ft, tgt_ft, pos, save_root=None):
    num_channel = src_ft.size(1)
    src_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(src_ft)
    tgt_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(tgt_ft)
    x, y = pos[0], pos[1]
    # interpolation from src_ft
    # src_vec = src_ft[0, :, int(y), int(x)].view(1, num_channel)  # 1, C
    x_norm = 2 * x / (IMG_SIZE - 1) - 1
    y_norm = 2 * y / (IMG_SIZE - 1) - 1
    src_vec = torch.nn.functional.grid_sample(src_ft, torch.tensor([[[[x_norm, y_norm]]]]).float().cuda(), align_corners=True).squeeze(2).squeeze(2)
    tgt_vecs = tgt_ft.view(1, num_channel, -1) # 1, C, H*W
    src_vec = F.normalize(src_vec) # 1, C
    tgt_vecs = F.normalize(tgt_vecs) # 1, C, HW
    cos_map = torch.matmul(src_vec, tgt_vecs).view(1, IMG_SIZE, IMG_SIZE).cpu().numpy() # 1, H, W

    return cos_map

def sample_highest(cos_map: np.ndarray):
    max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
    max_xy = (max_yx[1], max_yx[0])
    return max_xy, cos_map[0][max_yx]

def sample_region(cos_map: np.ndarray, cos_threshold=0.9, size_threshold=1000):
    '''sample regions with high confidence from cos_map'''
    high_confidence = cos_map > cos_threshold
    labeled, num_features = label(high_confidence)
    region_map = np.zeros_like(high_confidence)
    for i in range(1, num_features):
        region_mask = labeled == i
        if np.sum(region_mask) > size_threshold:
            region_map += i * region_mask
    return region_map

def sample_points_from_best_region(cos_map: np.ndarray, best_region_map, topk=10, cos_threshold=0.9):
    '''sample pixel points with highest confidences from the best region'''
    best_region_mask = best_region_map == 1
    cos_map = cos_map * best_region_mask
    cos_map[cos_map < cos_threshold] = 0
    max_idx = np.argsort(cos_map, axis=None)[-topk:]
    max_yx = np.unravel_index(max_idx, cos_map.shape)
    return max_yx # (vec_0, vec_y, vec_x)


def fit_linear_ransac(points, threshold=10, min_samples=2, max_trials=1000):
    '''fit a line to points using RANSAC'''
    if min_samples >= len(points):
        min_samples = len(points) // 2
    best_inliers = []
    best_line = None
    for _ in range(max_trials):
        sample = points[np.random.choice(len(points), min_samples, replace=False)]
        line = np.polyfit(sample[:, 0], sample[:, 1], 1)
        inliers = np.abs(points[:, 1] - (line[0] * points[:, 0] + line[1])) < threshold
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = line
    print("Number of best inliers:", np.sum(best_inliers))
    inlier_points = points[best_inliers]
    # turn best_line into direction
    best_line = np.array([1, best_line[0]])
    best_line = best_line / np.linalg.norm(best_line)
    # determine the sign of best_line
    positive_value = 0
    for idx in range(inlier_points.shape[0]-1):
        positive_value += np.dot(inlier_points[idx+1] - inlier_points[idx], best_line)
    if positive_value < 0:
        best_line = -best_line
    # end_start_vec = points[-1] - points[0]
    # if np.dot(best_line, end_start_vec) < 0:
    #     best_line = -best_line
    return inlier_points, best_line


def horizontal_flip_augmentation(src_img_PIL, src_pos_list):
    size = src_img_PIL.size
    augmented_img_PILs = [src_img_PIL]
    augmented_pos_lists = [src_pos_list]
    flipped_img_PIL = src_img_PIL.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_pos_list = []
    for pos in src_pos_list:
        flipped_pos_list.append((size[0] - pos[0], pos[1]))
    augmented_img_PILs.append(flipped_img_PIL)
    augmented_pos_lists.append(flipped_pos_list)
    return augmented_img_PILs, augmented_pos_lists

def choose_from_augmentation(augmented_img_PILs, augmented_pos_lists, tgt_img_PIL, prompt, ftype='sd'):
    max_cos = -1e6
    max_idx = -1
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    for idx in range(len(augmented_img_PILs)):
        src_ft = extract_ft(augmented_img_PILs[idx], prompt=prompt, ftype=ftype)
        cos_map = match_fts(src_ft, tgt_ft, augmented_pos_lists[idx][0])
        _, cos = sample_highest(cos_map)
        if cos > max_cos:
            max_cos = cos
            max_idx = idx
    return augmented_img_PILs[max_idx], augmented_pos_lists[max_idx]
        

def transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=None, ftype='sd'):
    ori_cos_map = None
    max_xy_list = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    match_fts(src_ft, tgt_ft, src_pos_list[0], save_root)
    for src_pos in src_pos_list:
        cos_map = match_fts(src_ft, tgt_ft, src_pos)
        if ori_cos_map is None:
            ori_cos_map = cos_map
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
    src_pos_list_np = np.array(src_pos_list)
    max_xy_list_np = np.array(max_xy_list)
    src_pos_inliers, src_best_line = fit_linear_ransac(src_pos_list_np, threshold=5, min_samples=10)
    max_xy_inliers, tgt_best_line = fit_linear_ransac(max_xy_list_np, threshold=5, min_samples=10)
    # src_best_line and tgt_best_line should be in the same direction (under similar viewpoints)
    if np.dot(src_best_line, tgt_best_line) < 0:
        tgt_best_line = -tgt_best_line
    contact_point = (int(max_xy_list[0][0]), int(max_xy_list[0][1]))
    if save_root:
        print('src & tgt best lines:\n', src_best_line, tgt_best_line)
        visualize_max_xy(save_root, src_pos_list[0], contact_point, src_img_PIL, tgt_img_PIL, heatmap=ori_cos_map[0])
        visualize_max_xy_list(save_root, src_pos_list_np, max_xy_list_np, src_img_PIL, tgt_img_PIL, filename='max_xy_list_all')
        visualize_max_xy_list(save_root, src_pos_inliers, max_xy_inliers, src_img_PIL, tgt_img_PIL)
        visualize_max_xy_linear(save_root, src_pos_list[0], src_best_line, contact_point, tgt_best_line, src_img_PIL, tgt_img_PIL)
    return contact_point, tgt_best_line

def transfer_affordance_w_mask(src_img_PIL, tgt_img_PIL, tgt_mask, prompt, src_pos_list, save_root=None, ftype='sd'):
    mask = torch.from_numpy(tgt_mask[...,0]).cuda() # h,w
    resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='nearest').squeeze().cuda() # h,w -> 1,IMG_SIZE,IMG_SIZE
    resized_mask = resized_mask.cpu().numpy()
    ori_cos_map = None
    max_xy_list = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    match_fts(src_ft, tgt_ft, src_pos_list[0], save_root)
    for src_pos_id, src_pos in enumerate(src_pos_list):
        cos_map = match_fts(src_ft, tgt_ft, src_pos) # 1,IMG_SIZE,IMG_SIZE
        if src_pos_id == 0:
            cos_map = cos_map * resized_mask
        if ori_cos_map is None:
            ori_cos_map = cos_map
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
    src_pos_list_np = np.array(src_pos_list)
    max_xy_list_np = np.array(max_xy_list)
    src_pos_inliers, src_best_line = fit_linear_ransac(src_pos_list_np, threshold=5, min_samples=10)
    max_xy_inliers, tgt_best_line = fit_linear_ransac(max_xy_list_np, threshold=5, min_samples=10)
    # src_best_line and tgt_best_line should be in the same direction (under similar viewpoints)
    if np.dot(src_best_line, tgt_best_line) < 0:
        tgt_best_line = -tgt_best_line
    contact_point = (int(max_xy_list[0][0]), int(max_xy_list[0][1]))
    if save_root:
        print('src & tgt best lines:\n', src_best_line, tgt_best_line)
        visualize_max_xy(save_root, src_pos_list[0], contact_point, src_img_PIL, tgt_img_PIL, heatmap=ori_cos_map[0])
        visualize_max_xy_list(save_root, src_pos_list_np, max_xy_list_np, src_img_PIL, tgt_img_PIL, filename='max_xy_list_all')
        visualize_max_xy_list(save_root, src_pos_inliers, max_xy_inliers, src_img_PIL, tgt_img_PIL)
        visualize_max_xy_linear(save_root, src_pos_list[0], src_best_line, contact_point, tgt_best_line, src_img_PIL, tgt_img_PIL)
    return contact_point, tgt_best_line

'''make sure the contact point is within the mask'''
def transfer_affordance_tt(src_img_PIL, tgt_img_PIL, tgt_mask, prompt, src_pos_list, save_root=None, ftype='sd'):
    mask = torch.from_numpy(tgt_mask[...,0]).cuda() # h,w
    resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='nearest').squeeze().cuda() # h,w -> 1,IMG_SIZE,IMG_SIZE
    resized_mask = resized_mask.cpu().numpy()
    ori_cos_map = None
    max_xy_list = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    match_fts(src_ft, tgt_ft, src_pos_list[0], save_root)
    for src_pos_id, src_pos in enumerate(src_pos_list):
        cos_map = match_fts(src_ft, tgt_ft, src_pos) # 1,IMG_SIZE,IMG_SIZE
        if src_pos_id == 0:
            cos_map = cos_map * resized_mask
        if ori_cos_map is None:
            ori_cos_map = cos_map
        max_xy, _ = sample_highest(cos_map)
        max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
        max_xy_list.append(max_xy)
    src_pos_list_np = np.array(src_pos_list)
    max_xy_list_np = np.array(max_xy_list)
    src_pos_inliers, src_best_line = fit_linear_ransac(src_pos_list_np, threshold=5, min_samples=10)
    max_xy_inliers, tgt_best_line = fit_linear_ransac(max_xy_list_np, threshold=5, min_samples=10)
    # tgt_best_line should be upwards
    if tgt_best_line[1] > 0:
        tgt_best_line = -tgt_best_line
    contact_point = (int(max_xy_list[0][0]), int(max_xy_list[0][1]))
    if save_root:
        print('src & tgt best lines:\n', src_best_line, tgt_best_line)
        visualize_max_xy(save_root, src_pos_list[0], contact_point, src_img_PIL, tgt_img_PIL, heatmap=ori_cos_map[0])
        visualize_max_xy_list(save_root, src_pos_list_np, max_xy_list_np, src_img_PIL, tgt_img_PIL, filename='max_xy_list_all')
        visualize_max_xy_list(save_root, src_pos_inliers, max_xy_inliers, src_img_PIL, tgt_img_PIL)
        visualize_max_xy_linear(save_root, src_pos_list[0], src_best_line, contact_point, tgt_best_line, src_img_PIL, tgt_img_PIL)
    return contact_point, tgt_best_line
    