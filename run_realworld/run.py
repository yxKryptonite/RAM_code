import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
from run_realworld.env import MiniEnv
import numpy as np
from run_realworld.utils import read_yaml_config
from vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
import torch
from PIL import Image
import glob, time
from vision.featurizer.run_featurizer import transfer_affordance
from vision.featurizer.utils.visualization import IMG_SIZE
import argparse
import traceback
import matplotlib
matplotlib.use('svg') # NOTE: fix backend error while GPU is in use
from tqdm import tqdm
import shutil
import random
import open3d as o3d

def backup(args, cfgs):
    shutil.copyfile(f"run_realworld/{args.config}", f"{cfgs['SAVE_ROOT']}/config.yaml")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    cfgs = read_yaml_config(f"run_realworld/{args.config}")
    os.makedirs(cfgs['SAVE_ROOT'], exist_ok=True)
    backup(args, cfgs)
    torch.set_printoptions(precision=4, sci_mode=False)
            
    instruction = cfgs['instruction']
    obj = cfgs['obj']
    prompt = cfgs['prompt']
    data_source = cfgs.get("DATA_SOURCE", "droid")
    
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    gym = MiniEnv(cfgs, grounded_dino_model, sam_predictor) 

    save_root = cfgs['SAVE_ROOT']
    points = rgb = None
    pcd = o3d.io.read_point_cloud("run_realworld/real_data/input/pcd.ply")
    points = np.array(pcd.points)
    rgb = Image.open("run_realworld/real_data/input/rgb.png")
        
    tgt_img_PIL = rgb
    tgt_img_PIL.save(f"{save_root}/tgt_img.png")
    rgb = np.array(rgb)
    
    tgt_masks = inference_one_image(np.array(tgt_img_PIL), grounded_dino_model, sam_predictor, box_threshold=cfgs['box_threshold'], text_threshold=cfgs['text_threshold'], text_prompt=obj, device="cuda").cpu().numpy() # you can set point_prompt to traj[0]
    tgt_mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    # if mask is false, make it white
    tgt_img_masked = np.array(tgt_img_PIL) * tgt_mask + 255 * (1 - tgt_mask)
    # tgt_img_masked, _, _ = crop_image(tgt_img_masked, tgt_mask)
    tgt_img_PIL = Image.fromarray(tgt_img_masked).convert('RGB')
    tgt_img_PIL.save(f"{save_root}/tgt_img_masked.png")
    
    ######## src
    ####################### SOURCE DEMONSTRATION ########################
    data_dict = np.load("run_realworld/real_data/demonstration/data.pkl", allow_pickle=True)
    traj = data_dict['traj']
    src_img_np = data_dict['masked_img']
    src_img_PIL = Image.fromarray(src_img_np).convert('RGB')
    src_img_PIL.save(f"{save_root}/src_img.png")
    ####################### SOURCE DEMONSTRATION ########################

    # scale cropped_traj to IMG_SIZE
    src_pos_list = []
    for xy in traj:
        src_pos_list.append((xy[0] * IMG_SIZE / src_img_PIL.size[0], xy[1] * IMG_SIZE / src_img_PIL.size[1]))
    
    while True:
        try:
            contact_point, post_contact_dir = transfer_affordance(src_img_PIL, tgt_img_PIL, prompt, src_pos_list, save_root=save_root, ftype='sd')
            break
        except Exception as transfer_e:
            traceback.print_exc()
            print('[ERROR] in transfer_affordance:', transfer_e)

    # contact point + post-contact direction
    ret_dict = gym.test_keypoint(rgb, pcd, contact_point, post_contact_dir)
    
    print("3D Affordance:\n", ret_dict)
    
    print("====== DONE ======")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config file') # e.g. configs/drawer_open.yaml
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    
    main(args)
