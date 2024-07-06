import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append("./vision")

# Grounding DINO
import GroundedSAM.GroundingDINO.groundingdino.datasets.transforms as T
from GroundedSAM.GroundingDINO.groundingdino.models import build_model
from GroundedSAM.GroundingDINO.groundingdino.util import box_ops
from GroundedSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundedSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from GroundedSAM.segment_anything.segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_image_from_array(image):
    # load image
    image_pil = Image.fromarray(image).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def inference_one_image(cam_image, model, predictor, box_threshold, text_threshold,
                        text_prompt, device, point_prompt=None):
    
    # load image
    image_pil, image = load_image_from_array(cam_image)
    
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    image =cam_image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_box = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])[:1,...].to(device)
    if point_prompt is not None:
        point_coords = torch.tensor(point_prompt).unsqueeze(0).unsqueeze(0).to(device)
        point_labels = torch.ones_like(point_coords[:, :, 0]).long().to(device)
        # print(point_coords.shape, point_labels.shape)
    else:
        point_coords = point_labels = None
    
    try:
        masks, _, _ = predictor.predict_torch(
            point_coords = point_coords,
            point_labels = point_labels,
            boxes = transformed_box,
            multimask_output = False
        )
    except:
        print(f'=> No bbox for {text_prompt} in this image')
        masks = torch.ones((1, 1, image.shape[0], image.shape[1])).cuda()

    return masks

def prepare_GroundedSAM_for_inference(sam_version, sam_checkpoint, grounded_checkpoint,
                                      config, device):
    
    model = load_model(config, grounded_checkpoint, device=device)
    
    # initialize SAM
    # if use_sam_hq:
    #     predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    # else:
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    return model, predictor

def prepare_gsam_model(device):
    sam_version = "vit_h"
    # download from GroundedSAM
    sam_checkpoint = "assets/ckpts/sam_vit_h_4b8939.pth"
    grounded_checkpoint = "assets/ckpts/groundingdino_swint_ogc.pth"
    config = "vision/GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    grounded_dino_model, sam_predictor = prepare_GroundedSAM_for_inference(
        sam_version=sam_version, sam_checkpoint=sam_checkpoint,
        grounded_checkpoint=grounded_checkpoint, config=config, device=device)
    return grounded_dino_model, sam_predictor

def get_bounding_box(binary_array):
    # Find the rows and columns where there are non-zero elements
    rows = np.any(binary_array, axis=1)
    cols = np.any(binary_array, axis=0)
    
    # Find the indices of the first and last non-zero row and column
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def crop_image(img, mask, traj=[], margin=50):
    '''
    crop img based on the seg mask, and transform the src_pos_list to the cropped image space
    @param
    :img: np.array, (h, w, 3)
    :mask: np.array, (h, w, 3) # the last channel is repeated 3 times
    :src_pos_list: list of tuple, [(x1, y1), ...]
    '''
    bbox = get_bounding_box(mask[..., 0])
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img.shape[1], x_max + margin)
    y_max = min(img.shape[0], y_max + margin)
    
    cropped_img = img[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_traj = []
    for p in traj:
        x, y = p
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue
        x = x - x_min
        y = y - y_min
        cropped_traj.append((x, y))
    return cropped_img, cropped_mask, cropped_traj
