import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

import matplotlib.pyplot as plt



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    # custom
    # this is the default automatic mask generator for an input ENTIRE image. You may check the official document in notebook form: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    downsample_manually = False
    if args.downsample == "1" or args.downsample_type == 'mask':
        IMAGE_DIR = os.path.join(args.image_root, 'images')
    else:
        IMAGE_DIR = os.path.join(args.image_root, 'images_'+str(args.downsample))
        if not os.path.exists(IMAGE_DIR):
            IMAGE_DIR = os.path.join(args.image_root, 'images')
            downsample_manually = True
            print("No downsampled images, do it manually.")

    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(args.image_root, 'sam_masks')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Extracting SAM segment everything masks...")
    
    for path in tqdm(sorted(os.listdir(IMAGE_DIR))):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        if downsample_manually:
            img = cv2.resize(img,dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        masks = mask_generator.generate(img) # the masks are now generated. 
        # Reference: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py


        # What is the returned masks:
        # list(dict(str, any)): A list over records for masks. Each record is a dict containing the following keys:
        #     segmentation (dict(str, any) or np.ndarray): The mask. If output_mode='binary_mask', is an array of shape HW. Otherwise, is a dictionary containing the RLE.
        #     bbox (list(float)): The box around the mask, in XYWH format.
        #     area (int): The area in pixels of the mask.
        #     predicted_iou (float): The model's own prediction of the mask's
        #       quality. This is filtered by the pred_iou_thresh parameter.
        #     point_coords (list(list(float))): The point coordinates input
        #       to the model to generate this mask.
        #     stability_score (float): A measure of the mask's quality. This
        #       is filtered on using the stability_score_thresh parameter.
        #     crop_box (list(float)): The crop of the image used to generate
        #       the mask, given in XYWH format.



        # print(len(masks)) # which basically how many masks are marked.

        ## PART A： saving the masks in  sam_masksk folder
        ## creating the masks in form of '.pt'
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if args.downsample_type == 'mask':
                m_score = torch.nn.functional.interpolate(m_score.unsqueeze(0).unsqueeze(0), size=(img.shape[0] // args.downsample, img.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                m_score = m_score.bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        masks = torch.stack(mask_list, dim=0)
        torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt')) # so the final `maks`



        # # PART B： saving the auto_mask png in  original image folder， it is for use of study, not necessary for the code by itself.
        # # creating images registering the masks
        # print(len(masks))
        # print(masks[0].keys())
        # plt.figure(figsize=(20,20))
        # plt.imshow(img)
        # show_anns(masks)
        # plt.axis('off')
        # plt.savefig(os.path.join(IMAGE_DIR, 'auto_masked_'+name+'.png'))

