# -*- coding: utf-8 -*-
# https://youtu.be/fVeW9a6wItM
"""

@author: Digitalsreeni (Sreenivas Bhattiprolu)

First make sure pytorch and torchcvision are installed, for GPU
In my case: pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

OR download the repo locally and install
and:  pip install -e .

Download the default trained model: 
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Other models are available:
    https://github.com/facebookresearch/segment-anything#model-checkpoints

"""
# Tested on python 3.9.16

import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import time
import pickle
import math

Path_Add = os.path.abspath("includes")
sys.path.insert(0, Path_Add)
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

Path_Add = os.path.abspath("includes/segment-anything")
sys.path.insert(0, Path_Add)
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

Path_Add = os.path.abspath("includes/pyshp-master")
sys.path.insert(0, Path_Add)


Path0 = "includes/Pictures"
FileList = sorted(GetFiles(Path0))
print(FileList)

i_num = 0
# FileName = FileList[i_num]
FileName = "B21-166b_cut.tif"
# FileName = "houses2.jpg"

Path = Path0 + "/" + FileName
image = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
CropNLayers = int(math.log2(max(image.shape[1:2])) - 9)
CropNLayers = (CropNLayers >= 0) * CropNLayers

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

if (torch.cuda.is_available() == False):
    device = "cpu"
else:
    device = "cuda"
print("Current device: " + device)
#1
print(1)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# There are several tunable parameters in automatic mask generation that control
# how densely points are sampled and what the thresholds are for removing low 
# quality or duplicate masks. Additionally, generation can be automatically 
# run on crops of the image to get improved performance on smaller objects, 
# and post-processing can remove stray pixels and holes. 
# Here is an example configuration that samples more masks:
# https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35

# Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
# and 0.92 and 0.96 for score_thresh



mask_generator_ = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32, #32
    points_per_batch=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers = 0, #CropNLayers,
    crop_overlap_ratio=512/1500,
    crop_n_points_downscale_factor = 2,
    min_mask_region_area=0,  # Requires open-cv to run post-processing
)


"""
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crops_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crops_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
"""


masks = mask_generator_.generate(image)
with open("mask.pkl", "wb") as fp:
    pickle.dump(masks, fp)
"""
print("load mask start")
with open("file.pkl", "rb") as fp:
    masks = pickle.load(fp)
print("load mask finish")
"""

def show_anns(ax, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = np.uint8(ann['segmentation'])
        img = np.ones((m.shape[0], m.shape[1], 3))
        temp = np.ones((m.shape[0], m.shape[1], 4), dtype=np.uint8) * 0.0
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))
        contours, hierarchy = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2image = cv2.drawContours(temp, contours, -1, (0, 0, 0, 1), 2)
        ax.imshow(cv2image)


"""
fig = plt.figure(figsize=(10,5))
ax = [fig.add_subplot(1, 2, 1),
      fig.add_subplot(1, 2, 2)]
ax[0].imshow(image)
ax[1].imshow(image)
show_anns(ax[1], masks)
ax[0].axis('off')
ax[1].axis('off')
plt.show()
"""

mask_write_treads("includes/Shape/Shape" + "{}".format(i_num), masks)

exit()

"""
Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

segmentation : the mask
area : the area of the mask in pixels
bbox : the boundary box of the mask in XYWH format
predicted_iou : the model's own prediction for the quality of the mask
point_coords : the sampled input point that generated this mask
stability_score : an additional measure of mask quality
crop_box : the crop of the image used to generate this mask in XYWH format
"""
