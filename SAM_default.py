import os
import sys
import PathCreator
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import pickle
import math
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from ListFiles import GetFiles
from ShpMaskWriter import mask_write, mask_write_treads

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
exit()

Path0 = "includes/Pictures"
# FileList = sorted(GetFiles(Path0))


# FileName = FileList[i_num]
# FileName = "B21-166b_cut.tif"

i_num = 0
FileName = "houses2.jpg"

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
device = "cpu"
print("Current device: " + device)

# model inicialization
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

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
