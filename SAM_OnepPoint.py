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
from typing import List, Any

# Tested on python 3.9.16

import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import matplotlib.backend_bases
import cv2
import sys
import os
import time
import pickle

Path_Add = os.path.abspath("includes")
sys.path.insert(0, Path_Add)
from ListFiles import GetFiles

Path_Add = os.path.abspath("includes/segment-anything")
sys.path.insert(0, Path_Add)
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

Path_Add = os.path.abspath("includes/pyshp-master")
sys.path.insert(0, Path_Add)
import shapefile

Path0 = "includes/Pictures/"
FileList = sorted(GetFiles(Path0))

# FileName = "houses2.jpg"
FileName = "B21-166b_cut.tif"
# FileName = "B21-50a_cut.tif"
Path = Path0 + "/" + FileName

image = cv2.imread(Path)  # Try houses.jpg or neurons.jpg
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #temp = np.ones((mask.shape[0], mask.shape[1], 4), dtype=np.uint8) * 0.0
    #cv2image = cv2.drawContours(temp, contours, -1, (0, 0, 0, 1), 2)
    return ax.imshow(mask_image)#, ax.imshow(cv2image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

from ShpMaskWriter import get_mask2polygon
def Mask_write(Path, masks):
    print("Start shapefile gen")
    w = shapefile.Writer(Path, shapefile.POLYGON)
    w.field("NAME", "C")
    for mask in masks:
        polygon = get_mask2polygon(np.uint8(mask))
        w.poly(polygon)
        w.record("Polygon")
    w.close()
    print("Finish shapefile gen")


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

if (torch.cuda.is_available() == False):
    device = "cpu"
else:
    device = "cuda"
print("Current device: " + device)


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

is_Save = False

if is_Save:
    predictor.set_image(image)
    predictor.save_image_embedding("Predictor_" + FileName + ".emb")
else:
    predictor.load_image_embedding("Predictor_" + FileName + ".emb")

def get_masks(x, y):
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks, scores, logits


class Handler:
    def __init__(self, fig, predictor):
        # Constructor
        self.predictor = predictor
        self.keys_keyboard = {'shift': False, 'control': False, 'enter': False}
        self.masks = []
        self.i = 0
        fig.canvas.mpl_connect("button_press_event", self.mouse_press)
        fig.canvas.mpl_connect('key_press_event', self.keyboard_press)
        fig.canvas.mpl_connect('key_release_event', self.keyboard_release)

    def keyboard_release(self, event: mlp.backend_bases.KeyEvent):
        try:
            self.keys_keyboard[event.key] = False
        except KeyError:
            pass
        if (event.key == 'enter'):
            Mask_write("includes/Shape_test/Shape_test", [self.masks[2]])
        # print('press', event.key)

    def keyboard_press(self, event: mlp.backend_bases.KeyEvent):
        try:
            self.keys_keyboard[event.key] = True
        except KeyError:
            pass
        # print('press', event.key)

    def mouse_press(self, event: mlp.backend_bases.MouseEvent):
        axes = event.inaxes
        # Если кликнули вне какого-либо графика, то не будем ничего делать
        if axes is None:
            return
        # Координаты клика в системе координат осей
        x = event.xdata
        y = event.ydata
        self.masks, scores, logits = get_masks(x, y)
        axim1 = show_mask(self.masks[0], axes)
        axes.figure.canvas.draw()
        axim1.remove()
        #axim2.remove()

#x = 241
#y = 167
#masks, scores, logits = get_masks(x, y)
#with open("file.pkl", "wb") as fp:
#    pickle.dump(masks, fp)

#with open("file.pkl", "rb") as fp:
#    masks = pickle.load(fp)
#Mask_write("includes/Shape_test/Shape_test", [masks[2]])


fig = plt.figure(figsize=(6, 6))
axes = fig.add_axes((0.1, 0.1, 0.8, 0.8))
axes.imshow(image)
hd = Handler(fig, predictor)
plt.show()


exit()
