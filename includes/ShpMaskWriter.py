import os
import cv2
import shapefile
import numpy as np
import threading


def get_mask2polygon(mask, bbox = [0,0,0,0]):
    polygon = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour, h_vec in zip(contours, hierarchy[0]):
        c = contour.reshape((contour.shape[0], 2)) * [1, -1] + [bbox[0], -bbox[1]]
        c = c.tolist()
        c.reverse()
        if len(c) > 2:
            polygon.append(c)
    return polygon


def mask_write(Path, masks):
    print("Start shapefile gen")
    w = shapefile.Writer(Path, shapefile.POLYGON)
    w.field("NAME", "C")
    for mask in masks:
        polygon = get_mask2polygon(
            np.uint8(mask['segmentation'][
                     mask['bbox'][1]:mask['bbox'][1]+mask['bbox'][3]+1,
                     mask['bbox'][0]:mask['bbox'][0]+mask['bbox'][2]+1]),
            mask['bbox'])
        w.poly(polygon)
        w.record("Polygon")
    w.close()
    print("Finish shapefile gen")


def Treadfunc(mask,plist, i):
    polygon = get_mask2polygon(
        np.uint8(mask['segmentation'][
                 mask['bbox'][1]:mask['bbox'][1] + mask['bbox'][3] + 1,
                 mask['bbox'][0]:mask['bbox'][0] + mask['bbox'][2] + 1]),
        mask['bbox'])
    plist[i] = polygon


def mask_write_treads(Path, masks):
    print("Start shapefile gen")
    w = shapefile.Writer(Path, shapefile.POLYGON)
    w.field("NAME", "C")
    plist = [None] * len(masks)
    tlist = [None] * len(masks)
    for i in range(len(masks)):
        tlist[i] = threading.Thread(target=Treadfunc, args=(masks[i], plist, i,))
        tlist[i].start()

    for i in range(len(masks)):
        tlist[i].join()
        w.poly(plist[i])
        w.record("Polygon")
    w.close()
    print("Finish shapefile gen")


