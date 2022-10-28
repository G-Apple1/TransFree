from detectron2.data import detection_utils as utils
import torch
import numpy as np
import copy
import os
import random
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.structures import BoxMode as boxmode
import math

def mapper(dataset_dict):
    data_dict = copy.deepcopy(dataset_dict)#独立复制
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="RGB")#"BGR"
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    h = image.shape[0]//2
    w = image.shape[1]//2
    transform = T.Resize((h,w))(auginput)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in data_dict.pop("annotations")
    ]
    data_dict["image"] = image
    data_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])

    return data_dict

def mosaic_mixup(dataset_dict,dataset_dicts):
    data_dict = copy.deepcopy(dataset_dict)  # 独立复制

    image, annos = load_mosaic(data_dict, 960, 460, dataset_dicts)
    if random.random() < 0.2:
        mip = random.sample(dataset_dicts, 1)[0]
        image, annos = mixup(image, annos, *load_mosaic(mip, 960, 460,dataset_dicts))
    if random.random() < 0.5:
        brg = random.randint(-50, 50)
        image = np.uint8(np.clip((cv2.add(image, brg)), 0, 255))

    # Plot
    # for lab in annos:
    #     # print("plot??")
    #     cv2.rectangle(image,(lab[1],lab[2]),(lab[3],lab[4]),(0,255,0),thickness=2)
    #     cv2.putText(image,"cls: %s"%lab[0],(lab[1],lab[2]-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
    # cv2.imwrite(f"/home/server/xcg/CenterNet/src/lib/datasets/mosaics/mos_mix_{data_dict['image_id']}.jpg", image)

    annotations = []
    for an in annos:#xyxy
        #xyxy2xywh
        bbox_1 = boxmode.convert(list(an[1:]),boxmode.XYXY_ABS,boxmode.XYWH_ABS)
        annotations.append({"iscrowd":0,"bbox":bbox_1,"category_id":an[0]+1})#,"bbox_mode":boxmode.XYWH_ABS //classes=3
        # annotations.append({"iscrowd":0,"bbox":bbox_1,"category_id":1})#classes=1


    data_dict["image"] = image
    # data_dict["height"] = image.shape[0]
    # data_dict["width"] = image.shape[1]
    # data_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])
    data_dict["annotations"] = annotations

    return data_dict

def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def load_mosaic(dataset_dict, s,inds,dataset_dicts):
    # loads images in a 4-mosaic
    # s:　img_size　(limit)
    # inds: 图片数量
    img_size = s

    labels4 = []
    mosaic_border = [-img_size // 2, -img_size // 2]
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y

    indices = [dataset_dict] + [d for d in random.sample(dataset_dicts, 3)]# 3 additional image indices

    for i, index in enumerate(indices):

        # Load image
        img = utils.read_image(index["file_name"], format="RGB")
        h,w = img.shape[0],img.shape[1]

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        bboxs = [anno["bbox"] for anno in index["annotations"]]#cate_id+bbox[4]
        bboxs = boxmode.convert(np.array(bboxs), 1, 0)
        ids = np.array([[anno["category_id"]] for anno in index["annotations"]])

        labels = np.hstack((ids,bboxs))

        if labels.size:
            labels[:, 1:] = xyxy_pad(labels[:, 1:], padw=padw, padh=padh)  # xyxy to pixel xyxy_pad format

        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:]):
    #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=0.0,
                                       translate=0.05,
                                       scale=0.3,
                                       shear=0.0,
                                       perspective=0.0,
                                       border=mosaic_border) # border to remove

    return img4, labels4

def xyxy_pad(x, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] + padw  # top left x
    y[:, 1] = x[:, 1] + padh  # top left y
    y[:, 2] = x[:, 2] + padw  # bottom right x
    y[:, 3] = x[:, 3] + padh  # bottom right y
    return y

def random_perspective(im, targets=(), degrees=10, translate=.1, scale=.1,
                       shear=10, perspective=0.0, border=(0, 0)):
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:# warp boxes
        new = np.zeros((n, 4))
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr) # candidates
