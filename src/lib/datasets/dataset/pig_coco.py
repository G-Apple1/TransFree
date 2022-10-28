from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from detectron2.data.datasets import register_coco_instances

import torch.utils.data as data

class PIG_COCO(data.Dataset):
  num_classes = 3 ##         #modify category
  default_resolution = [960, 540]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(PIG_COCO, self).__init__()
    self.data_dir = '/home/server/xcg/CenterNet/data'        #os.path.join(opt.data_dir,' 'litchi')
    self.test_img_dir = '/home/server/xcg/CenterNet/data/test/images'    #os.path.join(self.data_dir, 'images')
    self.train_img_dir = '/home/server/xcg/CenterNet/data/train-aug/images'    #os.path.join(self.data_dir, 'images')
    if split == 'val':
      self.annot_path = os.path.join(
          self.data_dir, 'test',
          'pig_annotation_test.json').format(split)##
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'train.json').format(split)
      if split == 'test':
        self.annot_path = os.path.join(
          self.data_dir, 'test',
          'pig_annotation_test.json').format(split)##
      else:#train
        self.annot_path = os.path.join(
          self.data_dir, 'train-aug', 'pig_annotation_train_aug.json').format(split)##_aug
    self.max_objs = 50
    self.class_name = ['__background__', ' 1', '2', '3']##
    self._valid_ids = [1]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing coco pig {} data.'.format(split))
    '''
    anns
    catToImgs
    cats
    dataset
    imgToAnns
    imgs
    '''
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    if split == 'train':
      register_coco_instances('pig_coco_train', {}, self.train_img_dir+'/../pig_annotation_train_aug.json', self.train_img_dir)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        #category_id = self._valid_ids[cls_ind - 1]
        category_id = cls_ind
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    # print(save_dir)
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))

    from .confusion_matrix import ConfusionMatrix, xywh2xyxy, process_batch, ap_per_class
    import torch
    C_M = ConfusionMatrix(nc=self.num_classes, conf=0.5, iou_thres=0.5)
    stats = []
    image_ids = [i for i in self.coco.imgToAnns]
    for i in image_ids:  # 49张图
      bbox_gt = np.array([y['bbox'] for y in self.coco.imgToAnns[i]])
      class_gt = np.array([[y['category_id'] - 1] for y in self.coco.imgToAnns[i]])
      labels = np.hstack((class_gt, bbox_gt))
      labels = np.array(labels, dtype=np.float)

      bbox_dt = np.array([y['bbox'] for y in coco_dets.imgToAnns[i]])
      conf_dt = np.array([[y['score']] for y in coco_dets.imgToAnns[i]])
      class_dt = np.array([[y['category_id'] - 1] for y in coco_dets.imgToAnns[i]])
      predictions = np.hstack((np.hstack((bbox_dt, conf_dt)), class_dt))

      C_M.process_batch(predictions, labels)

      '''PR等曲线'''
      # detects = torch.tensor(xywh2xyxy(predictions))
      # labs = torch.tensor(np.hstack((labels[:, 0][:, None], xywh2xyxy(labels[:, 1:]))))
      # iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
      # correct = process_batch(detects, labs, iouv)
      # tcls = labs[:, 0].tolist()  # target class
      # stats.append((correct.cpu(), detects[:, 4].cpu(), detects[:, 5].cpu(), tcls))

    R,P,F1= C_M.print()

    if "best" in self.opt.load_model:
      _dir = os.path.dirname(self.opt.load_model)+"/best_"
    elif "last" in self.opt.load_model:
      _dir = os.path.dirname(self.opt.load_model)+"/last_"
    else:
      _dir = os.path.dirname(self.opt.load_model)+f"/{os.path.basename(self.opt.load_model)[5:]}_"

    plot_dir = _dir#"/home/server/xcg/CenterNet/exp/ctdet/"

    names = {0: "lc"}
    # stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # if len(stats) and stats[0].any():
    #   p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=plot_dir, names=names)
    '''plot confusion matrix'''
    # C_M.plot(save_dir=plot_dir + 'confu_mat_rec_{%.4f}.png'%F1, names=['Ventral lying','Lateral lying','Standing'], rec_or_pred=0)
    # C_M.plot(save_dir=plot_dir + 'confu_mat_pred_{%.4f}.png'%F1, names=['Ventral lying','Lateral lying','Standing'], rec_or_pred=1)

    '''coco metrics'''
    # coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    # coco_eval.evaluate()#每张图算出结果
    # coco_eval.accumulate()#对每张图的结果做统计
    # coco_eval.summarize()
