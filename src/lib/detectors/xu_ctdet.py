from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .xu_base_detect import BaseDetector


class CtdetDetector_xu(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector_xu, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]  # 加载模型
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        print("\n===========debug========")
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results,image_or_path_or_tensor,index,file):
        print("========xu_show_results=========")
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            count = len(np.where(results[j][:, 4] > self.opt.vis_thresh)[0])
            file.write(str(index) + " " + str(count) + '\n')
            # txt_path = '/media/scau2/1T1/lsq/CenterNet/output5/fruit_num.txt'
            # f = open(txt_path, 'a', encoding='utf-8')
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    #print(bbox[:4])
                    # old_img = cv2.imread(image_or_path_or_tensor)
                    # crop_img = old_img[int(bbox[1]):int(bbox[3]),int(max(bbox[0],0)):int(bbox[2])]
                    # save_dir='/media/scau2/1T1/lsq/CenterNet/output5/crop_img/detect/%d'%index+ os.path.basename(image_or_path_or_tensor)
                    # print(save_dir)
                    # cv2.imwrite(save_dir,image)
                    # f.write(str(bbox[0])+" "+str(bbox[1])+" "+str(bbox[2])+" "+str(bbox[3]) + '\n')
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet', count=count)
    # debugger.show_all_imgs(pause=self.pause)
    debugger.save_all_imgs(path='/media/scau2/1T1/lsq/CenterNet/output5/crop_img/result_2/', genID=True)









