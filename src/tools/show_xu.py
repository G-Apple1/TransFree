from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]

    txt_path = '/media/scau2/1T1/lsq/CenterNet/output5/fruit_num.txt'
    if os.path.exists(txt_path):
        os.remove(txt_path)
    f = open(txt_path, 'a', encoding='utf-8')

    for (image_name) in image_names:
        print("\n",image_name)
        ret = detector.run(image_name,f)

if __name__ == '__main__':
    opt = opts().init()
    opt.task = 'xu_ctdet'
    '''果树框'''
    #opt.demo = "/media/scau2/1T1/lsq/CenterNet/single_pic"
    #opt.load_model = "/media/scau2/1T1/lsq/CenterNet/exp/ctdet/ctdet_coco_dla_2x/model_best.pth"#果树的框
    '''果检测'''
    opt.demo = "/media/scau2/1T1/lsq/CenterNet/output5/crop_img"#扣图
    opt.load_model = "/media/scau2/1T1/lsq/CenterNet/exp/ctdet/model_save/model_best.pth"#果的检测
    demo(opt)

