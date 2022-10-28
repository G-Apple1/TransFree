from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import math
import os

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 1000
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.opt.fix_res:
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.opt.flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, file,meta=None):
        print("----xuxuuxuux!!!!---")
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            print("type(image_or_path_or_tensor) == type ('')")
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        hs = math.ceil( image.shape[0]/540 )#多少行（浮点数向上取整）
        ws = math.ceil( image.shape[1]/768 )#多少列
        h_c = image.shape[0]%540
        w_c = image.shape[1]%768
        print(image.shape[0],image.shape[1],h_c,w_c)
        image = cv2.copyMakeBorder(image, 0, 540-h_c, 0, 768-w_c, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        print("图片可以裁剪成 %d X %d 个子图"%(ws,hs))
        for scale in self.scales:
            index = 0
            for row in range(hs):#按行
                #print(row)
                for col in range(ws):#按列
                    fruits_num = []
                    index = index+1
                    print(col)
                    image_new = image[540*row:540*(row+1),768*col:768*(col+1)]

                    scale_start_time = time.time()

                    images, meta = self.pre_process(image_new, scale, meta)

                    images = images.to(self.opt.device)
                    torch.cuda.synchronize()
                    pre_process_time = time.time()
                    pre_time += pre_process_time - scale_start_time
                    output, dets, forward_time = self.process(images, return_time=True)  #

                    torch.cuda.synchronize()
                    net_time += forward_time - pre_process_time
                    decode_time = time.time()
                    dec_time += decode_time - forward_time

                    if self.opt.debug >= 2:
                        self.debug(debugger, images, dets, output, scale)

                    dets = self.post_process(dets, meta, scale)
                    torch.cuda.synchronize()
                    post_process_time = time.time()
                    post_time += post_process_time - decode_time

                    #detections.append(dets)
                    fruits_num.append(dets)

                    results = self.merge_outputs(fruits_num)#xu_ctdet.py nms
                    torch.cuda.synchronize()
                    end_time = time.time()
                    merge_time += end_time - post_process_time
                    tot_time += end_time - start_time

                    if self.opt.debug >= 1:
                        save_dir = '/media/scau2/1T1/lsq/CenterNet/output5/crop_img/detect/' + \
                                   os.path.basename(image_or_path_or_tensor)[:-4]+str(index)+".jpg"
                        print(save_dir)
                        #cv2.imwrite(save_dir, image_new)

                        #count = len(np.where(results[1][:, 4] > self.opt.vis_thresh)[0])
                        #file.write(str(index)+" "+str(count) + '\n')

                        self.show_results(debugger, image_new, results,image_or_path_or_tensor,index,file)
                        # assert self.opt.debug <= 1,"overover!!!"

        # return {'results': results, 'tot': tot_time, 'load': load_time,
        #         'pre': pre_time, 'net': net_time, 'dec': dec_time,
        #         'post': post_time, 'merge': merge_time}