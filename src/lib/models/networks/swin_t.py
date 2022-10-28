from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
from .swint.config import add_swint_config, add_swins_config, add_swinb_config, add_swinl_config
import cv2
from detectron2.config import get_cfg
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .swint.swin_transformer import build_swint_fpn_backbone, build_swint_backbone
from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

'''swin+fpn+up and swin+fpn'''
class SWIN_FPN(nn.Module):
    def __init__(self, down_ratio, cfg, heads, head_conv, in_chan, out_channel=0):
        super(SWIN_FPN, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))#first_level=2
        self.backbone = build_swint_fpn_backbone(cfg, in_chan)
        self.last_level = 5
        channels = [16, 32, 64, 128, 256, 512]
        final_kernel = 1
        self.heads = heads
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.sw_up = SWFPNUp(256, 256, [0,2,4])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.backbone(x)  # swint+fpn
        x_ = []
        for i, px in enumerate(list(x.keys())):
            # print(i, x[px].shape)
            x_.append(x[px])
        # print("self.backbone---------------------------------------\n")

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_[i].clone())
        # for i, lay in enumerate(y):
            # print(i, lay.shape)
        # print("y---------------------------------------\n")
        '''up'''
        self.sw_up(y, 0, len(y))

        z = {}
        '''torch.Size([1, 3, 136, 248])
            torch.Size([1, 2, 136, 248])
            torch.Size([1, 2, 136, 248])'''
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1]) #swint+fpn+up
            # z[head] = self.__getattr__(head)(y[0])# swint+up
            # print(z[head].shape)
        return [z]

'''swin+up'''
class SWIN(nn.Module):
    def __init__(self, down_ratio, cfg, heads, head_conv, in_chan, out_channel=0):
        super(SWIN, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))#first_level=2
        self.backbone = build_swint_backbone(cfg, in_chan)
        self.last_level = 5
        channels = [16, 32, 64, 128, 256, 512]
        heads_chans = {'stage2':96,"stage3":192,"stage4":384,"stage5":768}
        final_kernel = 1
        self.heads = heads
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.sw_up = SWUp(heads_chans, [0,2,4])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.backbone(x)  # swint
        x_ = []
        for i, px in enumerate(list(x.keys())):
            # print(i, x[px].shape)
            x_.append(x[px])
        # print("self.backbone---------------------------------------\n")

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x_[i].clone())
        # for i, lay in enumerate(y):
            # print(i, lay.shape)
        # print("y---------------------------------------\n")
        self.sw_up(y, 0, len(y))#up

        z = {}
        '''torch.Size([1, 3, 136, 248])
            torch.Size([1, 2, 136, 248])
            torch.Size([1, 2, 136, 248])'''
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
            # print(z[head].shape)
        return [z]

'''swin'''
class SWIN_NONE(nn.Module):
    def __init__(self, down_ratio, cfg, heads, head_conv, in_chan, out_channel=0):
        super(SWIN_NONE, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))#first_level=2
        self.backbone = build_swint_backbone(cfg, in_chan)
        self.last_level = 5
        channels = [16, 32, 64, 128, 256, 512]
        heads_chans = {'stage2':96,"stage3":192,"stage4":384,"stage5":768}
        final_kernel = 1
        self.heads = heads
        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.conv_chans_up = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=True),
                            nn.ConvTranspose2d(256, 256,  8* 2, stride=8, padding=8 // 2, output_padding=0,
                                groups=256, bias=False))

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(256, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.backbone(x)  # swint
        x_ = []
        for i, px in enumerate(list(x.keys())):
            # print(i, x[px].shape)
            x_.append(x[px])
        # print("self.backbone---------------------------------------\n")

        y = self.conv_chans_up(x_[-1])

        z = {}
        '''torch.Size([1, 3, 136, 248])
            torch.Size([1, 2, 136, 248])
            torch.Size([1, 2, 136, 248])'''
        for head in self.heads:
            z[head] = self.__getattr__(head)(y)
            # print(z[head].shape)
        return [z]


class SWFPNUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(SWFPNUp, self).__init__()
        for i in range(1, len(up_f)):
            proj = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=True)
            node = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=True)

            up = nn.ConvTranspose2d(256, 256, up_f[i] * 2, stride=up_f[i], padding=up_f[i] // 2, output_padding=0, groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])

class SWUp(nn.Module):
    def __init__(self,channels, up_f):
        super(SWUp, self).__init__()
        for i in range(1, len(up_f)):
            proj = nn.Conv2d(channels[f'stage{i+2}'], 256, kernel_size=3, stride=1,padding=1, bias=True)
            proj2 = nn.Conv2d(channels[f'stage{i+1}'], 256, kernel_size=3, stride=1,padding=1, bias=True)

            node = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1, bias=True)

            '''替换IDAUP模块'''
            up = nn.ConvTranspose2d(256, 256, up_f[i] * 2, stride=up_f[i], padding=up_f[i] // 2, output_padding=0, groups=256, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'proj2_' + str(i), proj2)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            project2 = getattr(self, 'proj2_' + str(i - startp))
            node = getattr(self, 'node_' + str(i - startp))
            if i == 1:
                layers[i] = node(layers[i]+project2(layers[i-1]))
            else:
                layers[i] = node(layers[i] + layers[i - 1])

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

'''best'''
def get_swin_fpn_net(heads, down_ratio=4,  head_conv=256, cfg_name ="centernet_swint_T_FPN.yaml",input_channels=3):
  cfg_dir = "/home/server/xcg/CenterNet/src/lib/models/networks/swint/cfgs/"+ cfg_name
  cfg = get_cfg()
  add_swint_config(cfg)
  # add_swins_config(cfg)
  cfg.merge_from_file(cfg_dir)
  model = SWIN_FPN(heads=heads,head_conv=head_conv,down_ratio=down_ratio,cfg=cfg,in_chan=input_channels)
  return model

def get_swin_net(heads, down_ratio=4,  head_conv=256, cfg_name ="centernet_swint_T_FPN.yaml",input_channels=3):
  cfg_dir = "/home/server/xcg/CenterNet/src/lib/models/networks/swint/cfgs/"+ cfg_name
  cfg = get_cfg()
  add_swint_config(cfg)
  cfg.merge_from_file(cfg_dir)
  model = SWIN(heads=heads, head_conv=head_conv, down_ratio=down_ratio, cfg=cfg, in_chan=input_channels)
  return model

def get_swin_none_net(heads, down_ratio=4,  head_conv=256, cfg_name ="centernet_swint_T_FPN.yaml",input_channels=3):
  cfg_dir = "/home/server/xcg/CenterNet/src/lib/models/networks/swint/cfgs/"+ cfg_name
  cfg = get_cfg()
  add_swint_config(cfg)
  cfg.merge_from_file(cfg_dir)
  model = SWIN_NONE(heads=heads, head_conv=head_conv, down_ratio=down_ratio, cfg=cfg, in_chan=input_channels)
  return model