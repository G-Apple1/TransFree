from Evison import Display, show_network
from torchvision import models
import argparse
import cv2
import numpy as np
import torch
import glob
import os
import timm
from tqdm import tqdm
from opts import opts
from models.model import create_model, load_model, save_model
from pytorch_grad_cam.utils.image import show_cam_on_image,  preprocess_image
# Load the image you want to show
from PIL import Image

if __name__ == '__main__':
    opt = opts().parse()
    opt.heads = {'hm': 3, 'wh': 2, "reg": 2}
    opt.load_model = "/home/server/xcg/CenterNet/exp/ctdet/ctdet_coco_swint_2/model_last-2.pth"
    opt.exp_id = "fm_cam"
    opt.arch = "swint"
    # opt.keep_res = True

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # print(model)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # create efficientnet model
    model = models.resnet18(pretrained=True)

    # show which layer we can visualize
    show_network(model)

    # create visualization object
    visualized_layer = 'layer4.0.bn1'
    display = Display(model, visualized_layer, target_output=0, img_size=(992, 544))

    image_path = '/home/server/xcg/CenterNet/data/test/images/'
    # # print(image_path, os.listdir(image_path))
    # pabr = tqdm(total=len(os.listdir(image_path)), desc=f"{args.method}-process", unit="cls_id")
    # for data_cls in ['fuwo', "cewo", "zhanli"]:
    #     save_cls_dir = f"/home/server/xcg/CenterNet/exp/fm_cam/{args.method}/" + data_cls
    #     os.makedirs(save_cls_dir, exist_ok=True)
    #
    # for cls in os.listdir(image_path):
    #     pabr.update(1)
    for im in glob.glob(image_path+"/*.jpg"):
        #print(im)
        image = Image.open(im).resize((992, 544))
        display.save(image)