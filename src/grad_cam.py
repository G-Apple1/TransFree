import argparse
import cv2
import numpy as np
import torch
import glob
import os
import timm
from tqdm import tqdm
from opts import opts
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from models.model import create_model, load_model, save_model
from pytorch_grad_cam.utils.image import show_cam_on_image,  preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', #default='--use-cuda',#False
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/server/xcg/CenterNet/data/test/images/',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    height = 17
    width = 31
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    opt = opts().parse()
    opt.heads = {'hm': 3, 'wh': 2, "reg": 2}
    opt.load_model = "/home/server/xcg/CenterNet/exp/ctdet/ctdet_coco_swint_2/model_last-2.pth"
    opt.exp_id = "fm_cam"
    opt.arch = "swint"
    # opt.keep_res = True

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.937, nesterov=True)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layer = [model.backbone.bottom_up.layers[-1].blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,target_layers=target_layer, use_cuda=args.use_cuda,reshape_transform=reshape_transform)

    image_path = args.image_path #""
    # print(image_path, os.listdir(image_path))
    pabr = tqdm(total=len(os.listdir(image_path)), desc=f"{args.method}-process", unit="cls_id")
    for data_cls in ['fuwo', "cewo", "zhanli"]:
        save_cls_dir = f"/home/server/xcg/CenterNet/exp/fm_cam/{args.method}/" + data_cls
        os.makedirs(save_cls_dir, exist_ok=True)

    for cls in os.listdir(image_path):
        pabr.update(1)
        for im in glob.glob(image_path+"/*.jpg"):
            #print(im)
            rgb_img = cv2.imread(im, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img,(992,544))
            #rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img)#, mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None#int(cls)##

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 128

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, 0.8)
            im_name = os.path.basename(im)
    #         cv2.imwrite(args.image_path + f"piglet_pose_cls_{args.method}/" + str(cls) +f'/{im_name[:-4]}_{args.method}.jpg', cam_image)
    # pabr.close()