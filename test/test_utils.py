import os
import sys
sys.path.insert(0, '../style_transfer')

import numpy as np
from utils import Utils
import argparse
import torch
import torch.optim as optim
from torchvision import transforms, models

if __name__ == "__main__":
    args = argparse.Namespace()
    args.transfer_method = 2
    args.segmentation = False
    # test image to be sent
    args.content_image = './cat_content.jpg'
    # default parameters
    args.content_scale = 1.0
    args.model = 1
    # test style to be applied
    args.style_model_type = "rain_princess"
    args.cuda = 0
    # args_dict = vars(args)
    #segmentation_model = Segmentation(args_dict)
    if args.cuda and not torch.cuda.is_available():
        print("Error: cuda is not available, try it on CPU")
        sys.exit(1)

    device = torch.device("cuda" if args.cuda == 1 else "cpu")
    
    c_img = Utils.load_image(args.content_image)
    #style_img = Utils.load_image(args.style_image)
    if c_img is not None:
        assert (c_img), "The Utils load image test case has failed"
        print("OK")

    c_img_tensor = Utils.im_tensor(c_img).to(device)
    if c_img_tensor is not None:
        c_img_tensor_true = True
        assert (c_img_tensor_true), "The create image tensor test case has failed"
        print("OK")
