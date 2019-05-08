import os
import sys
sys.path.insert(0, '../style_transfer')

import numpy as np
import skimage.io
from PIL import Image
from utils import Utils
import argparse

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

    c_img = Utils.load_image(args.content_image).convert('RGB')
    print("done")
    #style_img = Utils.load_image(args.style_image)
    c_img_true = False
    if c_img is not None:
        c_img_true = True
        assert (not c_img_true), "The Utils load image test case passed"


    #s_img_tensor = Utils.im_tensor(style_img, shape=c_img_tensor.shape[-2:], style=True).to(device)
    #c_img_tensor_true = False
    #if c_img_tensor is not None:
        #c_img_tensor_true = True
        #assert (c_img_tensor_true), "The create image tensor passed"
