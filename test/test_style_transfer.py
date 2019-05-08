#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '../style_transfer')

from utils import Utils
from models.vgg_transfer import *
from models.segmentation import *
from models.fast_style_transfer import *
import argparse
from PIL import Image
import time
import skimage.io
from pathlib import Path
import numpy as np
import sys
import style_transfer
import os
import re

def test_main():
    args = argparse.Namespace()

    # For Fast Style Transfer

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
    # calling the stylize now to create a new result now
    style_transfer.stylize(args)
    
    path = '../style_transfer/outputs/'
    dirs = os.listdir( path )
    exists = None
    for file in dirs:
        if re.search("results", file):
            exists = True
            break
    assert(not exists), "The test case passed"

if __name__ == "__main__":
    test_main()
