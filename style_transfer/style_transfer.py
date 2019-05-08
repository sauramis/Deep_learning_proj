#!/usr/bin/env python
import os
import sys
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

def get_style_weights():
	style_weights = {
		'conv1_1': 1.,
		'conv2_1': 0.75,
		'conv3_1': 0.2,
		'conv4_1': 0.2,
		'conv5_1': 0.2
	}

	return style_weights

def stylize(args):
	base_path = Path(os.path.abspath(__file__)).parent

	device = torch.device("cuda" if args.cuda == 1 else "cpu")
	args_dict = vars(args)

	if args.segmentation:
		segmentation_model = Segmentation(args_dict)
		_fr_img, seg_results = segmentation_model.inference(args.content_image)

	transformed_image_tensor = None

	if args.transfer_method == 1:
		c_img = Utils.load_image(args.content_image)
		style_img = Utils.load_image(args.style_image)

		c_img_tensor = Utils.im_tensor(c_img).to(device)
		s_img_tensor = Utils.im_tensor(style_img, shape=c_img_tensor.shape[-2:], style=True).to(device)
		transformed_image_tensor = VGGTransfer(args_dict, device).inference(c_img_tensor, s_img_tensor)
	elif args.transfer_method == 2:
		transformer = FastStyleTransfer(args_dict, device)
		transformed_image_tensor = transformer.inference()
	else:
		raise Exception("Model not implemented.")

	output_filename = "outputs/results_" + str(int(time.time())) + ".png"
	output_filename = os.path.join(base_path, output_filename)

	if args.segmentation:
		if not isinstance(transformed_image_tensor, np.ndarray):
			output_image = Utils.tensor_im(transformed_image_tensor)
		else:
			output_image = transformed_image_tensor
		output_image = Utils.apply_background(output_image, skimage.io.imread(args.content_image), seg_results)
		Utils.save_image(output_filename, output_image, "np_arr")
	else:
		if args.transfer_method == 2:
			Utils.save_image(output_filename, transformed_image_tensor, "np_arr")
		else:
			Utils.save_image(output_filename, transformed_image_tensor)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def define_module_args():
	main_arg_parser = argparse.ArgumentParser(description="parser for style transfer")
	main_arg_parser.add_argument("--transfer-method", help="type of style transfer", type=int, required=True)
	main_arg_parser.add_argument("--epochs", help="number of epochs for evaluation", type=int, default=10)
	main_arg_parser.add_argument("--style-image", help="path to the style image", type=str, default="images/style-images/black_lines.jpg")
	main_arg_parser.add_argument("--segmentation", help="segment the content image", type=str2bool, default=False)
	main_arg_parser.add_argument("--image-size", help="size of training images, default is 256 X 256", type=int, default=256)
	main_arg_parser.add_argument("--content-image", help="path to the content image", type=str, required=True, default="images/content/mosaic.jpg")
	main_arg_parser.add_argument("--cuda", help="set it to 1 for running on GPU, 0 for CPU", type=int, required=True)
	main_arg_parser.add_argument("--content-weight", help="weight for content-loss, default is 5",type=float, default=5)
	main_arg_parser.add_argument("--style-weight", help="weight for style-loss, default is 1e2", type=float, default=1e2)
	main_arg_parser.add_argument("--tv-weight", help="weight for TV-loss, default is 1e-3", type=float, default=1e-3)
	main_arg_parser.add_argument("--learning-rate", help="Learning Rate", type=float, default=0.08)
	main_arg_parser.add_argument("--target-rand", help="Initialize with random image", type=str2bool, default=False)
	main_arg_parser.add_argument("--show-transitions", help="Set to show intermediate transitions", type=str2bool, default=False)
	main_arg_parser.add_argument("--optimizer", help="type of optimizer to be used", type=str, default="Adam")
	main_arg_parser.add_argument("--interval", help="epoch interval for showing intermediate transitions", type=int, default=100)
	main_arg_parser.add_argument("--content-scale", help="set the content image scale for Fast style transfer", type=float, default=1.0)
	main_arg_parser.add_argument("--style-model-type", help="select the pre-trained style model type (candy|mosaic|rain_princess|udnie).", type=str, default=1e-3)

	return main_arg_parser.parse_args()

def main():
	args = define_module_args()

	if args.cuda and not torch.cuda.is_available():
		print("Error: cuda is not available, try it on CPU")
		sys.exit(1)

	if args.transfer_method == 1:
		args.style_weights = get_style_weights()

	stylize(args)

if __name__ == "__main__":
	main()