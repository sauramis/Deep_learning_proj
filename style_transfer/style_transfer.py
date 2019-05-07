#!/usr/bin/env python
import os
import sys
from utils import Utils
from models.vgg_transfer import *
from models.segmentation import *
import argparse
from PIL import Image
from datetime import datetime

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
	device = torch.device("cuda" if args.cuda == 1 else "cpu")

	if args.segmentation:
		segmentation_model = Segmentation(args)
		_fr_img, seg_results = segmentation_model.inference(args.content_image)

	c_img = Utils.load_image(args.content_image)
	style_img = Utils.load_image(args.style_image)

	c_img_tensor = Utils.im_tensor(c_img).to(device)
	s_img_tensor = Utils.im_tensor(style_img, shape=c_img_tensor.shape[-2:], style=True).to(device)
	args_dict = vars(args)
	transformed_image_tensor = VGGTransfer(args_dict, device).inference(c_img_tensor, s_img_tensor)

	if args.segmentation:
		output_image = Utils.tensor_im(transformed_image_tensor)
		output_image = Utils.apply_background(output_image, Utils.load_image(args.content_image), seg_results)
		Utils.save_image('./results-'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png', output_image, "np_arr")
	else:
		Utils.save_image('./results-'+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'.png', transformed_image_tensor)

def define_module_args():
	main_arg_parser = argparse.ArgumentParser(description="parser for style transfer")
	subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")
	model_arg_parser = subparsers.add_parser("model", help="parser for model arguments")
	model_arg_parser.add_argument("--method", help="type of style transfer", type=str, required=True, default="original")
	model_arg_parser.add_argument("--epochs", help="number of epochs for evaluation", type=int, default=10)
	model_arg_parser.add_argument("--log-path", help="path to log directory",type=str, required=True)
	model_arg_parser.add_argument("--style-image", help="path to the style image",
		type=str, required=True, default="images/style-images/mosaic.jpg"
	)
	model_arg_parser.add_argument("--segmentation", help="segment the content image",
		type=bool, default=False
	)
	model_arg_parser.add_argument("--image-size", help="size of training images, default is 256 X 256",
		type=int, default=256
	)
	model_arg_parser.add_argument("--content-image", help="path to the content image",
		type=str, required=True, default="images/content/mosaic.jpg"
	)
	model_arg_parser.add_argument("--output-image", help="path for saving the output image",
		type=str, required=True
	)
	model_arg_parser.add_argument("--cuda", help="set it to 1 for running on GPU, 0 for CPU", 
    	type=int, required=True
	)
	model_arg_parser.add_argument("--content-weight", help="weight for content-loss, default is 1e5",
    	type=float, default=1e5
  	)
	model_arg_parser.add_argument("--style-weight", help="weight for style-loss, default is 1e10", 
  		type=float, default=1e10
	)

	return main_arg_parser.parse_args()

def main():
	# args = define_module_args()

	# if args.subcommand is None:
	# 	print("Error: specify model")
	# 	sys.exit(1)

	# if args.cuda and not torch.cuda.is_available():
	# 	print("Error: cuda is not available, try it on CPU")
	# 	sys.exit(1)
		
	args = argparse.Namespace()
	args.content_weight = 5
	args.style_weight = 1e2
	args.tv_weight = 1e-3
	args.target_rand = False
	args.epochs = 1000
	args.learning_rate = 0.08
	args.show_transitions = True
	args.optimizer = 'Adam'
	args.interval = 100
	args.content_image = '/content/content-sample.jpg'
	args.style_image = '/content/black_lines.jpg'
	args.segmentation = True
	args.cuda = 1
	args.style_weights = get_style_weights()

	stylize(args)

if __name__ == "__main__":
	main()