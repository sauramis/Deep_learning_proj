#!/usr/bin/env python
import os
import sys
from utils import Utils

def stylize(args):
	device = torch.device("cuda" if args.cuda == 1 else "cpu")

	if args.segmentation():
		segmentation_model = models.Segmentation(args)
		c_img, org_img, seg_results = segmentation_model.inference(args.content_image)
	else:
		c_img = Utils.load_image(args.content_image)

	style_img = Utils.load_image(args.style_image)

	c_img_tensor = Utils.im_tensor(c_img).to(device)
	s_img_tensor = Utils.im_tensor(style_img, shape=c_img.shape[-2:], style=True).to(device)


def define_module_args():
    main_arg_parser = argparse.ArgumentParser(description="parser for style transfer")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    model_arg_parser = subparsers.add_parser("model", help="parser for model arguments")

    model_arg_parser.add_argument("--method", help="type of style transfer", 
    	type=str, required=True, default="original"
   	)
   	model_arg_parser.add_argument("--epochs", help="number of epochs for evaluation",
   		type=int, default=10
	)
	model_arg_parser.add_argument("--log-path", help="path to log directory",
   		type=str, required=True
	)
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
	args = define_module_args()

	if args.subcommand is None:
		print("Error: specify model")
		sys.exit(1)

	if args.cuda and not torch.cuda.is_available():
		print("Error: cuda is not available, try it on CPU")
		sys.exit(1)


	stylize(args)


if __name__ == "__main__":
	main()