import cv2
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import yaml

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .lib.coco import coco
from .lib.mrcnn import utils
from .lib.mrcnn import model as modellib


class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


class Segmentaion(object):
	def __init__(self, args, verbosity=1, download=False):
		self.log_path = args.log_path
		self.weights_path = args.weight_path

		if download:
			utils.download_trained_weights(self.weights_path)

		self.config = InferenceConfig()
		self.model = self.initialize_model()
		self.class_names = self.load_class_names()
		self.verbosity = verbosity

	def load_class_names(self):
		class_names = []

		with open("./data/coco_class_names.yaml") as file_:
			class_names = yaml.safe_load(file_)

		return class_names

	def initialize_model(self):
		model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir = self.log_path)
		model.load_weights(self.weights_path, by_name=True)

		return model

	def load_image(self, image_path):
		return skimage.io.imread(image_path)

	def inference(self, image_path):
		image = self.load_image(image_path)
		result = self.model.detect([image.copy()], verbose=self.verbosity)

		f_img = self.extract_foreground(result, image.copy())

		return f_img, image, result[0]

	def extract_foreground(self, result, image):
		f_img = np.zeros_like(image)
		result_roi = result["rois"]
		(_, _, n_channel) = image.shape[:3]

		for iter_ in range(result_roi.shape[0]):
			mask = result["masks"][:, :, iter_]
			for c_iter in range(n_channel):
				f_img[:, :, c_iter] = np.where(mask == 1, image[:, :, c_iter], f_img[:, :, c_iter])

		return f_img
