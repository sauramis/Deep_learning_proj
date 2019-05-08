import torch
import torch.optim as optim
from torchvision import transforms, models
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import skimage.io

from utils import Utils

class VGGTransfer(object):
	def __init__(self, args, device):
		self.model = models.vgg19(pretrained = True).features
		self.device = device
		self.model.to(device)
		self.style_weights = args["style_weights"]
		self.content_weight = args["content_weight"]
		self.style_weight = args["style_weight"]
		self.tv_weight = args["tv_weight"]
		self.target_init_rand = args["target_rand"]
		self.epochs = args["epochs"]
		self.learning_rate = args["learning_rate"]
		self.show_transistions = args["show_transitions"]
		self.optimizer_type = args["optimizer"]
		self.interval = args["interval"]

	def set_optimizer(self, optimizer_type, target):
		if optimizer_type == "Adam":
			return optim.Adam([target], lr=self.learning_rate)
		else:
			raise Exception("Optmizer type not implemented.")

	def inference(self, c_image, s_image):
		img_features = self.extract_features(c_image)
		style_features = self.extract_features(s_image)

		style_grams = {
			layer: self.gram_matrix(style_features[layer]) for layer in style_features
		}

		target_img = self.init_target(c_image)
		optimizer = self.set_optimizer(self.optimizer_type, target_img)

		for iter_ in range(1, self.epochs + 1):
			target_features = self.extract_features(target_img)

			content_loss = self.content_loss(target_features, img_features)
			cumm_style_loss = 0

			for layer in self.style_weights:
				feature = target_features[layer]
				feature_gram = self.gram_matrix(feature)
				style_gram = style_grams[layer]

				layer_style_loss = self.style_weights[layer] * torch.mean((feature_gram - style_gram) ** 2)
				_, d, h, w = feature.shape
				cumm_style_loss += layer_style_loss / (d * h * w)

			target_image = Utils.tensor_im(target_img)
			t_loss = self.tv_loss(torch.from_numpy(target_image))
			
			total_loss = self.content_weight * content_loss + self.style_weight * cumm_style_loss + self.tv_weight * t_loss
			optimizer.zero_grad()
			total_loss.backward(retain_graph=True)
			optimizer.step()

			if self.show_transistions:
				if iter_ % self.interval == 0:
					plt.imshow(target_image)
					plt.axis('off')
					plt.show()

		return target_img

	def init_target(self, c_image):
		if self.target_init_rand:
			return c_image.copy()
		else:
			return c_image.clone().requires_grad_(True).to(self.device)

	def gram_matrix(self, im_tensor):
		"""
		Gram matrix content information is eliminated while style information is retained.
		"""
		_, d, h, w = im_tensor.size()
		im_tensor = im_tensor.view(d, h * w)
		gram = torch.mm(im_tensor, im_tensor.t())

		return gram

	def tv_loss(self, result_img):
		return torch.tensor(( 1 * (
			torch.sum(torch.abs(result_img[:, :, :-1] - result_img[:, :, 1:])) + 
			torch.sum(torch.abs(result_img[:, :-1, :] - result_img[:, 1:, :]))
		)), dtype = torch.float)

	def content_loss(self, target_features, content_features):
		return torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)

	def extract_features(self, image):
		# Use same layers as used in the Style Transfer paper
		# 21 for content and rest for style as having more features will give a complete representation of the style
		ext_features = {}

		for name, layer in self.model._modules.items():
			image = layer(image)

			if name in self.required_layers():
				ext_features[self.required_layers()[name]] = image
				
		return ext_features

	def required_layers(self):
		"""
		Maintains the mapping of the layers outputs to extract.
		Based on https://arxiv.org/abs/1508.06576
		"""
		return {
			"0": "conv1_1",
			"5": "conv2_1",
			"10": "conv3_1",
			"19": "conv4_1",
			"21": "conv4_2",
			"28": "conv5_1"
		}
