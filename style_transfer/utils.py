import numpy as np
import skimage.io
import torch
from torchvision import transforms
from PIL import Image

class Utils(object):
	
	@staticmethod
	def apply_background(style_image, org_img, seg_results):
		result_roi = seg_results["rois"]
		n_channel = image.shape[:3]
		image_out = org_img.copy()

		for iter_ in range(result_roi.shape[0]):
			mask = seg_results["masks"][:, :, iter_]

			for c_iter in range(n_channel):
				image_out[:, :, c_iter] = np.where(mask == 0, image_out[:, :, c_iter], style_image[:, :, c_iter])

		return image_out

	@staticmethod
	def load_image(image_path):
		return Image.open(image_path).convert('RGB')

	@staticmethod
	def im_tensor(image, max_size = 400, shape = None, style=False):
	    if style:
	        in_transform = transforms.Compose([
	          transforms.Resize(shape),  # maintains aspect ratio
	          transforms.ToTensor(),
	          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	      ])
	    else:
	        in_transform = transforms.Compose([
	          transforms.ToTensor(),
	          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	      ])

	    image_tensor = in_transform(image).unsqueeze(0)

	    return image_tensor

	@staticmethod
	def tensor_im(img_tensor):
		image = img_tensor.cpu().clone().detach().numpy()
		image = image.squeeze()
		image = image.transpose(1, 2, 0)
		image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
		image = image.clip(0, 1)
		
		return image

	@staticmethod
	def save_image(filename, data):
	    img = img.transpose(1, 2, 0).astype("uint8")
	    img = Image.fromarray(img)
	    img.save(filename)

