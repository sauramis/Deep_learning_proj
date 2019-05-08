# Image Segmentation and Style Transfer

This is a project for the course _CSCI-5922 Deep Learning and Neural Networks_ offered at _UCB_. 

* We have implmennted a library for performing Neural Style Transfer using the optimization technique discussed in [Gatys paper](https://arxiv.org/abs/1508.06576) but with the functionality of performing onky on foreground object in the image. See examples below 

  <img width="200" alt="Epoch 200" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/bride_200.jpg"> <img width="200" alt="Epoch 600" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/bride_600.jpg"> <img width="200" alt="Epoch 1000" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/bride_1000.jpg"> <img width="300" alt="Epoch 1400" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/bride_1400.jpg"> <img width="300" alt="Epoch 1800" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/bride_1800.jpg">

  To achieve this we have implemented the Optimization technique discussed in the Gatys paper mentioned above along with the addition of Total Variation loss which makes the results more smoother and the textures appear to transitiion more smoothly. For the style transfer only on the foreground object(in the above case the cat), we have extended the MR-CNN segmentation framework implemented by [Waleed Abdulla](https://github.com/matterport/Mask_RCNN). 
  
* Fast Style Transfer as discussed by [Justin Johnson, Alexandre Alahi, Li Fei-Fei](https://arxiv.org/abs/1603.08155) which uses perceptual loss function which is composed of the losses extracted from several layers of a VGG-Net trained on image classification task. The gradients to this loss is passed to a transformation network (decribed in details in the paper). The outputs of segmentation with with Fast style transfer.

  <img width="350" alt="Original Cat" src="https://github.com/sauramis/Style-transfer-library/blob/master/images/cat_content.jpg"> <img width="350" alt="With Candy" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/cat_seg_result_1.png"> <img width="350" alt="With Princess Dream" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/cat_seg_result_3.png"> <img width="350" alt="With Mosaic" src="https://github.com/sauramis/Style-transfer-library/blob/master/style_transfer/outputs/cat_seg_result_2.png">

## Setup
  ``` 
  pip install -r requirements.txt 
  ```
  
## Usage

* To run the fast style transfer with segmentation use the following command:
  ```
  cd Style-transfer-library/ && python style_transfer/style_transfer.py --transfer-method 2 --segmentation True --content-image images/cat_content.jpg --content-scale 1.0 --style-model-type rain_princess --cuda 1
  ```
* To run the original VGG style transfer use the following command:
  ```
  cd Style-transfer-library/ && python style_transfer/style_transfer.py --epochs 1000 --optimizer Adam --content-image images/cat_content.jpg --style-image images/style-images/black_lines.jpg --segmentation False --cuda 1 --transfer-method 1
  ```
