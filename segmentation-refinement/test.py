import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
from PIL import Image
import numpy as np

image = cv2.imread('test/1.jpg')
mask = cv2.imread('test/1.png', cv2.IMREAD_GRAYSCALE)

image = Image.open('test/1.jpg')
mask = Image.open('test/1.png').convert('L')

image = np.asarray(image)
mask = np.asarray(mask)

print("loading")
# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0',model_folder='saved_models')# device can also be 'cpu'
print("infering")
# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False)
Image.fromarray(output).save('result.png')
print("finish")
# plt.imshow(output)
# plt.show()
