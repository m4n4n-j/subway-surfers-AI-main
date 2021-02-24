# Function takes in an image and returns the grayscaled low resolution version of the image back.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from skimage.transform import resize

# Function to make image grayscale and resize to 128 X 128, then return this preprocessed image.
def preprocess_image(img):
	img_size = (128,128,3)
	img = resize(np.array(img), img_size)
	img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114]) #Source - https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
	img_gray = resize(img_gray, (128, 128))
	return np.expand_dims(img_gray, axis=0)





# image = img.imread('images/temp1.png')
# s_g = preprocess_image(image)
# plt.figure(figsize=(12,8))
# plt.imshow(s_g, cmap=plt.get_cmap('gray'))
# plt.axis('off')
# plt.show()