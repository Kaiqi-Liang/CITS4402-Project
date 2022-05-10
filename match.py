from curses import window
from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
import matplotlib.pyplot as plt

def candidate_match_discrimination(candidate_small_objects: list[np.ndarray]):	
	plt.imshow(candidate_small_objects[0][0], cmap='gray')
	plt.show()
	res = []
	for binary_image, gray_image in candidate_small_objects:
		# label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)
		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			# get centroid of pixels
			row, col = property.centroid
			row = round(row)
			col = round(col)
			# put 11 x 11 search window centered around centroid
			binary_window = binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]
			gray_window = gray_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]

			gray_binary = gray_window[binary_window]
			if len(gray_binary) <= 1:
				continue

			# find the pixel mean of window
			window_mean = np.average(gray_binary)
			# find standard deviation of window
			window_std = np.std(gray_binary)

			# get the upper quantile limit of candidate cluster
			upper_th = min(sp.stats.norm.ppf(0.995, loc=window_mean, scale=window_std), 255)
			# get the lower quantile limit of candidate cluster
			lower_th = sp.stats.norm.ppf(0.005, loc=window_mean, scale=window_std)
			# print(upper_th, lower_th)

			# mark the pixels within the quantile interval as being part of candidate cluster in object
			for i, pixels in enumerate(gray_window):
				for j, pixel in enumerate(pixels):
					if lower_th <= pixel <= upper_th:
						binary_window[i,j] = True
			binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)] = binary_window

	for binary_image, _ in candidate_small_objects:
		# label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)
		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			print(property.eccentricity)

	# res.append([object,object2])

	# Plotting "Region Grow Vs. Original Binary Image"
	# fig = plt.figure(figsize=(10, 7))
	# rows = 1
	# columns = 2

	plt.imshow(candidate_small_objects[0][0], cmap='gray')
	plt.show()


	# fig.add_subplot(rows, columns, 2)
	# plt.imshow(res[0][0], cmap='gray')
	# plt.title("ORIGINAL BINARY IMAGE")


	return res
			
					 
			

