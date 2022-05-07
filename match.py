from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
def candidate_match_discrimination(candidate_small_objects: list[np.ndarray]):	
	for object, gray_image in candidate_small_objects:	
		# label connected regions in binary image	
		labelled_image = skimage.measure.label(object)
		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			# get centroid of pixels
			row, col = property.centroid
			row = round(row)
			col = round(col)
			# put 11 x 11 search window centered around centroid
			window = gray_image[max(row - 5, 0):min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]
			# find the pixel mean of window
			window_mean = np.average(window)
			# find standard deviation of window
			window_std = np.std(window)
			# get the upper quantile limit of candidate cluster
			upper_th = sp.stats.norm.ppf(0.995, loc=window_mean, scale=window_std)
			# get the lower quantile limit of candidate cluster
			lower_th = sp.stats.norm.ppf(0.005, loc=window_mean, scale=window_std)
			# mark the pixels within the quantile interval as being part of candidate cluster in object
			for i in range(max(row-5,0), min(row+6)):
				for j in range(max(col - 5, 0), min(col + 6)):
					if lower_th <= object[i,j] <= upper_th:
						object[i,j] = 1
				
					 
			

