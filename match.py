from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
def candidate_match_discrimination(candidate_small_objects: list[np.ndarray]):
	for object in candidate_small_objects:
		labelled_image = skimage.measure.label(object)
		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			row, col = property.centroid
			row = round(row)
			col = round(col)
			window = object[max(row - 5, 0):min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]
			window_mean = np.average(window)
			window_std = np.std(window)
			upper_th = sp.stats.norm.ppf(0.995, loc=window_mean, scale=window_std)
			lower_th = sp.stats.norm.ppf(0.005, loc=window_mean, scale=window_std)
			# if lower_th <= pix_val <= upper_th:
				#reclassify as a candidate pixel 
			print(upper_th)

