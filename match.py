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
			window = object[max(row - 5, 0):row + 5, max(col - 5, 0):col + 5]
			print(window)
