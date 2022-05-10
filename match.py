from curses import window
from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
import matplotlib.pyplot as plt

def candidate_match_discrimination(parser: Parser, candidate_small_objects: list[np.ndarray]):
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

	# Thresholding values
	area_lower, area_upper = [2, 1000]
	extent_lower, extent_upper = [0, 1]
	maxis_lower, maxis_upper = [0, 500]
	eccentricity_lower, eccentricity_upper = [0, 1]

	for binary_image, _ in candidate_small_objects:
		# label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)

		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			area = property.area
			extent = area / property.area_bbox
			maxis = property.axis_major_length
			eccentricity = property.eccentricity
			if area_lower <= area <= area_upper and extent_lower <= extent <= extent_upper and maxis_lower <= maxis <= maxis_upper and eccentricity_lower <= maxis <= eccentricity_upper:
				centroid = property.centroid
				(min_row, min_col, max_row, max_col) = property.bbox
				res.append((binary_image[min_row:max_row, min_col:max_col], centroid))

	#intersection over union
	width, height = parser.get_gt(1)[0][-2:]
	area = width * height
	return res

def bb_intersection_over_union(box_predict, box_gt):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(box_predict[0], box_gt[0])
	yA = max(box_predict[1], box_gt[1])
	xB = min(box_predict[2], box_gt[2])
	yB = min(box_predict[3], box_gt[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	box_predictArea = (box_predict[2] - box_predict[0] + 1) * (box_predict[3] - box_predict[1] + 1)
	box_gtArea = (box_gt[2] - box_gt[0] + 1) * (box_gt[3] - box_gt[1] + 1)

	iou = interArea / float(box_predictArea + box_gtArea - interArea)

	return iou