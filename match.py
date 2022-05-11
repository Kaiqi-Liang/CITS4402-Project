from curses import window
from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
import matplotlib.pyplot as plt

def candidate_match_discrimination(parser: Parser, candidate_small_objects: list[np.ndarray]):
	res = []
	true_positive = 0
	false_positive = 0
	false_negative = 0
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
	area_lower, area_upper = [1, 1000]
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
				box_pred = property.bbox
				area_predict = (max_row - min_row) * (max_col - min_col)
				res.append([binary_image[min_row:max_row, min_col:max_col], centroid])

	#intersection over union
	width, height = parser.get_gt(1)[0][-2:]
	area = width * height
	return res

def bb_intersection_over_union(box_pred: list[int], box_gt: list[int]):

	predict_xmin, predict_ymin, predict_xmax, predict_ymax = box_pred

	gt_xmin, gt_ymin, gt_xmax, gt_ymax = box_gt
	
	#detemine x-y coords of the intersection rectangle 

	


	interArea = abs(max(xB-xA, 0)) max(0, xB - xA) * max(0, yB - yA)

	box_predictArea = (predict[2] - predict[0] + 1) * (predict[3] - predict[1] + 1)
	box_gtArea = (box_gt[2] - box_gt[0] + 1) * (box_gt[3] - box_gt[1] + 1)

	iou = interArea / float(box_predictArea + box_gtArea - interArea)

	return iou

def accuracy(res: list[np.ndarray]):
	res = []
	for binary_image, centroid, gt_boxes in res:
		for box_gt in gt_boxes:
			cal_iou = bb_intersection_over_union(binary_image, box_gt)
			if cal_iou >= 0.7:
				true_positive += 1
				res.append([binary_image, centroid])
			elif binary_image and cal_iou <= 0.7 :
				false_positive += 1
			elif box_gt and cal_iou <= 0.7 :
				false_negative += 1
	precision = true_positive / (true_positive + false_positive)
	print(precision)
	recall = true_positive / (true_positive + false_negative)		
	print(recall)
	return res	