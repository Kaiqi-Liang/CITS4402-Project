from kalman import filter_init
from parser import Parser
import cv2
import scipy as sp 
import numpy as np
import skimage
import matplotlib.pyplot as plt

def candidate_match_discrimination(parser: Parser, candidate_small_objects: list[np.ndarray]):
	pred_res = []
	gt_res = []
	true_positive = 0
	false_positive = 0
	false_negative = 0
	for binary_image, gray_image, _ in candidate_small_objects:
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
	area_lower, area_upper = [2, 11]
	extent_lower, extent_upper = [0.7, .95]
	maxis_lower, maxis_upper = [2, 7]
	eccentricity_lower, eccentricity_upper = [0.7, 0.9]
	for binary_image, _, gt in candidate_small_objects:
		# label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)

		properties = skimage.measure.regionprops(labelled_image)
		for property in properties:
			area = property.area
			extent = area / property.area_bbox
			maxis = property.axis_major_length
			eccentricity = property.eccentricity
			if area_lower <= area <= area_upper and extent_lower <= extent <= extent_upper and maxis_lower <= maxis <= maxis_upper and eccentricity_lower <= eccentricity <= eccentricity_upper:
				centroid = property.centroid
				min_row, min_col, max_row, max_col = property.bbox
				pred_res.append(((min_row, min_col, max_row, max_col), centroid))

		# intersection over union
		for gt_box in gt:
			topleftx, toplefty, width, height = gt_box[-4:]
			min_row = toplefty
			min_col = topleftx
			max_row = min_row + height
			max_col = min_col + width 
			gt_res.append((min_row, min_col, max_row, max_col))

	max_track_id = gt[-1][0]
	match = False
	for hypothesis in pred_res:
		for gt_track in gt_res:
			if (bb_intersection_over_union(hypothesis[0], gt_track) >= 0.7):
				print(hypothesis[0], gt_track)
				match = True
		if not match:
			max_track_id += 1
			filter_init(hypothesis, max_track_id)
	return pred_res

def bb_intersection_over_union(box_pred: list[int], box_gt: list[int]):
	#determine x-y coords of the intersection rectangle 
	xA = max(box_pred[0], box_gt[0]) # min row 
	yA = max(box_pred[1], box_gt[1]) # min col
	xB = min(box_pred[2], box_gt[2]) # max row
	yB = min(box_pred[3], box_gt[3]) # max col 

	# compute the area of intersection rectangle
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0

 	# compute the area of both the prediction and ground-truth
    # rectangles
	box_predArea = abs((box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1]))	
	box_gtArea = abs((box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1]))
	
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
	iou = interArea / float(box_predArea + box_gtArea - interArea)

	return iou

# calculate measure of quality of predicted candidate clusters
def accuracy(pred_res: list[np.ndarray], gt_res: list[np.ndarray]):
	pred_res1 = []
	gt_res1 = []
	for binary_image, centroid in pred_res:
		for box_gt in gt_res:
			cal_iou = bb_intersection_over_union(binary_image, box_gt)
			# if true positive append the binary image
			if cal_iou >= 0.7:
				true_positive += 1
				pred_res1.append([binary_image, centroid])
				gt_res1.append([box_gt])
			# if false positive
			elif binary_image and cal_iou <= 0.7 :
				false_positive += 1
			# if false negative
			elif box_gt and cal_iou <= 0.7 :
				false_negative += 1
		if len(pred_res) == 5 :
			break
		print(pred_res)
		print(gt_res)
	# calculate precision and recall
	precision = true_positive / (true_positive + false_positive)
	print(precision)
	recall = true_positive / (true_positive + false_negative)		
	print(recall)
	F1 = 2 * ((precision*recall)/(precision+recall))
	print(F1)
	return pred_res	