'''
Candidate Match Discrimination: to clean up candidate moving objects and remove incorrect matches caused by imaging noise or small motion of the satellite 
'''
from parser import Parser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp 
import skimage
import cv2

def region_growing(frames):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a binary image representing the candidate small objects.
	Output: for each frame index n from 1 to N-1, this step outputs a binary image with candidate small object areas grown based on a 11x11 search window 
	'''

	# loop through the frames 
	for _, gray_image, binary_image, _ in frames:

		# identify and label connected regions in binary image
		labelled_image = skimage.measure.label(binary_image)
		clusters = skimage.measure.regionprops(labelled_image)

		#loop through the clusters identified in the binary image 
		for cluster in clusters:

			# centroid of cluster 
			row, col = cluster.centroid
			row = round(row)
			col = round(col)

			# 11 x 11 search window centred around centroid
			binary_window = binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]
			gray_window = gray_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)]

			# binary mask applied so only 'candidate pixels' are includes in the search window 
			gray_binary = gray_window[binary_window]

			# ignore any candidate areas with pixels less than 3. Two pixels is not sufficent to represent a car and is much more likely to be noise. Standard deviation based on one or two pixels is illogical.
			if len(gray_binary) < 3:
				continue

			# pixel mean of the window
			window_mean = np.average(gray_binary)
			# pixel standard deviation of the window
			window_std = np.std(gray_binary)

			# upper and lower quantile limit of candidate cluster based on fitting a normal distribution 
			upper_th = min(sp.stats.norm.ppf(0.995, loc=window_mean, scale=window_std), 255)
			lower_th = sp.stats.norm.ppf(0.005, loc=window_mean, scale=window_std)			

			# mark the pixels within the quantile interval as being part of candidate cluster in object
			for i, pixels in enumerate(gray_window):
				for j, pixel in enumerate(pixels):
					if lower_th <= pixel <= upper_th:
						binary_window[i,j] = True
			binary_image[max(row - 5, 0): min(row + 6, 1024), max(col - 5, 0): min(col + 6, 1024)] = binary_window

	plt.title('region growing')
	plt.imshow(frames[0][2], 'gray')
	plt.savefig('region_growing.jpg')
	return frames

def get_candidate_small_objects(frames):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a grown binary image, i.e. the output of region_growing()
	Output: for each frame index n from 1 to N-1, this step outputs a binary image with candidate small object areas grown based on a 11x11 search window 
	'''
	output = []
	for _, _, binary_image, _ in frames:
		candidate_small_objects = [] #candidate small objects in each frame
		# label connected regions in now 'grown' binary image
		labelled_image = skimage.measure.label(binary_image)

		clusters = skimage.measure.regionprops(labelled_image)
		for cluster in clusters:
			area = cluster.area
			extent = area / cluster.area_bbox
			major_axis = cluster.axis_major_length
			eccentricity = cluster.eccentricity
			candidate_small_objects.append((cluster.centroid, cluster.bbox, area, extent, major_axis, eccentricity))
		output.append(candidate_small_objects)
	return output

def candidate_match_discrimination(frames: list[np.ndarray], areaTh: tuple[float, float], extentTh: tuple[float, float], majorAxisTh: tuple[float, float], eccentricityTh: tuple[float, float]):
	'''
	Input: for each frame index n from 1 to N-1, this step takes as input a binary image representing the candidate small objects.
	Output: for each frame index n from 1 to N-1, this step outputs the bounding box and centroid of each candidate small object.
	'''
	output = []

	# Morphological Cues
	area_lower, area_upper = areaTh
	extent_lower, extent_upper = extentTh
	major_axis_lower, major_axis_upper = majorAxisTh
	eccentricity_lower, eccentricity_upper = eccentricityTh

	candidate_small_objects = get_candidate_small_objects(frames)
	for i, (original_image, _, _, frame) in enumerate(frames):
		image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		centroids = []
		for centroid, bbox, area, extent, major_axis, eccentricity in candidate_small_objects[i]:
			if area_lower <= area <= area_upper and extent_lower <= extent <= extent_upper and major_axis_lower <= major_axis <= major_axis_upper and eccentricity_lower <= eccentricity <= eccentricity_upper:
				centroids.append(centroid)
				min_row, min_col, max_row, max_col = bbox
				cv2.rectangle(image, (min_row, min_col), (max_row, max_col), (255, 0, 0), 2)
		plt.title(f'frame {frame}')
		plt.imshow(image)
		plt.savefig(f'{frame}.jpg')
		output.append(centroids)

	plt.figure()
	plt.xlabel('Frame')
	plt.ylabel('Number of moving objects detected')
	plt.plot([frame[-1] for frame in frames], [len(frame) for frame in output])
	plt.savefig('graph.jpg')
	return output

# Threshold Calibration
def thresholds_calibration(parser: Parser, frames):
	candidate_small_objects = get_candidate_small_objects(frames)
	matches = []
	for i, frame in enumerate(frames):
		gt = parser.get_gt(frame[-1])

		gt_res = []
		for gt_box in gt:
			topleftx, toplefty, width, height = gt_box[-4:]
			min_row = toplefty
			min_col = topleftx
			max_row = min_row + height
			max_col = min_col + width
			gt_res.append((min_row, min_col, max_row, max_col))

		for _, bbox, area, extent, major_axis, eccentricity in candidate_small_objects[i]:
			for gt_track in gt_res:
				if (bb_intersection_over_union(bbox, gt_track) >= 0.3):
					matches.append((area, extent, major_axis, eccentricity))

	df = pd.DataFrame(matches, columns=['area', 'extent', 'majorAxis', 'eccentricity'])
	for i, column in enumerate(df.columns):
		plt.figure()
		df[column].plot.density(title=column)
		plt.savefig(f'{column}.jpg')

def bb_intersection_over_union(box_pred: list[int], box_gt: list[int]):
	'''
	Input: list of bounding box coordinates of the predicted candidate cluster and the ground truth candidate cluster. List should be in this order = [min row, min col, max row, max col]
	Output: intersection of union metric, representing the quality of matching between the ground truth regions and predicted regions. Any IOU greater tham 0.3 is considered matching 
	'''

	# determine x-y coords of the intersection 
	xA = max(box_pred[0], box_gt[0]) # min row 
	yA = max(box_pred[1], box_gt[1]) # min col
	xB = min(box_pred[2], box_gt[2]) # max row
	yB = min(box_pred[3], box_gt[3]) # max col 

	# compute the area of intersection 
	interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
	if interArea == 0:
		return 0

 	# compute the area of both the prediction and ground-truth bounding boxes
	box_predArea = abs((box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1]))	
	box_gtArea = abs((box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1]))
	
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas 
	return interArea / float(box_predArea + box_gtArea - interArea)

# calculate measure of quality of predicted candidate clusters
def accuracy(pred_res: list[np.ndarray], gt_res: list[np.ndarray]):
	'''
	Input: list of bounding box coordinates of the predicted candidate cluster and the ground truth candidate cluster. List should be in this order = [min row, min col, max row, max col]
	Output: intersection of union metric, representing the quality of matching between the ground truth regions and predicted regions. Any IOU greater tham 0.3 is considered matching 
	'''
	pred_res1 = []
	gt_res1 = []
	true_positive = 0
	false_positive = 0
	false_negative = 0
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